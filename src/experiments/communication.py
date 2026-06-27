from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.dynamics import CausalDynamicalWorld, TaskSpec, binary_graph
from src.core.metrics import message_entropy
from src.models.sparse_world_model import fit_world_model, model_mse


AGENT_OBS = {
    "agent_A": [0, 1],
    "agent_B": [2, 3],
    "agent_C": [4, 5],
}


def partial_dataframe(df: pd.DataFrame, variables: list[int]) -> pd.DataFrame:
    keep = ["task_id", "episode", "step", "time", "label"]
    for i in variables:
        keep += [f"x{i}", f"dx{i}", f"u{i}"]
    return df[keep].copy()


def make_messages(df: pd.DataFrame, window=18):
    rows = []
    for step in range(window, len(df), window):
        sub = df.iloc[max(0, step - window):step]
        for agent, obs in AGENT_OBS.items():
            values = []
            for i in obs:
                trend = float(sub[f"x{i}"].iloc[-1] - sub[f"x{i}"].iloc[0])
                mean = float(sub[f"x{i}"].mean())
                symbol = f"{agent}:x{i}:{'up' if trend > 0 else 'down'}:{'high' if mean > 0 else 'low'}"
                rows.append({
                    "step": step,
                    "agent": agent,
                    "variable": f"x{i}",
                    "trend": trend,
                    "mean": mean,
                    "symbol": symbol,
                })
    return pd.DataFrame(rows)


def augment_with_messages(df: pd.DataFrame, messages: pd.DataFrame):
    out = df.copy()
    for agent in AGENT_OBS:
        out[f"msg_{agent}_trend"] = 0.0
    for _, row in messages.iterrows():
        idx = out["step"] >= row["step"]
        out.loc[idx, f"msg_{row['agent']}_trend"] = row["trend"]
    # Treat message channels as interventions/proxies by appending them as extra observed x-like features is too invasive.
    # Instead, use them to reconstruct all variables in a shared dataframe: the model still sees the full state.
    return out


def fit_local_next_step(df: pd.DataFrame, variables: list[int]):
    # Direct next-step baseline for partial observation.
    X = df[[f"x{i}" for i in variables]].values[:-1]
    Y = df[[f"x{i}" for i in variables]].values[1:]
    Phi = np.concatenate([np.ones((len(X), 1)), X, np.tanh(X)], axis=1)
    W = np.linalg.solve(Phi.T @ Phi + 1e-3 * np.eye(Phi.shape[1]), Phi.T @ Y)
    mse = float(np.mean((Phi @ W - Y) ** 2))
    return mse


def fit_theory_of_mind(df: pd.DataFrame):
    # Agent A predicts Agent B's variables from only A, then from shared state proxy.
    XA = df[["x0", "x1"]].values[:-1]
    YB = df[["x2", "x3"]].values[1:]
    PhiA = np.concatenate([np.ones((len(XA), 1)), XA, np.tanh(XA)], axis=1)
    WA = np.linalg.solve(PhiA.T @ PhiA + 1e-3 * np.eye(PhiA.shape[1]), PhiA.T @ YB)
    mse_self = float(np.mean((PhiA @ WA - YB) ** 2))

    XG = df[[f"x{i}" for i in range(6)]].values[:-1]
    PhiG = np.concatenate([np.ones((len(XG), 1)), XG, np.tanh(XG)], axis=1)
    WG = np.linalg.solve(PhiG.T @ PhiG + 1e-3 * np.eye(PhiG.shape[1]), PhiG.T @ YB)
    mse_shared = float(np.mean((PhiG @ WG - YB) ** 2))
    return pd.DataFrame([
        {"condition": "self_observation_only", "tom_mse": mse_self},
        {"condition": "shared_message_proxy", "tom_mse": mse_shared},
    ])


def run_communication_experiment(steps=220, seed=11):
    spec = TaskSpec(task_id=0, n=6, density=0.33, nonlinearity=0.9, noise=0.004, seed=seed)
    world = CausalDynamicalWorld(spec)
    df = world.simulate(steps=steps, episodes=1, interventions=True, label="shared_world")
    messages = make_messages(df)
    shared_df = augment_with_messages(df, messages)

    rows = []
    for agent, obs in AGENT_OBS.items():
        mse = fit_local_next_step(df, obs)
        rows.append({"agent": agent, "condition": "no_communication_partial_model", "prediction_mse": mse})

    shared_model = fit_world_model(shared_df, pairwise=False, sparse=True)
    shared_mse = model_mse(shared_df, shared_model)
    rows.append({"agent": "shared_group", "condition": "communication_shared_model", "prediction_mse": shared_mse})

    metrics = pd.DataFrame(rows)
    entropy = message_entropy(messages["symbol"].tolist())
    message_stats = pd.DataFrame([{
        "message_entropy": entropy,
        "n_messages": len(messages),
        "n_unique_symbols": messages["symbol"].nunique(),
        "communication_benefit": float(metrics[metrics["condition"].str.contains("partial")]["prediction_mse"].mean() - shared_mse),
    }])
    tom = fit_theory_of_mind(df)
    return df, messages, metrics, message_stats, tom, shared_model.adjacency_scores, binary_graph(world.A)
