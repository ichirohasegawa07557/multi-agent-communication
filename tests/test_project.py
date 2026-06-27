from src.core.dynamics import TaskSpec, CausalDynamicalWorld
from src.experiments.communication import run_communication_experiment, make_messages
from src.models.sparse_world_model import fit_world_model

def test_world_simulates():
    world = CausalDynamicalWorld(TaskSpec(task_id=0))
    df = world.simulate(steps=20, episodes=1)
    assert len(df) == 20

def test_messages():
    world = CausalDynamicalWorld(TaskSpec(task_id=0))
    df = world.simulate(steps=80, episodes=1)
    messages = make_messages(df)
    assert len(messages) > 0
    assert "symbol" in messages.columns

def test_model_fits():
    world = CausalDynamicalWorld(TaskSpec(task_id=0))
    df = world.simulate(steps=40, episodes=2, interventions=True)
    model = fit_world_model(df)
    assert model.adjacency_scores.shape[0] == 6

def test_communication_runs():
    df, messages, metrics, message_stats, tom, shared_scores, true_graph = run_communication_experiment(steps=80)
    assert len(messages) > 0
    assert len(metrics) == 4
    assert shared_scores.shape == true_graph.shape
