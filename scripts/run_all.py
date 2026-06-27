import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))


from pathlib import Path

from src.experiments.communication import run_communication_experiment
from src.visualize import plot_bar, plot_message_usage, plot_matrix, rollout_gif

def main():
    results = Path("results")
    results.mkdir(exist_ok=True)

    df, messages, metrics, message_stats, tom, shared_scores, true_graph = run_communication_experiment()

    df.to_csv(results / "multi_agent_world.csv", index=False)
    messages.to_csv(results / "message_log.csv", index=False)
    metrics.to_csv(results / "communication_metrics.csv", index=False)
    message_stats.to_csv(results / "message_stats.csv", index=False)
    tom.to_csv(results / "theory_of_mind_metrics.csv", index=False)

    plot_bar(metrics, "agent", "prediction_mse", results / "communication_prediction_comparison.png", "Communication vs partial world models", "prediction MSE")
    plot_message_usage(messages, results / "message_symbol_usage.png")
    plot_bar(message_stats, "n_unique_symbols", "message_entropy", results / "message_entropy.png", "Message entropy", "entropy")
    plot_matrix(shared_scores, results / "shared_causal_graph.png", "Shared causal graph", "edge score")
    plot_bar(tom, "condition", "tom_mse", results / "theory_of_mind_error.png", "Theory-of-mind-style prediction", "MSE")
    rollout_gif(df, results / "multi_agent_rollout.gif")

    print("Multi-agent causal communication experiment completed.")
    print(metrics)

if __name__ == "__main__":
    main()
