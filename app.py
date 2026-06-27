from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Multi-Agent Causal Communication Research", layout="wide")
st.title("Multi-Agent Causal Communication Research")
st.write("Partial-observation agents, message exchange, shared world models, message entropy, and theory-of-mind-style prediction.")

results = Path("results")
if not results.exists():
    st.warning("Run `python scripts/run_all.py` first.")
    st.stop()

for img in [
    "communication_prediction_comparison.png",
    "message_symbol_usage.png",
    "message_entropy.png",
    "shared_causal_graph.png",
    "theory_of_mind_error.png",
    "multi_agent_rollout.gif",
]:
    p = results / img
    if p.exists():
        st.subheader(img)
        st.image(str(p), use_container_width=True)

for csv in [
    "communication_metrics.csv",
    "message_log.csv",
    "message_stats.csv",
    "theory_of_mind_metrics.csv",
]:
    p = results / csv
    if p.exists():
        st.subheader(csv)
        st.dataframe(pd.read_csv(p).head(50), use_container_width=True)
