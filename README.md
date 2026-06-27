# Multi-Agent Causal Communication

A research-oriented project on communication and shared causal world models in multi-agent systems.

This repository is not a casual agent demo. It implements a controlled experiment where agents observe different parts of a causal dynamical world and exchange discrete messages to support shared world-model construction.

## Research question

Can communication between partially informed agents improve reconstruction of a shared causal world model?

## Experimental design

The environment is a causal dynamical system.  
Each agent observes only a subset of state variables.

The experiment compares:

```text
local partial model without communication
vs
shared model with message exchange
```

It also measures message entropy and a theory-of-mind-style prediction proxy.

## Implemented components

```text
causal dynamical world
partial-observation agents
discrete symbolic message protocol
no-communication baseline
communication-enabled shared model
message entropy
symbol usage analysis
communication benefit metric
theory-of-mind-style prediction
multi-agent rollout animation
```

## Repository structure

```text
src/core/dynamics.py                   causal dynamical world
src/core/metrics.py                    entropy and graph metrics
src/models/sparse_world_model.py       shared sparse causal model
src/experiments/communication.py       communication experiment
scripts/run_all.py                     full experiment
app.py                                 Streamlit result viewer
tests/                                 verification tests
docs/                                  research documentation
```

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m pytest -q
python scripts/run_all.py
streamlit run app.py
```

## Outputs

```text
results/multi_agent_world.csv
results/message_log.csv
results/communication_metrics.csv
results/message_stats.csv
results/theory_of_mind_metrics.csv
results/communication_prediction_comparison.png
results/message_symbol_usage.png
results/message_entropy.png
results/shared_causal_graph.png
results/theory_of_mind_error.png
results/multi_agent_rollout.gif
```

