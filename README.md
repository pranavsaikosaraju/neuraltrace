# NeuralTrace

**Question:** language models carry an internal signal for "I don't recognise this entity". When can that signal be used to reduce hallucination, and how does the answer change with model size and language?

**Status:** in progress. Results are added to `results/` as each experiment finishes; nothing in this README is claimed before it is measured.

## Plan
See `BLUEPRINT.md` for hypotheses, experiments (E0 to E6), controls and stop rules.

## Layout
- `neuraltrace/` library (metrics, directions, data, activations, interventions)
- `notebooks/` one Kaggle notebook per experiment
- `results/` JSON and figures, committed
- `tests/` unit tests for the non-GPU logic (`pytest`)

## License
MIT
