# NeuralTrace blueprint

## Question

Language models carry an internal signal for "I don't recognise this entity". When can that signal be used to reduce hallucination, and how does the answer change with model size and language?

## Hypotheses (fixed before any results)

- **H1 (scale).** The signal is detectable at every model size, at a similar relative depth.
- **H2 (asymmetry).** Pushing a model to hallucinate (steering towards "known") is easier than making it abstain correctly (steering towards "unknown"), at every size, relative to random-direction nulls.
- **H3 (language).** A direction found in English transfers to Spanish and Hindi, with larger losses for lower-resource languages.
- **H4 (method).** A simple difference-of-means direction matches a trained linear probe for detection. (SAE comparison is deferred; see "Not done".)

Any outcome on each is reportable. A null result on H2 or a collapse under the controls is a finding, not a failure.

## Models

Qwen2.5-Instruct at 0.5B, 1.5B, 3B and 7B parameters (ungated, a 14x size range, fp16 on two T4 GPUs). Gemma was the original plan but needs a licence acceptance that only the account holder can give; Gemma 3 plus the Gemma Scope 2 SAEs is kept as an optional second phase.

## Benchmark (NeuralTrace-Entities)

- Entity types: cities (gold: country), films (gold: director), albums (gold: performing artist), taken from Wikidata.
- Languages: English, Spanish, Hindi. Questions use one template per type and language. The Spanish and Hindi templates were written by an AI model and are not native-speaker verified.
- Real entities are sampled in five popularity bands (Wikipedia sitelink counts) so both well-known and obscure entities appear.
- Entities whose label is shared by another entity, or that contain digits, are dropped.
- Fictional entities are built by recombining pieces of real names (words for multi-word names, prefix+suffix splices for single words), matched in length to real names, and checked to be absent from the real-name set. They are "very likely non-existent", not verified non-existent.
- Train/val/test splits are a hash of the entity id, so an entity never crosses splits in any language.

## Classes (from each model's own behaviour)

- known: real entity the model answers correctly with no instruction to abstain.
- unk_real: real entity the model gets wrong or declines.
- fict: invented name.

Answers are judged by normalised string match against the gold label and its English alias (surname match allowed for people), and abstention by a phrase list per language. Both are heuristics; the unit tests and the saved example generations are the check on them.

## Experiments

| # | What | Controls |
|---|---|---|
| E1 | Behaviour: accuracy, abstention, hallucination per model, language, type, popularity band; plain vs "say you don't know" prompt | none needed |
| E2 | Detection by layer (AUROC), difference of means vs logistic probe, at the last entity token and at the final prompt token | train/val/test split by entity; surface-feature baseline (name length, words, tokens); token-count-matched subset |
| E3 | Scale sweep: E2 across the four sizes | identical data and code |
| E4 | Cross-language transfer of the direction; cosine similarity between languages' directions | per-language baseline |
| E5 | Causal use: add or remove the direction at one layer; measure abstention and accuracy per class | random directions, shuffled-label directions, prompting baseline |

Steering unit: alpha is measured in multiples of the gap between the known and unknown class means along the direction. alpha = +1 moves an unknown entity to the average known one.

## Compute

Kaggle free tier: 2x T4 (15 GB each), 30 GPU-hours per week. Each model is one run; activations are not stored, only summary statistics.

## Stop rules

- If the signal vanishes under the token-count-matched control, the headline becomes "this known result is partly a surface artifact".
- If a model produces non-finite activations in fp16, that size is dropped and reported.
- Nothing is tuned after seeing test-set results.

## Not done (and why)

- SAE-based detection and steering (needs Gemma and a licence acceptance).
- Circuit-level mechanism (attribution graphs): out of scope for this compute budget.
- Native-speaker validation of the Spanish and Hindi templates.
