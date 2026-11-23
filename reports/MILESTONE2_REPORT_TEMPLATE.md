# Milestone 2 — Technical Checkpoint (Group 24)

*Report worksheet. Replace all square brackets with evidence from actual runs. Follow Section 5 of your course project guidelines if it specifies additional headings. Max 3 pages excluding references.*

## 1. Project objective and current status

- RGB clothing photo -> clothing category, colour and (where labels support it) pattern.
- Baseline: independent small CNNs. Proposed: shared feature CNN with three heads.
- State what is implemented, working and not yet working.

## 2. Dataset and preprocessing

- Source: Fashion Product Images (Small), Kaggle. Licence: MIT.
- Source sample count: approximately 44,000 products before filtering.
- Actual processed sample counts: train [ ], validation [ ], test [ ].
- Pattern-label method and verified count: [ ]. Explain uncertain labels.
- Image size: 128 x 128 RGB; train-only affine augmentation; seed [ ].

## 3. Architecture and implementation

- Three convolutional layers (32, 64, 128 channels), ReLU, pooling and adaptive average pooling.
- Individual classifier heads versus shared extractor; cross-entropy, Adam, learning rate, epochs.
- Which repository files run the steps, and the saved model-checkpoint format.

## 4. Measured baseline and proposed-model results

| Model | Task | Validation accuracy | Validation macro-F1 | Test accuracy | Parameter count |
|---|---|---:|---:|---:|---:|
| Separate CNN | Category | [ ] | [ ] | [ ] | [ ] |
| Separate CNN | Colour | [ ] | [ ] | [ ] | [ ] |
| Separate CNN | Pattern | [ ] | [ ] | [ ] | [ ] |
| Shared CNN | Category | [ ] | [ ] | [ ] | [ ] |
| Shared CNN | Colour | [ ] | [ ] | [ ] | [ ] |
| Shared CNN | Pattern | [ ] | [ ] | [ ] | [ ] |

- Include a genuine training/validation loss curve saved by the run.
- If pattern labels are insufficient, mark pattern rows N/A; explain why. Never invent metrics.

## 5. Challenges, contributions and next steps

- Summarize what failed, how you diagnosed it, and what is still incomplete.
- Name each member's actual contribution and reference `CONTRIBUTIONS.md`.
- Next: audit pattern labels, increase training data/epochs, tune loss weights and compare multiple runs.

## References

[1] P. Aggarwal, "Fashion Product Images (Small)," Kaggle, 2019, doi: 10.34740/KAGGLE/DS/175990.
