

<!-- ACETEAM_TRAINED_CANDIDATE -->
# AceTeam trained student candidate

Fine-tuned the official 26,181-parameter Policy on the uploaded teacher-v2 recording. Student inputs, architecture, action decoding, prediction handling and recurrent runtime remain the official implementation. Teacher/recorder code was left unchanged.

## Recording and split

14,745 rows; 17 columns; 13 model inputs. Game time 0.95–347.024994 seconds. All numerical fields finite, timestamps strictly increasing, recorded callback intervals consistent with o12. Replayed teacher agreed except one mode label at a phase boundary (CSV time precision). The supplied 22 teacher state-machine tests passed.

Modes in the whole recording: Neutral 2,376; Chase 11,966; Jump 328; Dodge 75. Only 10 speed-flip sequences were recorded. The on_ground field is used by the teacher but is never fed to the student.

Training: first 10,820 rows. Gap: 238 rows (approximately 5 seconds). Validation: last 3,687 rows, starting at 263.850006 seconds. No adjacent-frame random split. Validation replays preceding observations to warm memory without optimizing on validation labels. This is a within-match holdout, not independent-match validation. No final test match is available.

## Training

Two selected stages: starter fine-tuning with paired mirrored streams, then refinement emphasizing complete flip windows and release frames. Only o1/o4 and steering are negated for mirrored training. Separate GRU memories persist across chunks. AdamW, sequence chunks of 128, gradient clipping 1.0, seed 42. Stage-one selected epoch 56 at learning rate 0.001; refinement selected epoch 23 at learning rate 0.0003. Full details and checkpoint hashes are in the provenance files.

Six configurations were explored on the same validation segment: initial fine-tuning, fresh initialization, paired mirroring at two learning rates, release-only emphasis, and full-flip emphasis. The release-only candidate had lower steering error but failed to initiate the held-out flip sequence; the packaged candidate was preferred after inspecting action order. These are tuning results, not an unbiased test estimate.

## Measured offline comparison

| Metric | Starter | Packaged candidate |
|---|---:|---:|
| Chase steering mean absolute error | 0.277059 | 0.210932 |
| Chase steering RMSE | 0.505852 | 0.416965 |
| Mode accuracy | 95.28% | 97.18% |
| Macro mode F1 | 0.664867 | 0.788118 |
| False Jump/Dodge frames among all validation frames | 0.00% | 1.25% |

Steering MAE reduction: 23.9%. This measures imitation of teacher steering on recorded states, not win rate.

| Mode | Validation support | Precision | Recall |
|---|---:|---:|---:|
| Neutral | 735 | 100.00% | 100.00% |
| Chase | 2866 | 97.98% | 98.39% |
| Jump | 76 | 30.30% | 26.32% |
| Dodge | 10 | 100.00% | 80.00% |

One held-out teacher flip window had a student Jump → release → Dodge sequence. That is one offline example; it does not establish successful flips under student-controlled physics. Extra/mistimed jumps remain a risk.

## Verification and live results

Checkpoint loads with weights_only=True into the exact Policy. Tensors and outputs are finite. Official parameter count is 26,181. Starter/candidate weight cosine similarity: 0.821278.

Installation backs up the old model and affected project files. An injected missing-report failure verified restoration of original files after initialization. Official preflight and submission-validator results are recorded separately in offline_validation.txt. These checks do not launch Rocket League.

LIVE GAMEPLAY: NOT TESTED. WIN RATE: NOT MEASURED. Windows installation must be run using the included installer; real matches must be tested on the user machine.

After live testing, append opponent, side, score, observations and errors here. Preserve failures as well as successes.

<!-- END_ACETEAM_TRAINED_CANDIDATE -->
