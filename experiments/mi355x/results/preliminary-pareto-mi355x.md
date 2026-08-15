# Preliminary MI355X Pareto screen

This is an intentionally small screen, not the planned 42-point sweep. It uses
the pinned InferenceX GPT-OSS-120B random 8192-input/1024-output workload and
only concurrency 4 and 8 for two AFD layouts.

| System | Layout | Concurrency | Total tok/s | Median interactivity (output tok/s) | Correctness |
|---|---:|---:|---:|---:|---|
| FastAFD | 4:4 | 4 | 1,271.11 | 36.91 | Provisional |
| FastAFD | 4:4 | 8 | 1,839.41 | 28.54 | Provisional |
| FastAFD | 7:1 | 4 | 1,262.70 | 36.70 | Passed |
| FastAFD | 7:1 | 8 | 2,003.12 | 29.02 | Passed |
| vLLM reproduced | TP8 | 4 | 10,304.87 | 299.65 | Baseline |
| vLLM reproduced | TP8 | 8 | 18,929.00 | 277.05 | Baseline |

The vLLM values above are eight-GPU totals reconstructed from the report's
per-GPU values. TP8 is the closest topology comparison because both systems use
one endpoint spanning all eight GPUs. At concurrency 4, the best AFD point is
8.1x lower in throughput and interactivity. At concurrency 8, the best AFD
point is 9.5x lower in throughput and 9.5x lower in interactivity. This is not
close enough to justify the full sweep before further performance work.

The 7:1 points ran on `marlowe-mi355x-4` and passed the four-prompt alignment
gate (three exact and one exact-logit near-tie). The 4:4 points ran on the
identical `marlowe-mi355x-2` node and are provisional: the alignment probe had
three near-ties and one real divergence at 15 BF16 ULP. Raw paths are retained
in the normalized CSV.
