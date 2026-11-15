# Trained model artifacts

The release includes compact checkpoints for the final manual-sparse and
all-gauges experiments. Raw predictions, graph artifacts, and training data are
distributed separately because they are too large for Git.

Each directory contains the final GATGCN checkpoint and meta-MLP artifact. The
manual-sparse directory also includes the final MLP checkpoint. Checkpoint
selection used validation data only; 2020-2024 remained the test period.

The manual-sparse GATGCN predates the corrected disjoint encoded-late fusion and
therefore passes ERA5/IMERG through both A1 and the auxiliary encoder. The
all-gauges checkpoint uses the corrected architecture: A1 receives gauge value
and availability only, while ERA5/IMERG enter through the auxiliary encoder.
