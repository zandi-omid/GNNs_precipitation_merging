# Proposed external data bundle

## Core bundle (recommended)

The core bundle is approximately 1 GB before packaging and is sufficient to
load the released graph checkpoints and reproduce MLP training without storing
generated sequence tensors in Git.

| Artifact | Approximate size |
| --- | ---: |
| Manual-sparse graph | 2.7 MB |
| All-gauges cross-fit graph | 4.2 MB |
| Manual-sparse MLP feature table | 143 MB |
| All-gauges MLP feature table | 803 MB |
| Manual gauge split | 32 KB |

The bundle should also contain a machine-readable manifest with SHA-256 hashes,
coordinate reference information, date coverage, feature definitions, and the
exact scripts/configuration used to generate every artifact.

## Optional prebuilt sequences

The manual-sparse and all-gauges sequence directories contain 7,276 samples
each and occupy approximately 37 GB and 47 GB, respectively. Because these are
deterministically generated, they should be offered as a separate optional
download rather than included in the core bundle.

## Distribution

Google Drive is suitable for initial project distribution. For the final paper,
archive the versioned bundle in Zenodo or an institutional repository to obtain
a persistent DOI, while retaining Drive as a convenient mirror.
