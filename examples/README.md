# Three-step example

`data/sequence_t3.pt` contains the last three time steps from one real
manual-sparse sequence ending on 2005-01-30. It retains the full 7,565-node
study domain, A1/A2 edges, targets, masks, and feature names. The graph artifact
is included because distance-aware A1 checkpoint loading requires node
coordinates and edge distances.

The example is intended as an executable format and checkpoint compatibility
test, not as a statistically meaningful training subset.
