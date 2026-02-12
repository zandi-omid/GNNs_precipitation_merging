import pickle, numpy as np, xarray as xr, torch
from pathlib import Path

GRAPH_PKL = Path("/xdisk/behrangi/omidzandi/GNNs/data/graphs/graph_with_features_labels.pkl")
NC_PATH   = Path("/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/inference/TGCN_T30_log_normal_test2020_2024/pred_inputs_daily_maps.nc")
SEQ_DIR   = Path("/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/data/pyg_sequences_tgcn_T030_all_years")

# load graph + node_list used in writer
with open(GRAPH_PKL, "rb") as f:
    payload = pickle.load(f)
G = payload["graph"]
node_list = sorted(G.nodes())  # THIS is what your writer assumed

# open nc
ds = xr.open_dataset(NC_PATH)

# pick one day that exists in both
test_date = "2020-02-01"  # change if needed
seq_file = next(SEQ_DIR.glob(f"*{test_date}*"), None)

remember = {}
if seq_file is None:
    # your seq files don't contain date in filename, so search by reading a few
    for p in list(SEQ_DIR.glob("seq_*.pt"))[:2000]:
        d = torch.load(p, map_location="cpu", weights_only=True)
        if str(d["date"]) == test_date:
            seq_file = p
            break

assert seq_file is not None, "Could not find seq file for test_date"

d = torch.load(seq_file, map_location="cpu", weights_only=True)
x_last = d["x"].numpy()[-1].astype(np.float32)  # [N,2]
era5_nodes_seq = x_last[:,0]                    # seq order (truth)

# extract nc values at node grid positions IN node_list order
era5_nc = ds["era5"].sel(time=np.datetime64(test_date)).values
era5_nodes_nc = np.array([era5_nc[i,j] for (i,j) in node_list], dtype=np.float32)

diff = np.nanmax(np.abs(era5_nodes_nc - era5_nodes_seq))
print("Max abs diff (era5):", diff)
print("Mean abs diff (era5):", np.nanmean(np.abs(era5_nodes_nc - era5_nodes_seq)))