#%%
import torch
from pathlib import Path

#%%
seq_dir = Path("/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/data/pyg_sequences_tgcn_T14")
files = sorted(seq_dir.glob("seq_*.pt"))

first = torch.load(files[0])
last  = torch.load(files[-1])

print("First sample date:", first.get("date"))
print("Last sample date :", last.get("date"))
# %%
import torch
from pathlib import Path

seq_dir = Path("/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/data/pyg_sequences_tgcn_T14")
f = sorted(seq_dir.glob("seq_*.pt"))[5]

sample = torch.load(f, map_location="cpu")

print("📁 File:", f.name)
print("Keys:", sample.keys())
print()

x = sample["x"]
y = sample["y"]
y_mask = sample["y_mask"]
edge_index = sample["edge_index"]
edge_weight = sample["edge_weight"]
date = sample.get("date", "N/A")

print("📅 Target date:", date)
print()
print("x shape (T, N, F):", x.shape)
print("y shape:", y.shape)
print("y_mask shape:", y_mask.shape)
print("edge_index shape:", edge_index.shape)
print("edge_weight shape:", edge_weight.shape)
# %%
