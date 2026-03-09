#%%
import torch
p = "/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/data/pyg_sequences_tgcn_T030_all_years_IDW_added/seq_00000.pt"
d = torch.load(p, map_location="cpu")
print(d["x"].shape)
# %%
