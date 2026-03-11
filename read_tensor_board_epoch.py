#%%
import os
import re
import math
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# ---------------- CONFIG ----------------
RUN_DIR = "/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/logs/TGCN_T30_quantiles_train2005_2018_val2019_fixed_avg10/version_0"
OUT_DIR = os.path.join(RUN_DIR, "epoch_scalar_plots")
OUT_FIG = os.path.join(RUN_DIR, "training_metrics_epoch_only.png")
os.makedirs(OUT_DIR, exist_ok=True)

# Epoch-level metrics you care about
PREFERRED_METRICS = [
    "train/loss",
    "train/mse",
    "train/rmse",
    "train/bias",
    "train/cc",
    "val/loss",
    "val/mse",
    "val/rmse",
    "val/bias",
    "val/cc",
    "lr_epoch",
]

def safe_name(s: str) -> str:
    return re.sub(r"[^\w\-_.]+", "_", s)

# ---------------- LOAD EVENTS ----------------
ea = event_accumulator.EventAccumulator(RUN_DIR)
ea.Reload()

available = ea.Tags().get("scalars", [])
print("Available scalar tags:", available)

# Keep only epoch-level metrics that actually exist
metrics = [m for m in PREFERRED_METRICS if m in available]
print("Plotting epoch-level metrics:", metrics)

# ---------------- Helper: read one scalar as epoch series ----------------
def get_epoch_series(tag):
    events = ea.Scalars(tag)
    values = [e.value for e in events]
    epochs = list(range(len(values)))   # one point per epoch
    return epochs, values

# ---------------- Individual plots ----------------
for tag in metrics:
    epochs, values = get_epoch_series(tag)

    if len(values) == 0:
        continue

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, values, marker="o", linewidth=1.8)
    plt.title(tag)
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, safe_name(tag) + ".png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"Saved: {out_path}  (n_epochs={len(values)}, epoch_first={epochs[0]}, epoch_last={epochs[-1]})")

# ---------------- Combined figure ----------------
n = len(metrics)
cols = 2
rows = math.ceil(n / cols)

fig, axs = plt.subplots(rows, cols, figsize=(12, 4 * rows))
axs = axs.flatten()

for ax, tag in zip(axs, metrics):
    epochs, values = get_epoch_series(tag)

    if len(values) == 1:
        ax.scatter(epochs[0], values[0], s=80, color="red", zorder=3)
        ax.axhline(values[0], linestyle="--", alpha=0.5)
        ax.set_title(f"{tag} (single epoch)")
    else:
        ax.plot(epochs, values, marker="o", linewidth=1.8)
        ax.set_title(tag)

    ax.set_xlabel("epoch")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.35)

# Hide unused panels
for i in range(len(metrics), len(axs)):
    axs[i].axis("off")

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nSaved combined epoch-only figure to:\n{OUT_FIG}")
# %%