# %%

# read_tensor_board_tgcn.py
import os, re
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# RUN_DIR = "/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/logs/TGCN_T14_70train_30test/version_8"
RUN_DIR = "/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/logs/TGCN_T30_log_normal_train2005_2018_val2019/version_2"
OUT_DIR = os.path.join(RUN_DIR, "scalar_plots")
os.makedirs(OUT_DIR, exist_ok=True)

def safe_name(s: str) -> str:
    return re.sub(r"[^\w\-_.]+", "_", s)

ea = event_accumulator.EventAccumulator(RUN_DIR)
ea.Reload()

scalar_tags = ea.Tags().get("scalars", [])
print("Available scalar tags:", scalar_tags)

# Choose the tags you care about (in your case: all of them)
tags_to_plot = scalar_tags

for tag in tags_to_plot:
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]

    if len(values) == 0:
        continue

    plt.figure(figsize=(7, 4))
    plt.plot(steps, values, linewidth=1.8)
    plt.title(tag)
    plt.xlabel("step")
    plt.ylabel("value")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.show()
    out_path = os.path.join(OUT_DIR, safe_name(tag) + ".png")
    # plt.savefig(out_path, dpi=200, bbox_inches="tight")
    # plt.close()

    print(f"Saved: {out_path}  (n={len(values)}, step_first={steps[0]}, step_last={steps[-1]})")
# %%

# read_tensorboard_tgcn_all_in_one.py

import os
import math
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# ---------------- CONFIG ----------------
# RUN_DIR = "/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/logs/TGCN_T14_70train_30test/version_1"
# RUN_DIR = "/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/logs/TGCN_T14_train2005_2018_val2019/version_8/"
RUN_DIR = "/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/logs/TGCN_T30_log_normal_train2005_2018_val2019/version_2"
OUT_FIG = os.path.join(RUN_DIR, "training_metrics_all.png")

# Which metrics to plot (order matters)
METRICS = [
    "train/loss_step",
    "train/mse",
    "train/rmse",
    "train/bias",
    "train/cc",
    "test_loss",
    "val/mse",
    "val/cc"
]

# ---------------- LOAD EVENTS ----------------
ea = event_accumulator.EventAccumulator(RUN_DIR)
ea.Reload()

available = ea.Tags().get("scalars", [])
print("Available scalar tags:", available)

metrics = [m for m in METRICS if m in available]
print("Plotting metrics:", metrics)

# ---------------- FIGURE LAYOUT ----------------
n = len(metrics)
cols = 2
rows = math.ceil(n / cols)

fig, axs = plt.subplots(rows, cols, figsize=(12, 4 * rows))
axs = axs.flatten()

# ---------------- PLOTTING ----------------
for ax, tag in zip(axs, metrics):
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]

    if len(values) == 1:
        # Single-point metric (e.g. test_loss)
        ax.scatter(steps[0], values[0], s=80, color="red", zorder=3)
        ax.axhline(values[0], linestyle="--", alpha=0.5)
        ax.set_title(f"{tag} (final)")
    else:
        ax.plot(steps, values, linewidth=1.8)
        ax.set_title(tag)

    ax.set_xlabel("step")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.35)

# Hide unused subplots
for i in range(len(metrics), len(axs)):
    axs[i].axis("off")

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
plt.show()

print(f"\n✅ Saved combined figure to:\n{OUT_FIG}")
# %%
