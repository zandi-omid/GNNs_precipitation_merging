# %%
import os
import re
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

#%%
# ---------------- CONFIG ----------------
RUN_DIR = "/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/logs/TGCN_T14_70train_30test/version_0"
OUT_DIR = "tensorboard_scalar_plots_TGCN"

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- FIND LATEST EVENT FILE ----------------
event_files = sorted(
    f for f in os.listdir(RUN_DIR)
    if f.startswith("events.out.tfevents")
)

if not event_files:
    raise FileNotFoundError("❌ No TensorBoard event files found.")

event_file = os.path.join(RUN_DIR, event_files[-1])
print(f"✅ Using event file:\n{event_file}")

#%%
# ---------------- LOAD EVENTS ----------------
ea = event_accumulator.EventAccumulator(event_file)
ea.Reload()

scalar_tags = ea.Tags()["scalars"]
print("\nAvailable scalar tags:")
for t in scalar_tags:
    print(" -", t)

# ---------------- HELPER ----------------
def safe_name(name: str) -> str:
    return re.sub(r"[^\w\-_.]", "_", name)

#%%

for tag in scalar_tags:
    events = ea.Scalars(tag)
    print(tag, "num_points =", len(events),
          "first =", (events[0].step, events[0].value) if events else None,
          "last  =", (events[-1].step, events[-1].value) if events else None)

#%%
# ---------------- PLOT EACH SCALAR ----------------
for tag in scalar_tags:
    events = ea.Scalars(tag)
    if len(events) == 0:
        print(f"⚠️ {tag}: no scalar points")
        continue

    steps = [e.step for e in events]
    values = [e.value for e in events]

    plt.figure(figsize=(6, 4))

    if len(events) == 1:
        # single point -> make it obvious
        plt.scatter(steps, values, s=60)
    else:
        # line + markers so it's never "invisible"
        plt.plot(steps, values, marker="o", markersize=3, linewidth=1.4)

    plt.title(f"{tag} (n={len(events)})")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

#%%
# ---------------- COMBINED FIGURE ----------------
n = len(scalar_tags)
cols = 2
rows = (n + 1) // cols

fig, axs = plt.subplots(rows, cols, figsize=(12, 4 * rows))
axs = axs.flatten()

for i, tag in enumerate(scalar_tags):
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]

    axs[i].plot(steps, values, linewidth=1.8)
    axs[i].set_title(tag)
    axs[i].set_xlabel("Step")
    axs[i].set_ylabel("Value")
    axs[i].grid(True, alpha=0.4)

# Hide unused axes
for j in range(i + 1, len(axs)):
    axs[j].axis("off")

plt.tight_layout()
plt.savefig(
    os.path.join(OUT_DIR, "ALL_SCALARS.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
# %%
