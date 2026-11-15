#!/usr/bin/env python3
from pathlib import Path
import sys
import warnings

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "GSI"))

from GSI.inference_global import load_model


def main() -> None:
    sample_path = ROOT / "examples/data/sequence_t3.pt"
    graph_path = ROOT / "examples/data/manual_sparse_graph.pkl"
    checkpoint = ROOT / "checkpoints/manual_sparse/gatgcn_best.ckpt"

    sample = torch.load(sample_path, map_location="cpu", weights_only=True)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="A1 and auxiliary feature channels overlap.*",
            category=UserWarning,
        )
        model = load_model(checkpoint, sample, graph_path, torch.device("cpu"))
    model.eval()
    with torch.no_grad():
        prediction = model(sample["x"].unsqueeze(0))

    print(f"date: {sample['date']}")
    print(f"input shape: {tuple(sample['x'].shape)}")
    print(f"prediction shape: {tuple(prediction.shape)}")
    print(f"finite predictions: {int(torch.isfinite(prediction).sum()):,}")


if __name__ == "__main__":
    main()
