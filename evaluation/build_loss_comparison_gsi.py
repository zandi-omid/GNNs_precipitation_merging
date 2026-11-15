#!/usr/bin/env python3

from pathlib import Path
import argparse

import pandas as pd
import tomli as tomllib


KEYS = ["date", "node"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)

    mse = pd.read_csv(cfg["paths"]["mse_csv"])
    quantile = pd.read_csv(cfg["paths"]["quantile_csv"])
    keep = KEYS + ["gatgcn_mean", "mlp_mean", "gatgcn_q70", "mlp_q75"]
    missing = [c for c in keep if c not in quantile.columns]
    if missing:
        raise ValueError(f"Quantile baseline CSV is missing columns: {missing}")

    quantile = quantile[keep].rename(
        columns={
            "gatgcn_mean": "gatgcn_quantile_q50",
            "mlp_mean": "mlp_quantile_q50",
            "gatgcn_q70": "gatgcn_quantile_q70",
            "mlp_q75": "mlp_quantile_q75",
        }
    )
    out = mse.merge(quantile, on=KEYS, how="inner", validate="one_to_one")
    out["scenario"] = cfg["meta"]["scenario"]
    out_path = Path(cfg["paths"]["out_csv"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"MSE rows: {len(mse):,}")
    print(f"Quantile rows: {len(quantile):,}")
    print(f"Shared comparison rows: {len(out):,}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
