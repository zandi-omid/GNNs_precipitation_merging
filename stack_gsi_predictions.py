#!/usr/bin/env python3
# coding: utf-8

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import argparse
import pickle

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


KEY_COLS = ["date", "node"]
REFERENCE_COLS = ["idw_train", "era5", "imerg", "prism", "mrms", "mswep", "gmcp"]


def load_toml(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def cfg_get(cfg: Dict[str, Any], section: str, key: str, default=None):
    return cfg.get(section, {}).get(key, default)


def parse_args():
    p = argparse.ArgumentParser(
        description="Train/apply a leakage-aware level-2 stacker from existing GSI base-model CSVs."
    )
    p.add_argument("--config", required=True)
    return p.parse_args()


def _paired_frame(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols).copy()
    return out


def _read_meta_train_csvs(cfg: Dict[str, Any]) -> pd.DataFrame:
    csvs = cfg_get(cfg, "paths", "meta_train_csvs", None)
    if csvs is None:
        csvs = [cfg_get(cfg, "paths", "meta_train_csv")]
    if not csvs:
        raise ValueError("Set [paths].meta_train_csv or [paths].meta_train_csvs.")

    frames = []
    for path in csvs:
        path = Path(path)
        print("Meta-training CSV:", path)
        frame = pd.read_csv(path)
        frame["_meta_source_csv"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _merge_auxiliary_features(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    aux_csv = cfg_get(cfg, "auxiliary", "csv", None)
    aux_feature_cols = [str(c) for c in cfg_get(cfg, "auxiliary", "feature_cols", [])]
    if not aux_csv or not aux_feature_cols:
        return df, feature_cols

    aux_date_col = str(cfg_get(cfg, "auxiliary", "date_col", "date"))
    aux_node_col = str(cfg_get(cfg, "auxiliary", "node_col", "node"))

    aux = pd.read_csv(aux_csv, usecols=[aux_date_col, aux_node_col, *aux_feature_cols])
    aux = aux.rename(columns={aux_date_col: "date", aux_node_col: "node"})
    aux["date"] = pd.to_datetime(aux["date"]).dt.strftime("%Y-%m-%d")
    aux["node"] = aux["node"].astype(df["node"].dtype, copy=False)

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out = out.merge(aux, on=KEY_COLS, how="left", validate="many_to_one")

    merged_features = list(feature_cols)
    for col in aux_feature_cols:
        if col not in merged_features:
            merged_features.append(col)
    return out, merged_features


def _date_years(df: pd.DataFrame) -> Tuple[int, int]:
    dt = pd.to_datetime(df["date"])
    return int(dt.dt.year.min()), int(dt.dt.year.max())


def _check_no_overlap(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    train_keys = set(map(tuple, train_df[KEY_COLS].to_numpy()))
    test_keys = set(map(tuple, test_df[KEY_COLS].to_numpy()))
    overlap = train_keys & test_keys
    if overlap:
        examples = list(overlap)[:10]
        raise ValueError(
            "Meta-training and final-evaluation CSVs overlap in date/node rows. "
            f"First overlaps: {examples}"
        )


def _make_model(cfg: Dict[str, Any]):
    kind = str(cfg_get(cfg, "meta_model", "kind", "ridge")).lower()
    if kind == "ridge":
        return Ridge(
            alpha=float(cfg_get(cfg, "meta_model", "alpha", 1.0)),
            fit_intercept=bool(cfg_get(cfg, "meta_model", "fit_intercept", True)),
            positive=bool(cfg_get(cfg, "meta_model", "positive", False)),
        )
    if kind == "linear":
        return LinearRegression(
            fit_intercept=bool(cfg_get(cfg, "meta_model", "fit_intercept", True)),
            positive=bool(cfg_get(cfg, "meta_model", "positive", False)),
        )
    if kind == "mlp":
        hidden = cfg_get(cfg, "meta_model", "hidden_layer_sizes", [32, 16])
        if isinstance(hidden, int):
            hidden = [hidden]
        hidden = tuple(int(v) for v in hidden)
        mlp = MLPRegressor(
            hidden_layer_sizes=hidden,
            activation=str(cfg_get(cfg, "meta_model", "activation", "relu")),
            solver=str(cfg_get(cfg, "meta_model", "solver", "adam")),
            alpha=float(cfg_get(cfg, "meta_model", "alpha", 1e-4)),
            batch_size=cfg_get(cfg, "meta_model", "batch_size", "auto"),
            learning_rate_init=float(cfg_get(cfg, "meta_model", "learning_rate_init", 1e-3)),
            max_iter=int(cfg_get(cfg, "meta_model", "max_iter", 300)),
            random_state=int(cfg_get(cfg, "meta_model", "random_state", 42)),
            early_stopping=bool(cfg_get(cfg, "meta_model", "early_stopping", False)),
            validation_fraction=float(cfg_get(cfg, "meta_model", "validation_fraction", 0.1)),
            n_iter_no_change=int(cfg_get(cfg, "meta_model", "n_iter_no_change", 20)),
            tol=float(cfg_get(cfg, "meta_model", "tol", 1e-4)),
            verbose=bool(cfg_get(cfg, "meta_model", "verbose", False)),
        )
        if bool(cfg_get(cfg, "meta_model", "standardize", True)):
            return make_pipeline(StandardScaler(), mlp)
        return mlp
    raise ValueError(
        f"Unsupported [meta_model].kind={kind!r}; use 'ridge', 'linear', or 'mlp'."
    )


def _paired(obs, sim):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    ok = np.isfinite(obs) & np.isfinite(sim)
    return obs[ok], sim[ok]


def cc(obs, sim):
    o, s = _paired(obs, sim)
    if len(o) < 2 or np.std(o) == 0 or np.std(s) == 0:
        return np.nan
    return float(np.corrcoef(o, s)[0, 1])


def rmse(obs, sim):
    o, s = _paired(obs, sim)
    return float(np.sqrt(np.mean((s - o) ** 2))) if len(o) else np.nan


def bias(obs, sim):
    o, s = _paired(obs, sim)
    return float(np.mean(s - o)) if len(o) else np.nan


def alpha_ratio(obs, sim):
    o, s = _paired(obs, sim)
    if len(o) < 2:
        return np.nan
    sig_o = np.std(o, ddof=0)
    return float(np.std(s, ddof=0) / sig_o) if sig_o != 0 else np.nan


def beta_ratio(obs, sim):
    o, s = _paired(obs, sim)
    if len(o) < 2:
        return np.nan
    mu_o = np.mean(o)
    return float(np.mean(s) / mu_o) if mu_o != 0 else np.nan


def kge(obs, sim):
    r = cc(obs, sim)
    a = alpha_ratio(obs, sim)
    b = beta_ratio(obs, sim)
    if not np.isfinite(r) or not np.isfinite(a) or not np.isfinite(b):
        return np.nan
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (a - 1.0) ** 2 + (b - 1.0) ** 2))


def metric_row(product: str, obs, sim) -> Dict[str, Any]:
    return {
        "Product": product,
        "KGE_global": kge(obs, sim),
        "CC_global": cc(obs, sim),
        "RMSE_global": rmse(obs, sim),
        "Bias_global": bias(obs, sim),
        "Alpha_global": alpha_ratio(obs, sim),
        "Beta_global": beta_ratio(obs, sim),
        "n_samples": len(_paired(obs, sim)[0]),
    }


def per_node_metrics(df: pd.DataFrame, products: List[Tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for (node, lon, lat), g in df.groupby(["node", "lon", "lat"], sort=False):
        for col, label in products:
            rows.append({
                "node": int(node),
                "lon": float(lon),
                "lat": float(lat),
                "Product": label,
                "KGE": kge(g["gauge"], g[col]),
                "CC": cc(g["gauge"], g[col]),
                "RMSE": rmse(g["gauge"], g[col]),
                "Bias": bias(g["gauge"], g[col]),
                "Alpha": alpha_ratio(g["gauge"], g[col]),
                "Beta": beta_ratio(g["gauge"], g[col]),
                "n": len(_paired(g["gauge"], g[col])[0]),
            })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    cfg = load_toml(args.config)

    final_eval_csv = Path(cfg_get(cfg, "paths", "final_eval_csv"))
    out_csv = Path(cfg_get(cfg, "paths", "out_csv"))
    model_out = Path(cfg_get(cfg, "paths", "model_out", out_csv.with_suffix(".pkl")))
    metrics_out = Path(cfg_get(cfg, "paths", "metrics_out", out_csv.with_name(out_csv.stem + "_global_metrics.csv")))
    per_node_out = Path(cfg_get(cfg, "paths", "per_node_metrics_out", out_csv.with_name(out_csv.stem + "_per_node_metrics.csv")))

    target_col = str(cfg_get(cfg, "stacking", "target_col", "gauge"))
    feature_cols = [str(c) for c in cfg_get(cfg, "stacking", "feature_cols", [])]
    output_col = str(cfg_get(cfg, "stacking", "output_col", "stack_ridge"))
    output_label = str(cfg_get(cfg, "stacking", "output_label", "Stack-Ridge"))
    clip_min = cfg_get(cfg, "stacking", "clip_min", 0.0)
    clip_max = cfg_get(cfg, "stacking", "clip_max", None)
    passthrough_cols = [str(c) for c in cfg_get(cfg, "stacking", "comparison_cols", [])]

    if not feature_cols:
        raise ValueError("[stacking].feature_cols must list base-model prediction columns.")

    print("Final-evaluation CSV:", final_eval_csv)
    print("Feature columns:", feature_cols)
    print("Target column:", target_col)

    train_df_raw = _read_meta_train_csvs(cfg)
    test_df_raw = pd.read_csv(final_eval_csv)
    _check_no_overlap(train_df_raw, test_df_raw)

    train_df_raw, feature_cols = _merge_auxiliary_features(train_df_raw, cfg, feature_cols)
    test_df_raw, feature_cols = _merge_auxiliary_features(test_df_raw, cfg, feature_cols)
    print("Final feature columns:", feature_cols)

    train_years = _date_years(train_df_raw)
    test_years = _date_years(test_df_raw)
    print(f"Meta-training years: {train_years[0]}-{train_years[1]}")
    print(f"Final-evaluation years: {test_years[0]}-{test_years[1]}")

    train_df = _paired_frame(train_df_raw, [target_col, *feature_cols])
    test_df = _paired_frame(test_df_raw, [target_col, *feature_cols])
    if len(train_df) == 0:
        raise RuntimeError("No finite meta-training rows after filtering.")
    if len(test_df) == 0:
        raise RuntimeError("No finite final-evaluation rows after filtering.")

    model = _make_model(cfg)
    x_train = train_df[feature_cols].to_numpy(dtype=np.float64)
    y_train = train_df[target_col].to_numpy(dtype=np.float64)
    model.fit(x_train, y_train)

    x_test = test_df[feature_cols].to_numpy(dtype=np.float64)
    pred = model.predict(x_test).astype(np.float32)
    if clip_min is not None or clip_max is not None:
        lo = None if clip_min is None else float(clip_min)
        hi = None if clip_max is None else float(clip_max)
        pred = np.clip(pred, lo, hi)

    out = test_df_raw.copy()
    out[output_col] = np.nan
    out.loc[test_df.index, output_col] = pred

    scenario = str(cfg_get(cfg, "meta", "scenario", "stacked_gsi"))
    out["scenario"] = scenario

    coef = np.asarray(getattr(model, "coef_", []), dtype=float).reshape(-1)
    intercept = float(getattr(model, "intercept_", 0.0))
    print("Model:", model)
    if coef.size:
        print("Intercept:", intercept)
        print("Coefficients:")
        for col, val in zip(feature_cols, coef):
            print(f"  {col}: {val:.6f}")
    else:
        print("Model does not expose linear coefficients.")

    products = []
    for col in passthrough_cols:
        if col in out.columns:
            products.append((col, col))
    products.append((output_col, output_label))

    global_metrics = pd.DataFrame([
        metric_row(label, out[target_col], out[col])
        for col, label in products
    ])
    node_metrics = per_node_metrics(out, products)
    summary = (
        node_metrics
        .groupby("Product", as_index=False)
        .agg(
            KGE_median=("KGE", "median"),
            KGE_mean=("KGE", "mean"),
            CC_median=("CC", "median"),
            CC_mean=("CC", "mean"),
            RMSE_median=("RMSE", "median"),
            RMSE_mean=("RMSE", "mean"),
            Bias_median=("Bias", "median"),
            Bias_mean=("Bias", "mean"),
            Alpha_median=("Alpha", "median"),
            Alpha_mean=("Alpha", "mean"),
            Beta_median=("Beta", "median"),
            Beta_mean=("Beta", "mean"),
            n_nodes=("KGE", "count"),
        )
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    per_node_out.parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(out_csv, index=False)
    global_metrics.to_csv(metrics_out, index=False)
    node_metrics.to_csv(per_node_out, index=False)
    summary.to_csv(per_node_out.with_name(per_node_out.stem + "_summary.csv"), index=False)
    with open(model_out, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "feature_cols": feature_cols,
                "auxiliary_feature_cols": [
                    str(c) for c in cfg_get(cfg, "auxiliary", "feature_cols", [])
                ],
                "target_col": target_col,
                "output_col": output_col,
                "scenario": scenario,
                "intercept": intercept,
                "coef": coef,
            },
            f,
        )

    print("Rows used for meta training:", len(train_df))
    print("Rows used for final stack prediction:", len(test_df))
    print("Saved stacked CSV:", out_csv)
    print("Saved stacker model:", model_out)
    print("Saved global metrics:", metrics_out)
    print(global_metrics.sort_values("KGE_global", ascending=False))
    print("Saved per-node metrics:", per_node_out)
    print("Station summary:")
    print(summary.sort_values("KGE_median", ascending=False))


if __name__ == "__main__":
    main()
