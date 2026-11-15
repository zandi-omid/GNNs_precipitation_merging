# utils/data/tgsi_seq_datamodule.py
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info


def seed_tgsi_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed + int(worker_id))


def targets_with_active_a1_context(
    target_mask: torch.Tensor,
    context_mask: torch.Tensor,
    edge_index_a1: torch.Tensor,
) -> torch.Tensor:
    """
    Retain only targets with at least one active A1 context sender.

    Edge convention:
        edge_index_a1[0] = receiver
        edge_index_a1[1] = sender
    """
    target_mask = target_mask.bool()
    context_mask = context_mask.bool()

    recv = edge_index_a1[0].long()
    send = edge_index_a1[1].long()

    edge_has_active_sender = context_mask[send]

    incoming_context_count = torch.zeros(
        target_mask.numel(),
        dtype=torch.long,
    )

    incoming_context_count.scatter_add_(
        dim=0,
        index=recv,
        src=edge_has_active_sender.long(),
    )

    has_context_neighbor = incoming_context_count > 0

    return target_mask & has_context_neighbor

class TGSIRandomMaskedPTDataset(Dataset):
    def __init__(
        self,
        files,
        samples_per_epoch: int = 10000,
        context_fraction: float = 0.50,
        context_fraction_min: Optional[float] = None,
        context_fraction_max: Optional[float] = None,
        gauge_value_channel: int = 0,
        gauge_mask_channel: int = 1,
        min_context_gauges: int = 5,
        min_loss_gauges: int = 5,
        seed: int = 42,
        max_resample_tries: int = 20,
        dynamic_masking: bool = True,
        masking_mode: str = "random_fraction",
        n_folds: int = 10,
    ):
        self.files = list(files)
        self.samples_per_epoch = int(samples_per_epoch)
        self.context_fraction = float(context_fraction)
        self.context_fraction_min = float(
            self.context_fraction if context_fraction_min is None else context_fraction_min
        )
        self.context_fraction_max = float(
            self.context_fraction if context_fraction_max is None else context_fraction_max
        )
        self.gauge_value_channel = int(gauge_value_channel)
        self.gauge_mask_channel = int(gauge_mask_channel)
        self.min_context_gauges = int(min_context_gauges)
        self.min_loss_gauges = int(min_loss_gauges)
        self.seed = int(seed)
        self.max_resample_tries = int(max_resample_tries)
        self.dynamic_masking = bool(dynamic_masking)
        self.masking_mode = str(masking_mode).lower()
        self.n_folds = int(n_folds)
        if not self.files:
            raise ValueError(
                "No files provided to TGSIRandomMaskedPTDataset."
            )
        if not (0.0 < self.context_fraction < 1.0):
            raise ValueError(
                "context_fraction must be between 0 and 1, "
                f"got {self.context_fraction}"
            )
        if not (
            0.0 < self.context_fraction_min
            <= self.context_fraction_max < 1.0
        ):
            raise ValueError(
                "context fraction range must satisfy "
                "0 < min <= max < 1; got "
                f"[{self.context_fraction_min}, {self.context_fraction_max}]"
            )
        if self.masking_mode not in {"random_fraction", "crossfit", "all_available"}:
            raise ValueError(f"Unknown masking_mode: {self.masking_mode!r}")

    def __len__(self):
        return self.samples_per_epoch

    def _empty_sample(self, s, x, available):
        empty = torch.zeros_like(available).bool()
        out = dict(s)
        out["x"] = x
        out["train_mask"] = empty
        out["context_mask"] = empty
        out["loss_mask"] = empty
        return out

    def __getitem__(self, idx):

        if self.dynamic_masking:
            # Use the worker's continuously advancing RNG state.
            # This allows the date and masking pattern to change each epoch.
            rng = random
            gen = None
        else:
            # Deterministic mode:
            # the same idx always returns the same date and gauge split.
            rng = random.Random(self.seed + int(idx))
            gen = torch.Generator()
            gen.manual_seed(self.seed + int(idx))

        chosen = None
        for _ in range(self.max_resample_tries):
            fp = self.files[rng.randrange(len(self.files))]
            s = torch.load(fp, map_location="cpu", weights_only=True)

            required = [
                "x", "y", "train_available_mask", "eval_mask",
                "edge_index_a1"
            ]
            for key in required:
                if key not in s:
                    raise KeyError(f"Missing key '{key}' in {fp}")

            y = s["y"].float()
            available = s["train_available_mask"].bool() & torch.isfinite(y)
            valid_idx = torch.where(available)[0]

            min_required = (
                self.min_loss_gauges
                if self.masking_mode == "all_available"
                else self.min_context_gauges + self.min_loss_gauges
            )
            if int(valid_idx.numel()) >= min_required:
                chosen = (s, valid_idx, available)
                break

        if chosen is None:
            s = torch.load(self.files[rng.randrange(len(self.files))], map_location="cpu", weights_only=True)
            x = s["x"].float().clone()
            y = s["y"].float()
            available = s["train_available_mask"].bool() & torch.isfinite(y)

            x[:, :, self.gauge_value_channel] = 0.0
            x[:, :, self.gauge_mask_channel] = 0.0
            return self._empty_sample(s, x, available)

        s, valid_idx, available = chosen

        x_full = s["x"].float()
        x = x_full.clone()
        y = s["y"].float()

        context_mask = torch.zeros_like(available).bool()
        loss_mask = torch.zeros_like(available).bool()

        # Hide all gauge observations first.
        x[:, :, self.gauge_value_channel] = 0.0
        x[:, :, self.gauge_mask_channel] = 0.0

        n_valid = int(valid_idx.numel())

        if self.masking_mode == "all_available":
            context_idx = torch.empty(0, dtype=torch.long)
            loss_idx = valid_idx
        elif self.masking_mode == "crossfit":
            if "gauge_fold_id" not in s:
                raise KeyError("crossfit masking requires gauge_fold_id in sequence samples")
            fold_id = s["gauge_fold_id"].long()
            valid_folds = [
                fold
                for fold in range(self.n_folds)
                if int((available & (fold_id == fold)).sum()) >= self.min_loss_gauges
                and int((available & (fold_id != fold)).sum()) >= self.min_context_gauges
            ]
            if not valid_folds:
                return self._empty_sample(s, x, available)
            held_fold = valid_folds[rng.randrange(len(valid_folds))]
            context_idx = torch.where(available & (fold_id != held_fold))[0]
            loss_idx = torch.where(available & (fold_id == held_fold))[0]
        elif self.dynamic_masking:
            perm_order = torch.randperm(n_valid)
        else:
            perm_order = torch.randperm(
                n_valid,
                generator=gen,
            )

        if self.masking_mode == "random_fraction":
            perm = valid_idx[perm_order]
            sampled_context_fraction = rng.uniform(
                self.context_fraction_min,
                self.context_fraction_max,
            )
            n_context = int(round(sampled_context_fraction * n_valid))
            n_context = max(self.min_context_gauges, n_context)
            n_context = min(n_context, n_valid - self.min_loss_gauges)
            context_idx = perm[:n_context]
            loss_idx = perm[n_context:]

        required_context = 0 if self.masking_mode == "all_available" else self.min_context_gauges
        if int(context_idx.numel()) < required_context or int(loss_idx.numel()) < self.min_loss_gauges:
            return self._empty_sample(s, x, available)

        context_mask[context_idx] = True
        loss_mask[loss_idx] = True

        # Keep only hidden targets that have at least one currently
        # active context gauge connected through A1.
        if self.masking_mode != "all_available":
            loss_mask = targets_with_active_a1_context(
                target_mask=loss_mask,
                context_mask=context_mask,
                edge_index_a1=s["edge_index_a1"],
            )

        if int(loss_mask.sum()) < self.min_loss_gauges:
            return self._empty_sample(s, x, available)

        # Reveal gauge observations only after target filtering.
        x[:, context_idx, self.gauge_value_channel] = x_full[:, context_idx, self.gauge_value_channel]
        x[:, context_idx, self.gauge_mask_channel] = x_full[:, context_idx, self.gauge_mask_channel]

        out = dict(s)
        out["x"] = x
        out["train_mask"] = loss_mask
        out["context_mask"] = context_mask
        out["loss_mask"] = loss_mask
        return out


class TGSIHoldoutPTDataset(Dataset):
    def __init__(
        self,
        files,
        gauge_value_channel: int = 0,
        gauge_mask_channel: int = 1,
        min_context_gauges: int = 5,
        min_loss_gauges: int = 5,
    ):
        self.files = list(files)
        self.gauge_value_channel = int(gauge_value_channel)
        self.gauge_mask_channel = int(gauge_mask_channel)
        self.min_context_gauges = int(min_context_gauges)
        self.min_loss_gauges = int(min_loss_gauges)

        if not self.files:
            raise ValueError("No files provided to TGSIHoldoutPTDataset.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fp = self.files[idx]
        s = torch.load(fp, map_location="cpu", weights_only=True)

        required = [
            "x", "y", "train_available_mask", "eval_mask",
            "edge_index_a1",
        ]
        for key in required:
            if key not in s:
                raise KeyError(f"Missing key '{key}' in {fp}")

        x_full = s["x"].float()
        x = x_full.clone()
        y = s["y"].float()

        train_available = s["train_available_mask"].bool() & torch.isfinite(y)
        eval_mask = s["eval_mask"].bool() & torch.isfinite(y)

        context_mask = train_available.clone()
        loss_mask = eval_mask.clone()

        loss_mask = targets_with_active_a1_context(

            target_mask=loss_mask,

            context_mask=context_mask,

            edge_index_a1=s["edge_index_a1"],

        )

        x[:, :, self.gauge_value_channel] = 0.0
        x[:, :, self.gauge_mask_channel] = 0.0

        context_idx = torch.where(context_mask)[0]

        if int(context_idx.numel()) >= self.min_context_gauges and int(loss_mask.sum()) >= self.min_loss_gauges:
            x[:, context_idx, self.gauge_value_channel] = x_full[:, context_idx, self.gauge_value_channel]
            x[:, context_idx, self.gauge_mask_channel] = x_full[:, context_idx, self.gauge_mask_channel]
        else:
            loss_mask = torch.zeros_like(loss_mask)

        out = dict(s)
        out["x"] = x
        out["train_mask"] = loss_mask
        out["context_mask"] = context_mask
        out["loss_mask"] = loss_mask
        return out


class TGSICrossfitPTDataset(Dataset):
    def __init__(
        self,
        files,
        n_folds: int = 10,
        gauge_value_channel: int = 0,
        gauge_mask_channel: int = 1,
        min_context_gauges: int = 5,
        min_loss_gauges: int = 1,
    ):
        self.files = list(files)
        self.n_folds = int(n_folds)
        self.gauge_value_channel = int(gauge_value_channel)
        self.gauge_mask_channel = int(gauge_mask_channel)
        self.min_context_gauges = int(min_context_gauges)
        self.min_loss_gauges = int(min_loss_gauges)

    def __len__(self):
        return len(self.files) * self.n_folds

    def __getitem__(self, idx):
        fp = self.files[int(idx) // self.n_folds]
        held_fold = int(idx) % self.n_folds
        s = torch.load(fp, map_location="cpu", weights_only=True)
        if "gauge_fold_id" not in s:
            raise KeyError(f"Missing gauge_fold_id in {fp}")
        y = s["y"].float()
        available = s["train_available_mask"].bool() & torch.isfinite(y)
        fold_id = s["gauge_fold_id"].long()
        context_mask = available & (fold_id != held_fold)
        loss_mask = available & (fold_id == held_fold)
        loss_mask = targets_with_active_a1_context(
            loss_mask, context_mask, s["edge_index_a1"]
        )
        x = s["x"].float().clone()
        x[:, :, self.gauge_value_channel] = 0.0
        x[:, :, self.gauge_mask_channel] = 0.0
        if context_mask.sum() >= self.min_context_gauges and loss_mask.sum() >= self.min_loss_gauges:
            context_idx = torch.where(context_mask)[0]
            x[:, context_idx, self.gauge_value_channel] = s["x"][:, context_idx, self.gauge_value_channel]
            x[:, context_idx, self.gauge_mask_channel] = s["x"][:, context_idx, self.gauge_mask_channel]
        else:
            loss_mask = torch.zeros_like(loss_mask)
        out = dict(s)
        out["x"] = x
        out["train_mask"] = loss_mask
        out["context_mask"] = context_mask
        out["loss_mask"] = loss_mask
        return out


def collate_tgsi(batch):
    x = torch.stack(
        [b["x"] for b in batch],
        dim=0,
    ).float()

    y = torch.stack(
        [b["y"] for b in batch],
        dim=0,
    ).float()

    train_mask = torch.stack(
        [b["train_mask"] for b in batch],
        dim=0,
    ).bool()

    eval_mask = torch.stack(
        [b["eval_mask"] for b in batch],
        dim=0,
    ).bool()

    context_mask = torch.stack(
        [b["context_mask"] for b in batch],
        dim=0,
    ).bool()

    loss_mask = torch.stack(
        [b["loss_mask"] for b in batch],
        dim=0,
    ).bool()

    out = {
        "x": x,
        "y": y,
        "train_mask": train_mask,
        "eval_mask": eval_mask,
        "context_mask": context_mask,
        "loss_mask": loss_mask,
        "date": [b.get("date", None) for b in batch],
        "edge_index_a1": batch[0]["edge_index_a1"].long(),
        "source_nodes": batch[0].get("source_nodes", None),
        "feature_names": batch[0].get("feature_names", None),
    }

    # Retain this for the later edge-aware model, although the
    # naive GAT does not use it.
    if "edge_weight_a1" in batch[0]:
        out["edge_weight_a1"] = (
            batch[0]["edge_weight_a1"].float()
        )

    if "edge_distance_a1_km" in batch[0]:
        out["edge_distance_a1_km"] = (
            batch[0]["edge_distance_a1_km"].float()
        )

    if "node_coords" in batch[0]:
        out["node_coords"] = torch.stack(
            [b["node_coords"] for b in batch],
            dim=0,
        ).float()

    for key in ("train_available_mask", "gauge_fold_id"):
        if key in batch[0]:
            out[key] = torch.stack(
                [b[key] for b in batch],
                dim=0,
            )
    if "train_available_mask" in out:
        out["train_available_mask"] = out["train_available_mask"].bool()
    if "gauge_fold_id" in out:
        out["gauge_fold_id"] = out["gauge_fold_id"].long()

    if "edge_index_a2" in batch[0]:
        out["edge_index_a2"] = batch[0]["edge_index_a2"].long()
        if "edge_weight_a2" in batch[0]:
            out["edge_weight_a2"] = batch[0]["edge_weight_a2"].float()

    return out

class TGSIPTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        seq_dir: str,
        batch_size: int = 8,
        num_workers: int = 4,
        train_years: Tuple[int, int] = (2005, 2018),
        val_years: Optional[Tuple[int, int]] = None,
        test_years: Tuple[int, int] = (2020, 2024),
        exclude_train_years: Optional[Tuple[Tuple[int, int], ...]] = None,
        val_fraction: float = 0.15,
        samples_per_epoch: int = 10000,
        context_fraction: float = 0.50,
        context_fraction_min: Optional[float] = None,
        context_fraction_max: Optional[float] = None,
        gauge_value_channel: int = 0,
        gauge_mask_channel: int = 1,
        min_context_gauges: int = 5,
        min_loss_gauges: int = 5,
        seed: int = 42,
        dynamic_masking: bool = True,
        masking_mode: str = "random_fraction",
        eval_masking_mode: Optional[str] = None,
        n_folds: int = 10,
    ):
        super().__init__()

        self.seq_dir = Path(seq_dir)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.train_years = tuple(train_years)
        self.val_years = tuple(val_years) if val_years is not None else None
        self.test_years = tuple(test_years)
        self.exclude_train_years = tuple(exclude_train_years or ())
        self.val_fraction = float(val_fraction)
        self.samples_per_epoch = int(samples_per_epoch)
        self.context_fraction = float(context_fraction)
        self.context_fraction_min = context_fraction_min
        self.context_fraction_max = context_fraction_max
        self.gauge_value_channel = int(gauge_value_channel)
        self.gauge_mask_channel = int(gauge_mask_channel)
        self.min_context_gauges = int(min_context_gauges)
        self.min_loss_gauges = int(min_loss_gauges)
        self.seed = int(seed)
        self.dynamic_masking = bool(dynamic_masking)
        self.masking_mode = str(masking_mode).lower()
        self.eval_masking_mode = str(eval_masking_mode or masking_mode).lower()
        self.n_folds = int(n_folds)
        if self.eval_masking_mode not in {"random_fraction", "crossfit", "all_available"}:
            raise ValueError(f"Unknown eval_masking_mode: {self.eval_masking_mode!r}")
        self.files = sorted(self.seq_dir.glob("seq_*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No seq_*.pt files found in {self.seq_dir}")

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    @staticmethod
    def _year_from_file(fp: Path) -> int:
        s = torch.load(fp, map_location="cpu", weights_only=True)
        if "date" not in s:
            raise KeyError(f"Missing 'date' key in {fp}")
        return int(str(s["date"])[:4])

    def setup(self, stage=None):
        train_lo, train_hi = self.train_years
        test_lo, test_hi = self.test_years

        train_files, val_files, test_files = [], [], []

        if self.val_years is not None:
            val_lo, val_hi = self.val_years

        for fp in self.files:
            year = self._year_from_file(fp)
            excluded_from_train = any(
                lo <= year <= hi
                for lo, hi in self.exclude_train_years
            )

            if train_lo <= year <= train_hi and not excluded_from_train:
                train_files.append(fp)

            if self.val_years is not None and val_lo <= year <= val_hi:
                val_files.append(fp)

            if test_lo <= year <= test_hi:
                test_files.append(fp)

        train_files = sorted(train_files)
        val_files = sorted(val_files)
        test_files = sorted(test_files)

        if not train_files:
            raise ValueError("Train split is empty. Check train_years.")

        if self.val_years is None:
            rng = random.Random(self.seed)
            shuffled = train_files[:]
            rng.shuffle(shuffled)

            n_val = max(1, int(round(self.val_fraction * len(shuffled))))
            val_files = sorted(shuffled[:n_val])
            train_files = sorted(shuffled[n_val:])

        if not val_files:
            raise ValueError("Validation split is empty.")

        self.train_dataset = TGSIRandomMaskedPTDataset(
            files=train_files,
            samples_per_epoch=self.samples_per_epoch,
            context_fraction=self.context_fraction,
            context_fraction_min=self.context_fraction_min,
            context_fraction_max=self.context_fraction_max,
            gauge_value_channel=self.gauge_value_channel,
            gauge_mask_channel=self.gauge_mask_channel,
            min_context_gauges=self.min_context_gauges,
            min_loss_gauges=self.min_loss_gauges,
            seed=self.seed,
            dynamic_masking=self.dynamic_masking,
            masking_mode=self.masking_mode,
            n_folds=self.n_folds,
        )

        dataset_cls = (
            TGSICrossfitPTDataset
            if self.eval_masking_mode == "crossfit"
            else TGSIHoldoutPTDataset
        )
        val_kwargs = {"n_folds": self.n_folds} if self.eval_masking_mode == "crossfit" else {}
        self.val_dataset = dataset_cls(
            files=val_files,
            gauge_value_channel=self.gauge_value_channel,
            gauge_mask_channel=self.gauge_mask_channel,
            min_context_gauges=self.min_context_gauges,
            min_loss_gauges=self.min_loss_gauges,
            **val_kwargs,
        )

        self.test_dataset = (
            dataset_cls(
                files=test_files,
                gauge_value_channel=self.gauge_value_channel,
                gauge_mask_channel=self.gauge_mask_channel,
                min_context_gauges=self.min_context_gauges,
                min_loss_gauges=self.min_loss_gauges,
                **val_kwargs,
            )
            if test_files
            else None
        )

        rank_zero_info(f"[TGSIDataModule] train files: {len(train_files)}")
        rank_zero_info(f"[TGSIDataModule] val files:   {len(val_files)}")
        rank_zero_info(f"[TGSIDataModule] test files:  {len(test_files)}")
        if self.exclude_train_years:
            rank_zero_info(
                f"[TGSIDataModule] excluded train year ranges: {self.exclude_train_years}"
            )
        rank_zero_info(f"[TGSIDataModule] samples_per_epoch: {self.samples_per_epoch}")
        rank_zero_info(f"[TGSIDataModule] context_fraction: {self.context_fraction}")
        rank_zero_info(
            "[TGSIDataModule] training context fraction range: "
            f"[{self.train_dataset.context_fraction_min}, "
            f"{self.train_dataset.context_fraction_max}]"
        )
        rank_zero_info(f"[TGSIDataModule] eval_masking_mode: {self.eval_masking_mode}")
        rank_zero_info(
            f"[TGSIDataModule] gauge channels: value={self.gauge_value_channel}, "
            f"mask={self.gauge_mask_channel}"
        )
        rank_zero_info(
            f"[TGSIDataModule] dynamic_masking: "
            f"{self.dynamic_masking}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_tgsi,
            drop_last=True,
            worker_init_fn=seed_tgsi_worker,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_tgsi,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return None

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_tgsi,
            drop_last=False,
        )
