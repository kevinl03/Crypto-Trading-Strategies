#!/usr/bin/env python3
"""End-to-end runner for the LSTM z-score notebook pipeline (Phases 1–5)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lstm_zscore_lib as L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs_lstm")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--no-hf", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="Tiny train for plumbing check")
    ap.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Keep every Nth decision time when building sequences (memory)",
    )
    ap.add_argument(
        "--test-stride",
        type=int,
        default=2,
        help="Stride for test sequences (1 = densest eval)",
    )
    args = ap.parse_args()

    cfg = L.TrainConfig()
    if args.max_epochs is not None:
        cfg.max_epochs = args.max_epochs
    if args.seq_len is not None:
        cfg.seq_len = args.seq_len
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.smoke:
        cfg.max_epochs = 2
        cfg.patience = 2
        cfg.seq_len = 16
        cfg.batch_size = 128
        args.stride = max(args.stride, 20)
        args.test_stride = max(args.test_stride, 10)

    L.set_seed(cfg.seed)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    local_root = L.resolve_local_data_root()
    hf_token = L.resolve_hf_token()
    use_hf = not args.no_hf
    print("LOCAL_DATA_ROOT:", local_root)
    print("HF auth:", bool(hf_token), "USE_HF:", use_hf)

    train_windows = [w for w in L.WINDOWS if w["role"] == "train"]
    test_windows = [w for w in L.WINDOWS if w["role"] == "test"]
    if args.smoke:
        # Prefer smallest window for smoke if available after load; still use full WINDOWS list
        # but cap sequences later.
        pass

    print("=" * 60)
    print("Phase 1 — load")
    train_raw = L.pool_windows(
        train_windows, local_root=local_root, hf_token=hf_token, use_hf=use_hf
    )
    test_raw = L.pool_windows(
        test_windows, local_root=local_root, hf_token=hf_token, use_hf=use_hf
    )
    for table in ("spread_matrix", "ticker", "orderbook", "trades"):
        print(f"  {table:20s} train={len(train_raw[table]):,} test={len(test_raw[table]):,}")
    assert len(train_raw["spread_matrix"]) > 0 and len(test_raw["spread_matrix"]) > 0

    print("=" * 60)
    print("Phase 2 — features / sequences")
    import gc

    train_feat = L.apply_row_transforms(L.panel_to_feature_frame(L.build_panel(train_raw)))
    del train_raw
    gc.collect()
    test_feat = L.apply_row_transforms(L.panel_to_feature_frame(L.build_panel(test_raw)))
    del test_raw
    gc.collect()
    winsor_bounds = L.winsorize_fit(train_feat, L.WINSOR_CHANNELS)
    train_feat = L.winsorize_apply(train_feat, winsor_bounds)
    test_feat = L.winsorize_apply(test_feat, winsor_bounds)

    print(f"Building sequences with train_stride={args.stride}, test_stride={args.test_stride}")
    train_bundle, coin_to_id, pair_to_id = L.build_sequences(
        train_feat, seq_len=cfg.seq_len, stride=args.stride
    )
    test_bundle, _, _ = L.build_sequences(
        test_feat,
        seq_len=cfg.seq_len,
        coin_to_id=coin_to_id,
        pair_to_id=pair_to_id,
        stride=args.test_stride,
    )
    print("train", train_bundle.X.shape, "test", test_bundle.X.shape)
    assert len(train_bundle.y) > 0 and len(test_bundle.y) > 0

    if cfg.max_train_seqs and len(train_bundle.y) > cfg.max_train_seqs:
        train_bundle = L.subsample_bundle(train_bundle, cfg.max_train_seqs, keep_tail=True)
        print("subsampled train ->", train_bundle.X.shape)
    if cfg.max_test_seqs and len(test_bundle.y) > cfg.max_test_seqs:
        test_bundle = L.subsample_bundle(test_bundle, cfg.max_test_seqs, keep_tail=False)
        print("subsampled test ->", test_bundle.X.shape)

    print("=" * 60)
    print("Phase 3 — train")
    tr_mask, va_mask = L.chronological_val_split(train_bundle, val_fraction=cfg.val_fraction)
    scaler = L.fit_scaler(train_bundle.X[tr_mask])
    X_tr = L.transform_X(train_bundle.X[tr_mask], scaler)
    X_va = L.transform_X(train_bundle.X[va_mask], scaler)
    X_te = L.transform_X(test_bundle.X, scaler)

    model = L.ZScoreLSTM(
        n_channels=len(L.CHANNEL_NAMES),
        n_coins=len(coin_to_id),
        n_pairs=len(pair_to_id),
        cfg=cfg,
    )
    print("device:", model.device)
    model.fit(
        X_tr,
        train_bundle.y[tr_mask],
        train_bundle.coin_id[tr_mask],
        train_bundle.pair_id[tr_mask],
        X_va,
        train_bundle.y[va_mask],
        train_bundle.coin_id[va_mask],
        train_bundle.pair_id[va_mask],
        checkpoint_path=out / "best_val.pt",
    )

    print("=" * 60)
    print("Phase 4 — eval")
    pred = model.predict(X_te, test_bundle.coin_id, test_bundle.pair_id)
    metrics = L.evaluate_model_and_naive(
        test_bundle.y,
        pred,
        test_bundle.z_now,
        tau=cfg.entry_tau,
        meta=test_bundle.meta,
    )
    print("\n=== HEADLINE (filtered |pred|>tau) ===")
    print("LSTM :", json.dumps(metrics["lstm"]["headline_filtered"], indent=2))
    print("Naive:", json.dumps(metrics["naive_zt"]["headline_filtered"], indent=2))
    print(json.dumps(metrics, indent=2))

    print("=" * 60)
    print("Phase 5 — export")
    L.export_artifacts(
        output_dir=out,
        model=model,
        scaler=scaler,
        coin_to_id=coin_to_id,
        pair_to_id=pair_to_id,
        channel_names=L.CHANNEL_NAMES,
        cfg=cfg,
        metrics=metrics,
        winsor_bounds=winsor_bounds,
    )
    print("done →", out)


if __name__ == "__main__":
    main()
