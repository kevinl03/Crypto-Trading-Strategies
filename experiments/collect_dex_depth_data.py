#!/usr/bin/env python3
"""
DEX Depth Collector — execution quotes, gas, and perp funding (separate from collect_dex_data).

Does NOT use DexScreener. Complements the pool-level DEX run with CEX-like depth signals:

  dex_quotes     Jupiter lite (Solana) + Paraswap/1inch (EVM) at $10k notional
  dex_gas        Chain gas / priority-fee snapshots via public RPC
  perp_funding   Hyperliquid perp funding + mark (CEX-adjacent derivative signal)

Usage:
    python -m experiments.collect_dex_depth_data --interval 60 --hours 96
    python -m experiments.collect_dex_depth_data --resume data/dex_depth/20260626_120000

Environment (optional — sources skip gracefully if unset):
    JUPITER_API_KEY   Jupiter quote/price API (Solana execution quotes)
    ONEINCH_API_KEY   1inch swap quote API (EVM execution quotes)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.collect_dex_data import TOKEN_ADDRESSES, _active_tokens

# ── Config ───────────────────────────────────────────────────────────────────

QUOTE_NOTIONAL_USD = 10_000
JUPITER_LITE_QUOTE = "https://lite-api.jup.ag/swap/v1/quote"
JUPITER_QUOTE_BASE = "https://quote-api.jup.ag/v6/quote"
ONEINCH_QUOTE_BASE = "https://api.1inch.dev/swap/v6.0"
PARASWAP_QUOTE_BASE = "https://apiv5.paraswap.io/prices"
ETHERSCAN_GAS_URL = "https://api.etherscan.io/v2/api?chainid=1&module=gastracker&action=gasoracle"
HYPERLIQUID_INFO = "https://api.hyperliquid.xyz/info"

USDC_BY_CHAIN = {
    "solana": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    "ethereum": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    "arbitrum": "0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
    "base": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
    "avalanche": "0xB97EF9Ef8734C71904D8002F8b6Bc66Dd9c48a6E",
    "optimism": "0x0b2C639c533813f4Aa9D7837BAe676af3ad61e7",
}

ONEINCH_CHAIN_ID = {
    "ethereum": 1,
    "arbitrum": 42161,
    "base": 8453,
    "avalanche": 43114,
    "optimism": 10,
}

RPC_URLS = {
    "ethereum": ["https://eth.llamarpc.com", "https://rpc.ankr.com/eth"],
    "arbitrum": ["https://arb1.arbitrum.io/rpc"],
    "base": ["https://mainnet.base.org"],
    "avalanche": ["https://api.avax.network/ext/bc/C/rpc"],
    "optimism": ["https://mainnet.optimism.io"],
    "solana": ["https://api.mainnet-beta.solana.com"],
}

# Decimals for Paraswap / 1inch quote sizing (dest token)
TOKEN_DECIMALS: dict[str, int] = {
    "BTC": 8,   # WBTC
}

EVM_QUOTE_CHAINS = frozenset(ONEINCH_CHAIN_ID)

# Hyperliquid perp symbols aligned with volatile coins
HL_SYMBOLS = {
    "BTC", "ETH", "SOL", "AVAX", "CRV", "LDO", "UNI", "AAVE", "ARB", "OP",
    "PEPE", "WIF", "BONK", "SHIB", "SUI", "DOGE", "XRP", "ADA", "ENA", "WLD",
}

REQUEST_DELAY = 0.35


# ── Writer / state (same pattern as DEX pool collector) ──────────────────────

class DataWriter:
    def __init__(self, output_dir: str):
        self._dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._files: dict[str, Any] = {}

    def write(self, stream: str, record: dict):
        if stream not in self._files:
            path = os.path.join(self._dir, f"{stream}.jsonl")
            self._files[stream] = open(path, "a", encoding="utf-8")
        line = json.dumps(record, default=str)
        self._files[stream].write(line + "\n")
        self._files[stream].flush()
        os.fsync(self._files[stream].fileno())

    def close(self):
        for f in self._files.values():
            f.close()


def _save_state(output_dir: str, state: dict):
    with open(os.path.join(output_dir, "_state.json"), "w") as f:
        json.dump(state, f, default=str)


def _load_state(output_dir: str) -> dict | None:
    path = os.path.join(output_dir, "_state.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _http_json(
    url: str,
    *,
    method: str = "GET",
    body: dict | None = None,
    headers: dict | None = None,
    timeout: float = 15,
    retries: int = 3,
) -> dict | list | None:
    hdrs = {"Accept": "application/json", "User-Agent": "StatArbDexDepth/1.0"}
    if headers:
        hdrs.update(headers)
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        hdrs.setdefault("Content-Type", "application/json")
    last_err = None
    for attempt in range(retries):
        req = Request(url, data=data, headers=hdrs, method=method)
        try:
            with urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except (URLError, HTTPError, json.JSONDecodeError, TimeoutError, ConnectionResetError) as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(0.5 * (attempt + 1))
    print(f"    [WARN] HTTP {method} {url[:80]}: {last_err}", file=sys.stderr)
    return None


def _rpc_call(rpc_url: str, method: str, params: list) -> Any:
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    result = _http_json(rpc_url, method="POST", body=payload)
    if isinstance(result, dict):
        return result.get("result")
    return None


# ── Collectors ───────────────────────────────────────────────────────────────

def _parse_jupiter_quote(data: dict, token: str, usdc: str, mint: str, amount: int,
                         notional_usd: float, source: str) -> dict | None:
    out_amt = int(data.get("outAmount") or 0)
    if out_amt <= 0:
        return None
    price_impact = data.get("priceImpactPct")
    return {
        "token": token,
        "chain": "solana",
        "source": source,
        "side": "buy",
        "notional_usd": notional_usd,
        "in_mint": usdc,
        "out_mint": mint,
        "in_amount": amount,
        "out_amount": out_amt,
        "price_impact_pct": float(price_impact) if price_impact is not None else None,
        "route_plan_len": len(data.get("routePlan") or []),
    }


def fetch_jupiter_quote(
    token: str, mint: str, notional_usd: float, api_key: str | None
) -> dict | None:
    """Buy `token` with USDC on Solana via Jupiter lite (no key) or v6 (key)."""
    usdc = USDC_BY_CHAIN["solana"]
    amount = int(notional_usd * 1_000_000)
    q = f"inputMint={usdc}&outputMint={mint}&amount={amount}&slippageBps=50"

    data = _http_json(f"{JUPITER_LITE_QUOTE}?{q}")
    if isinstance(data, dict) and data.get("outAmount"):
        return _parse_jupiter_quote(data, token, usdc, mint, amount, notional_usd, "jupiter_lite")

    if api_key:
        data = _http_json(f"{JUPITER_QUOTE_BASE}?{q}", headers={"x-api-key": api_key})
        if isinstance(data, dict) and data.get("outAmount"):
            return _parse_jupiter_quote(data, token, usdc, mint, amount, notional_usd, "jupiter")
    return None


def fetch_paraswap_quote(
    token: str, chain: str, token_addr: str, notional_usd: float
) -> dict | None:
    network = ONEINCH_CHAIN_ID.get(chain)
    usdc = USDC_BY_CHAIN.get(chain)
    if network is None or not usdc:
        return None
    amount = int(notional_usd * 1_000_000)
    dest_dec = TOKEN_DECIMALS.get(token, 18)
    url = (
        f"{PARASWAP_QUOTE_BASE}/"
        f"?srcToken={usdc}&destToken={token_addr}&amount={amount}"
        f"&srcDecimals=6&destDecimals={dest_dec}&side=SELL&network={network}"
    )
    data = _http_json(url)
    if not isinstance(data, dict):
        return None
    route = data.get("priceRoute") or {}
    dst_amt = int(route.get("destAmount") or 0)
    if dst_amt <= 0:
        return None
    return {
        "token": token,
        "chain": chain,
        "source": "paraswap",
        "side": "buy",
        "notional_usd": notional_usd,
        "src_token": usdc,
        "dst_token": token_addr,
        "src_amount": amount,
        "dst_amount": dst_amt,
        "gas": route.get("gasCost"),
        "price_impact_pct": route.get("priceImpact"),
    }


def fetch_evm_quote(
    token: str, chain: str, token_addr: str, notional_usd: float, oneinch_key: str | None
) -> dict | None:
    if oneinch_key:
        q = fetch_oneinch_quote(token, chain, token_addr, notional_usd, oneinch_key)
        if q:
            return q
    return fetch_paraswap_quote(token, chain, token_addr, notional_usd)
def fetch_oneinch_quote(
    token: str, chain: str, token_addr: str, notional_usd: float, api_key: str
) -> dict | None:
    chain_id = ONEINCH_CHAIN_ID.get(chain)
    usdc = USDC_BY_CHAIN.get(chain)
    if chain_id is None or not usdc:
        return None
    amount = int(notional_usd * 1_000_000)
    url = (
        f"{ONEINCH_QUOTE_BASE}/{chain_id}/quote"
        f"?src={usdc}&dst={token_addr}&amount={amount}"
    )
    headers = {"Authorization": f"Bearer {api_key}"}
    data = _http_json(url, headers=headers)
    if not isinstance(data, dict):
        return None
    dst_amt = int(data.get("dstAmount") or data.get("toAmount") or 0)
    if dst_amt <= 0:
        return None
    return {
        "token": token,
        "chain": chain,
        "source": "1inch",
        "side": "buy",
        "notional_usd": notional_usd,
        "src_token": usdc,
        "dst_token": token_addr,
        "src_amount": amount,
        "dst_amount": dst_amt,
        "gas": data.get("gas"),
        "protocols": data.get("protocols"),
    }


def fetch_ethereum_gas_etherscan() -> dict | None:
    data = _http_json(ETHERSCAN_GAS_URL)
    if not isinstance(data, dict) or data.get("status") != "1":
        return None
    r = data.get("result") or {}
    gwei = r.get("ProposeGasPrice") or r.get("SafeGasPrice")
    if gwei is None:
        return None
    return {
        "chain": "ethereum",
        "gas_gwei": round(float(gwei), 4),
        "priority_fee_lamports": None,
        "rpc": "etherscan_gasoracle",
    }


def fetch_gas_snapshots() -> list[dict]:
    rows = []
    for chain, rpcs in RPC_URLS.items():
        if chain == "solana":
            result, used_rpc = None, rpcs[0]
            for rpc in rpcs:
                result = _rpc_call(rpc, "getRecentPrioritizationFees", [[]])
                if result is not None:
                    used_rpc = rpc
                    break
            fee = None
            if isinstance(result, list) and result:
                fee = result[-1].get("prioritizationFee")
            rows.append({
                "chain": chain,
                "gas_gwei": None,
                "priority_fee_lamports": fee,
                "rpc": used_rpc,
            })
            continue

        gwei_hex, used_rpc = None, rpcs[0]
        for rpc in rpcs:
            gwei_hex = _rpc_call(rpc, "eth_gasPrice", [])
            if gwei_hex is not None:
                used_rpc = rpc
                break
        gwei = int(gwei_hex, 16) / 1e9 if isinstance(gwei_hex, str) else None
        if gwei is None and chain == "ethereum":
            fallback = fetch_ethereum_gas_etherscan()
            if fallback:
                rows.append(fallback)
                continue
        rows.append({
            "chain": chain,
            "gas_gwei": round(gwei, 4) if gwei is not None else None,
            "priority_fee_lamports": None,
            "rpc": used_rpc,
        })
    return rows


def fetch_hyperliquid_funding() -> list[dict]:
    data = _http_json(HYPERLIQUID_INFO, method="POST", body={"type": "metaAndAssetCtxs"})
    if not isinstance(data, list) or len(data) < 2:
        return []
    meta, ctxs = data[0], data[1]
    universe = meta.get("universe") or []
    rows = []
    for asset, ctx in zip(universe, ctxs):
        sym = asset.get("name", "")
        if sym not in HL_SYMBOLS:
            continue
        funding = ctx.get("funding")
        mark = ctx.get("markPx")
        oi = ctx.get("openInterest")
        rows.append({
            "symbol": sym,
            "source": "hyperliquid",
            "funding_rate": float(funding) if funding is not None else None,
            "mark_px": float(mark) if mark is not None else None,
            "open_interest": float(oi) if oi is not None else None,
        })
    return rows


def collect_snapshot(
    writer: DataWriter,
    snapshot_idx: int,
    ts: str,
    *,
    jupiter_key: str | None,
    oneinch_key: str | None,
    notional_usd: float,
) -> dict:
    stats = {"quotes": 0, "gas": 0, "funding": 0}

    for token, chains in _active_tokens().items():
        for chain, address in chains.items():
            quote = None
            if chain == "solana":
                quote = fetch_jupiter_quote(token, address, notional_usd, jupiter_key)
                time.sleep(REQUEST_DELAY)
            elif chain in EVM_QUOTE_CHAINS:
                quote = fetch_evm_quote(token, chain, address, notional_usd, oneinch_key)
                time.sleep(REQUEST_DELAY)
            if quote:
                quote["snapshot_idx"] = snapshot_idx
                quote["timestamp"] = ts
                writer.write("dex_quotes", quote)
                stats["quotes"] += 1

    for row in fetch_gas_snapshots():
        row["snapshot_idx"] = snapshot_idx
        row["timestamp"] = ts
        writer.write("dex_gas", row)
        stats["gas"] += 1

    for row in fetch_hyperliquid_funding():
        row["snapshot_idx"] = snapshot_idx
        row["timestamp"] = ts
        writer.write("perp_funding", row)
        stats["funding"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="DEX depth collector (quotes, gas, perp funding)")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--hours", type=float, default=96.0)
    parser.add_argument("--notional-usd", type=float, default=QUOTE_NOTIONAL_USD,
                        help="Quote size in USD for execution quotes (default: 10000)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    jupiter_key = os.environ.get("JUPITER_API_KEY")
    oneinch_key = os.environ.get("ONEINCH_API_KEY")

    if args.resume:
        output_dir = args.resume
        state = _load_state(output_dir)
        if state:
            start_snapshot = state.get("last_snapshot", 0) + 1
            start_time = datetime.fromisoformat(state["start_time"])
            remaining_hours = max(0, args.hours - (datetime.now(timezone.utc) - start_time).total_seconds() / 3600)
            print(f"  [RESUME] snapshot {start_snapshot}, {remaining_hours:.1f}h remaining")
        else:
            start_snapshot = 0
            start_time = datetime.now(timezone.utc)
            remaining_hours = args.hours
    else:
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = args.output_dir or os.path.join("data", "dex_depth", ts_str)
        start_snapshot = 0
        start_time = datetime.now(timezone.utc)
        remaining_hours = args.hours

    total_snapshots = int(remaining_hours * 3600 / args.interval)
    if total_snapshots <= 0:
        print("No snapshots remaining.")
        return

    writer = DataWriter(output_dir)
    writer.write("run_config", {
        "start_time": start_time.isoformat(),
        "interval_s": args.interval,
        "hours": args.hours,
        "notional_usd": args.notional_usd,
        "tokens": list(_active_tokens()),
        "streams": ["dex_quotes", "dex_gas", "perp_funding"],
        "jupiter_key_set": bool(jupiter_key),
        "oneinch_key_set": bool(oneinch_key),
        "snapshot": 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    print("=== DEX Depth Collector ===")
    print(f"  Output: {os.path.abspath(output_dir)}")
    print(f"  Interval: {args.interval}s | Duration: {remaining_hours:.1f}h (~{total_snapshots} snaps)")
    print(f"  Jupiter: lite-api (no key) + optional v6 key")
    print(f"  EVM quotes: Paraswap (default) + optional 1inch key")
    print(f"  Gas + Hyperliquid funding: always on\n")

    snap_idx = start_snapshot - 1
    try:
        for i in range(total_snapshots):
            snap_idx = start_snapshot + i
            ts = datetime.now(timezone.utc).isoformat()
            t0 = time.time()
            stats = collect_snapshot(
                writer, snap_idx, ts,
                jupiter_key=jupiter_key,
                oneinch_key=oneinch_key,
                notional_usd=args.notional_usd,
            )
            elapsed = time.time() - t0
            print(f"  [{snap_idx + 1:>5}] quotes={stats['quotes']} gas={stats['gas']} "
                  f"funding={stats['funding']} ({elapsed:.1f}s)")
            _save_state(output_dir, {
                "last_snapshot": snap_idx,
                "start_time": start_time.isoformat(),
                "notional_usd": args.notional_usd,
            })
            sleep_time = max(0, args.interval - elapsed)
            if sleep_time > 0 and i < total_snapshots - 1:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\n  [STOPPED] Graceful shutdown.")
    finally:
        writer.close()
        print(f"\n=== Done === snapshots through {snap_idx + 1 if snap_idx >= start_snapshot else start_snapshot}")


if __name__ == "__main__":
    main()
