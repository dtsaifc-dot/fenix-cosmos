import asyncio
import json
import time
from collections import deque

import websockets

from config import BINANCE_WS_TRADES, BINANCE_WS_DEPTH, DEPTH_LEVELS


class BinanceState:
    def __init__(self):
        self.trades = deque(maxlen=4000)
        self.depth = {"bids": [], "asks": []}
        self.last_trade_ts = 0.0
        self.last_depth_ts = 0.0


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


async def collect_trades(state: BinanceState):
    while True:
        try:
            async with websockets.connect(
                BINANCE_WS_TRADES,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=10,
                max_size=2**20,
            ) as ws:
                print("TRADES WS CONNECTED", flush=True)

                async for msg in ws:
                    try:
                        data = json.loads(msg)

                        price = _safe_float(data.get("p"))
                        qty = _safe_float(data.get("q"))
                        trade_ts_ms = data.get("T", 0)
                        is_sell = bool(data.get("m", False))

                        if price <= 0 or qty <= 0:
                            continue

                        trade_ts = (
                            float(trade_ts_ms) / 1000.0
                            if trade_ts_ms
                            else time.time()
                        )

                        state.trades.append({
                            "price": price,
                            "qty": qty,
                            "ts": trade_ts,
                            "is_sell": is_sell,
                        })
                        state.last_trade_ts = trade_ts

                    except Exception as e:
                        print(f"TRADES PARSE ERROR: {e}", flush=True)

        except Exception as e:
            print(f"TRADES WS ERROR: {e}", flush=True)
            await asyncio.sleep(2)


async def collect_depth(state: BinanceState):
    while True:
        try:
            async with websockets.connect(
                BINANCE_WS_DEPTH,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=10,
                max_size=2**20,
            ) as ws:
                print("DEPTH WS CONNECTED", flush=True)

                async for msg in ws:
                    try:
                        data = json.loads(msg)

                        bids_raw = data.get("b") or data.get("bids") or []
                        asks_raw = data.get("a") or data.get("asks") or []

                        bids = []
                        asks = []

                        for row in bids_raw[:DEPTH_LEVELS]:
                            if len(row) < 2:
                                continue
                            price = _safe_float(row[0])
                            qty = _safe_float(row[1])
                            if price > 0 and qty >= 0:
                                bids.append((price, qty))

                        for row in asks_raw[:DEPTH_LEVELS]:
                            if len(row) < 2:
                                continue
                            price = _safe_float(row[0])
                            qty = _safe_float(row[1])
                            if price > 0 and qty >= 0:
                                asks.append((price, qty))

                        # сортировка на всякий случай
                        bids.sort(key=lambda x: x[0], reverse=True)
                        asks.sort(key=lambda x: x[0])

                        state.depth = {
                            "bids": bids,
                            "asks": asks,
                        }
                        state.last_depth_ts = time.time()

                    except Exception as e:
                        print(f"DEPTH PARSE ERROR: {e}", flush=True)

        except Exception as e:
            print(f"DEPTH WS ERROR: {e}", flush=True)
            await asyncio.sleep(2)


async def watchdog(state: BinanceState):
    while True:
        try:
            now = time.time()

            if state.last_trade_ts and now - state.last_trade_ts > 15:
                print("WATCHDOG: trades stale", flush=True)

            if state.last_depth_ts and now - state.last_depth_ts > 15:
                print("WATCHDOG: depth stale", flush=True)

        except Exception as e:
            print(f"WATCHDOG ERROR: {e}", flush=True)

        await asyncio.sleep(5)


async def run_binance_collectors(state: BinanceState):
    await asyncio.gather(
        collect_trades(state),
        collect_depth(state),
        watchdog(state),
    )
