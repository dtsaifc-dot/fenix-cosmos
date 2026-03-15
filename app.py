import asyncio
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template, make_response

from collector_binance import BinanceState, run_binance_collectors
from config import DASHBOARD_FILE, SIGNAL_FILE, HISTORY_FILE, ENTRY_LEAD_SEC, HIDE_SIGNAL_SEC
from features import build_features
from signal_engine import (
    build_signal,
    should_lock_signal,
    next_5m_open,
    current_5m_open,
    seconds_to_next_5m,
    phase_name,
)

AI_ENABLED = True
try:
    from ai_model import predict_ai, train_on_resolved_history
except Exception:
    AI_ENABLED = False

    def predict_ai(features):
        return {
            "direction": "UP" if features.get("buy_pct", 50) >= 50 else "DOWN",
            "p_up": 50.0,
            "p_down": 50.0,
            "confidence": 50.0,
            "raw_score": 0.0,
            "vector": {},
            "samples": 0,
        }

    def train_on_resolved_history(history):
        return False, {"samples": 0}


app = Flask(__name__)
state = BinanceState()

dashboard_cache = {
    "features_live": {},
    "signal_locked": {},
    "signal_preview": {},
    "history": [],
    "stats": {},
    "meta": {},
}

lock = threading.Lock()

history = []
current_locked = None
locked_target = None
last_candle_open_seen = None


def load_history():
    p = Path(HISTORY_FILE)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_history(hist):
    Path(HISTORY_FILE).write_text(
        json.dumps(hist, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def utc_now():
    return datetime.now(timezone.utc)


def direction_from_open_close(open_price, close_price):
    if open_price is None or close_price is None:
        return None
    if close_price > open_price:
        return "UP"
    if close_price < open_price:
        return "DOWN"
    return "FLAT"


def compute_stats(hist):
    resolved = [x for x in hist if x.get("result") in ("WIN", "LOSS")]
    wins = sum(1 for x in resolved if x["result"] == "WIN")

    up_rows = [x for x in resolved if x.get("signal") == "UP"]
    down_rows = [x for x in resolved if x.get("signal") == "DOWN"]

    up_wins = sum(1 for x in up_rows if x["result"] == "WIN")
    down_wins = sum(1 for x in down_rows if x["result"] == "WIN")

    up_wr = (up_wins / len(up_rows) * 100.0) if up_rows else 0.0
    down_wr = (down_wins / len(down_rows) * 100.0) if down_rows else 0.0
    overall_wr = (wins / len(resolved) * 100.0) if resolved else 0.0

    last_15 = []
    for x in hist:
        if x.get("result") == "WIN":
            last_15.append("+")
        elif x.get("result") == "LOSS":
            last_15.append("-")
        if len(last_15) >= 15:
            break

    pending_queue = []
    for x in hist:
        if x.get("result") == "PENDING":
            pending_queue.append({
                "signal": x.get("signal"),
                "open": x.get("open"),
                "conf": x.get("confidence"),
                "record_ts": x.get("signal_time"),
                "next_candle_open_ts": x.get("target_candle"),
            })

    return {
        "up_total": len(up_rows),
        "up_wins": up_wins,
        "down_total": len(down_rows),
        "down_wins": down_wins,
        "last_15": last_15,
        "pending_queue": pending_queue,
        "pending": pending_queue[0] if pending_queue else None,
        "winrate_up": round(up_wr, 1),
        "winrate_down": round(down_wr, 1),
        "winrate_total": round(overall_wr, 1),
    }


def classify_strength(confidence, status):
    if status == "NO TRADE":
        return "WEAK"
    if confidence >= 82:
        return "INSTITUTIONAL"
    if confidence >= 72:
        return "STRONG"
    if confidence >= 60:
        return "MEDIUM"
    return "WEAK"


def build_preview_ui(signal_obj, features_live, ai_pred, now, next_open, phase):
    return {
        "display_signal": signal_obj.get("display_signal", "COLLECTING"),
        "direction": signal_obj.get("direction", "-"),
        "status": signal_obj.get("status", "LIVE STATE"),
        "regime": signal_obj.get("regime", features_live.get("regime_live", "-")),
        "confidence": signal_obj.get("confidence", 50.0),
        "score": signal_obj.get("score", 0.0),
        "generated_at": signal_obj.get("generated_at", now.isoformat()),
        "target_open_utc": next_open,
        "seconds_left": round(seconds_to_next_5m(now), 1),
        "phase": phase,
        "reasons": signal_obj.get("reasons", []),
        "rule_conf": signal_obj.get("rule_conf", signal_obj.get("confidence", 50.0)),
        "ai_conf": signal_obj.get("ai_conf", ai_pred.get("confidence", 50.0)),
        "ai_direction": signal_obj.get("ai_direction", ai_pred.get("direction", "-")),
        "strength": classify_strength(
            signal_obj.get("confidence", 50.0),
            signal_obj.get("status", "LIVE STATE"),
        ),
        "decision": signal_obj.get("status", "WATCH"),
        "size_mode": None,
        "risk_label": None,
        "risk_score": None,
        "risk_reasons": [],
        "regime_type": features_live.get("regime_live", "-"),
        "regime_confidence": 0.0,
        "regime_direction": features_live.get("global_trend_1h", "NEUTRAL"),
    }


def on_new_candle_start(current_open_iso, current_price):
    global history

    for row in history:
        if row.get("target_candle") == current_open_iso and row.get("open") is None:
            row["open"] = round(current_price, 2) if current_price is not None else None

    for row in history:
        target = row.get("target_candle")
        if row.get("result") == "PENDING" and target and target < current_open_iso and row.get("open") is not None:
            close_price = round(current_price, 2) if current_price is not None else None
            row["close"] = close_price
            actual = direction_from_open_close(row["open"], close_price)
            row["actual"] = actual

            if row.get("status") == "NO TRADE":
                row["result"] = "SKIP"
            else:
                row["result"] = "WIN" if actual == row.get("signal") else "LOSS"

    if AI_ENABLED:
        try:
            changed, _ = train_on_resolved_history(history)
            if changed:
                save_history(history)
        except Exception as e:
            print(f"AI TRAIN ERROR: {e}", flush=True)

    history[:] = history[:500]
    save_history(history)


def collector_thread():
    asyncio.run(run_binance_collectors(state))


def engine_thread():
    global current_locked, locked_target, last_candle_open_seen, history

    history = load_history()

    while True:
        try:
            now = utc_now()
            features_live = build_features(state)

            current_open = current_5m_open(now).isoformat()
            next_open = next_5m_open(now).isoformat()
            last_price = features_live.get("last_price")
            phase = phase_name(now, HIDE_SIGNAL_SEC, ENTRY_LEAD_SEC)

            if last_candle_open_seen is None:
                last_candle_open_seen = current_open

            if current_open != last_candle_open_seen:
                on_new_candle_start(current_open, last_price)
                last_candle_open_seen = current_open

            try:
                ai_pred = predict_ai(features_live)
            except Exception as e:
                print(f"AI PREDICT ERROR: {e}", flush=True)
                ai_pred = {
                    "direction": "UP" if features_live.get("buy_pct", 50) >= 50 else "DOWN",
                    "p_up": 50.0,
                    "p_down": 50.0,
                    "confidence": 50.0,
                    "raw_score": 0.0,
                    "vector": {},
                    "samples": 0,
                }

            preview_signal = build_signal(features_live, ai_pred)
            preview_ui = build_preview_ui(preview_signal, features_live, ai_pred, now, next_open, phase)

            if current_locked is None:
                locked_ui = dict(preview_ui)

                if phase == "LIVE":
                    locked_ui["display_signal"] = "COLLECTING"
                    locked_ui["direction"] = "-"
                    locked_ui["status"] = "COLLECTING DATA"
                elif phase == "PREPARE":
                    locked_ui["display_signal"] = "PREPARE"
                    locked_ui["direction"] = preview_ui.get("direction", "-")
                    locked_ui["status"] = "LOCK WINDOW"
            else:
                locked_ui = dict(current_locked)

            if should_lock_signal(now, ENTRY_LEAD_SEC):
                if locked_target != next_open:
                    current_locked = dict(preview_ui)
                    current_locked["generated_at"] = now.isoformat()
                    current_locked["target_open_utc"] = next_open
                    current_locked["seconds_left"] = round(seconds_to_next_5m(now), 1)
                    current_locked["strength"] = classify_strength(
                        current_locked.get("confidence", 50.0),
                        current_locked.get("status", "GOOD SIGNAL"),
                    )

                    locked_target = next_open
                    locked_ui = dict(current_locked)

                    history.insert(0, {
                        "signal_time": now.isoformat(),
                        "target_candle": next_open,
                        "signal": current_locked.get("direction"),
                        "display_signal": current_locked.get("display_signal"),
                        "status": current_locked.get("status"),
                        "regime": current_locked.get("regime"),
                        "strength": current_locked.get("strength"),
                        "confidence": current_locked.get("confidence"),
                        "score": current_locked.get("score"),
                        "decision": current_locked.get("status"),
                        "size_mode": None,
                        "risk_label": None,
                        "risk_score": None,
                        "buy_pct": (
                            features_live["w10"]["buy_pct"]
                            if "w10" in features_live and isinstance(features_live["w10"], dict)
                            else features_live.get("buy_pct")
                        ),
                        "imbalance": features_live.get("imbalance"),
                        "cum_delta": (
                            features_live["w10"]["cum_delta"]
                            if "w10" in features_live and isinstance(features_live["w10"], dict)
                            else features_live.get("cum_delta")
                        ),
                        "open": None,
                        "close": None,
                        "actual": None,
                        "result": "PENDING" if current_locked.get("status") != "NO TRADE" else "SKIP",
                        "feature_snapshot": ai_pred.get("vector", {}) if ai_pred else {},
                        "ai_trained": False,
                    })

                    history = history[:500]
                    save_history(history)

            if current_locked is not None:
                current_locked["seconds_left"] = round(seconds_to_next_5m(now), 1)
                current_locked["phase"] = phase
                locked_ui = dict(current_locked)

            if locked_target and current_open == locked_target and phase == "LIVE":
                current_locked = None
                locked_target = None

            stats = compute_stats(history)

            with lock:
                dashboard_cache["features_live"] = features_live
                dashboard_cache["signal_preview"] = preview_ui
                dashboard_cache["signal_locked"] = locked_ui
                dashboard_cache["history"] = history
                dashboard_cache["stats"] = stats
                dashboard_cache["meta"] = {
                    "phase": phase,
                    "entry_lead_sec": ENTRY_LEAD_SEC,
                    "hide_signal_sec": HIDE_SIGNAL_SEC,
                    "server_time": now.isoformat(),
                    "locked_target": locked_target,
                    "ai_enabled": AI_ENABLED,
                    "ai_samples": ai_pred.get("samples", 0) if ai_pred else 0,
                    "ai_direction": ai_pred.get("direction", "-") if ai_pred else "-",
                    "ai_confidence": ai_pred.get("confidence", 50.0) if ai_pred else 50.0,
                    "decision": preview_ui.get("status", "WATCH"),
                    "size_mode": None,
                    "risk_label": None,
                    "risk_score": None,
                    "global_trend_1h": features_live.get("global_trend_1h", "-"),
                    "global_bias_15m": features_live.get("global_bias_15m", "-"),
                    "liquidity_sweep": features_live.get("liquidity_sweep", "NONE"),
                    "liquidity_sweep_dir": features_live.get("liquidity_sweep_dir", "NONE"),
                }

            Path(DASHBOARD_FILE).write_text(
                json.dumps(dashboard_cache, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            Path(SIGNAL_FILE).write_text(
                json.dumps(locked_ui, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )

        except Exception as e:
            print(f"ENGINE ERROR: {e}", flush=True)

        time.sleep(1)


@app.route("/api/dashboard")
def api_dashboard():
    with lock:
        resp = make_response(jsonify(dashboard_cache))
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp


@app.route("/api/signal")
def api_signal():
    with lock:
        resp = make_response(jsonify(dashboard_cache["signal_locked"]))
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp


@app.route("/")
def dashboard():
    return render_template("dashboard.html")


if __name__ == "__main__":
    t1 = threading.Thread(target=collector_thread, daemon=True)
    t2 = threading.Thread(target=engine_thread, daemon=True)
    t1.start()
    t2.start()
    app.run(host="0.0.0.0", port=8091, debug=False)
