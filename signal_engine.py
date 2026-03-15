import math
from datetime import datetime, timezone, timedelta

from config import NO_TRADE_THRESHOLD, BAD_SIGNAL_THRESHOLD


def current_5m_open(now=None):
    now = now or datetime.now(timezone.utc)
    minute_bucket = (now.minute // 5) * 5
    return now.replace(minute=minute_bucket, second=0, microsecond=0)


def next_5m_open(now=None):
    return current_5m_open(now) + timedelta(minutes=5)


def seconds_to_next_5m(now=None):
    now = now or datetime.now(timezone.utc)
    nxt = next_5m_open(now)
    return max(0.0, (nxt - now).total_seconds())


def phase_name(now=None, hide_sec=15, lock_sec=8):
    sec = seconds_to_next_5m(now)
    if sec > hide_sec:
        return "LIVE"
    if hide_sec >= sec > lock_sec:
        return "PREPARE"
    return "LOCK"


def should_lock_signal(now=None, lead_sec=8):
    return seconds_to_next_5m(now) <= lead_sec


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def _safe_window(features, name):
    w = features.get(name, {}) or {}
    return {
        "buy_pct": float(w.get("buy_pct", 50.0)),
        "sell_pct": float(w.get("sell_pct", 50.0)),
        "delta_ratio": float(w.get("delta_ratio", 0.0)),
        "cum_delta": float(w.get("cum_delta", 0.0)),
        "slope": float(w.get("slope", 0.0)),
        "burst_ratio": float(w.get("burst_ratio", 1.0)),
        "streak_bias": float(w.get("streak_bias", 0.0)),
        "count": float(w.get("count", 0)),
    }


def _sigmoid_conf(score: float):
    x = _clip(score / 2.8, -8.0, 8.0)
    p_up = 1.0 / (1.0 + math.exp(-x))
    p_down = 1.0 - p_up
    confidence = max(p_up, p_down) * 100.0
    return p_up * 100.0, p_down * 100.0, confidence


def _flow_score(w30, w10, w3):
    # Основа сигнала: order flow
    s30 = ((w30["buy_pct"] - 50.0) / 10.0) * 0.8
    s10 = ((w10["buy_pct"] - 50.0) / 10.0) * 1.8
    s3 = ((w3["buy_pct"] - 50.0) / 10.0) * 1.2
    return s30 + s10 + s3


def _delta_score(w30, w10, w3):
    # Delta и cum delta
    d30 = _clip(w30["delta_ratio"] * 10.0, -2.0, 2.0) * 0.6
    d10 = _clip(w10["delta_ratio"] * 10.0, -2.5, 2.5) * 1.2
    d3 = _clip(w3["delta_ratio"] * 10.0, -2.0, 2.0) * 0.8
    return d30 + d10 + d3


def _slope_score(w10, w3):
    # Локальный микро-тренд цены
    s10 = _clip(w10["slope"] / 4.0, -2.0, 2.0) * 1.1
    s3 = _clip(w3["slope"] / 2.0, -2.0, 2.0) * 0.9
    return s10 + s3


def _depth_score(features):
    imbalance = float(features.get("imbalance", 0.0))
    micro_bias_bps = float(features.get("micro_bias_bps", 0.0))
    bid_ask_ratio = float(features.get("bid_ask_ratio", 1.0))

    s_imb = _clip(imbalance / 0.12, -2.5, 2.5) * 1.4
    s_micro = _clip(micro_bias_bps / 1.0, -2.0, 2.0) * 0.8
    s_ratio = _clip((bid_ask_ratio - 1.0) / 0.35, -2.0, 2.0) * 0.7

    return s_imb + s_micro + s_ratio


def _alignment_bonus(w30, w10, w3):
    signs = []

    for w in (w30, w10, w3):
        if w["buy_pct"] >= 53:
            signs.append(1)
        elif w["buy_pct"] <= 47:
            signs.append(-1)
        else:
            signs.append(0)

    if signs == [1, 1, 1]:
        return 1.2
    if signs == [-1, -1, -1]:
        return -1.2
    return 0.0


def _burst_bonus(w3):
    burst = w3["burst_ratio"]
    bias = (w3["buy_pct"] - 50.0) / 10.0

    if burst >= 1.7:
        return _clip(bias, -2.0, 2.0) * 0.7
    return 0.0


def _contradiction_penalty(features, w10, w3):
    penalty = 0.0
    reasons = []

    imbalance = float(features.get("imbalance", 0.0))
    micro_bias_bps = float(features.get("micro_bias_bps", 0.0))
    divergence = features.get("divergence")

    # Flow BUY, depth SELL
    if w10["buy_pct"] >= 58 and imbalance <= -0.18:
        penalty += 1.4
        reasons.append("flow/depth conflict")
    if w10["sell_pct"] >= 58 and imbalance >= 0.18:
        penalty += 1.4
        reasons.append("flow/depth conflict")

    # Very short flow against 10s flow
    if w3["buy_pct"] >= 60 and w10["sell_pct"] >= 58:
        penalty += 0.8
        reasons.append("3s vs 10s conflict")
    if w3["sell_pct"] >= 60 and w10["buy_pct"] >= 58:
        penalty += 0.8
        reasons.append("3s vs 10s conflict")

    # Microprice conflict
    if w10["buy_pct"] >= 58 and micro_bias_bps < -0.8:
        penalty += 0.7
        reasons.append("microprice against buy")
    if w10["sell_pct"] >= 58 and micro_bias_bps > 0.8:
        penalty += 0.7
        reasons.append("microprice against sell")

    # Divergence only as penalty, not reversal engine
    if divergence == "bearish":
        penalty += 0.6
        reasons.append("bearish divergence")
    elif divergence == "bullish":
        penalty += 0.6
        reasons.append("bullish divergence")

    return penalty, reasons


def classify_status(confidence: float, abs_score: float):
    if confidence < NO_TRADE_THRESHOLD:
        return "NO TRADE", "NO TRADE"

    if abs_score < 2.2:
        return "NO TRADE", "NO TRADE"

    if confidence < BAD_SIGNAL_THRESHOLD:
        return "BAD SIGNAL", None

    return "GOOD SIGNAL", None


def build_signal(features: dict, ai_pred: dict | None = None):
    # AI здесь не используем для решения сигнала.
    # Можно оставить только для отображения, но не для торговли.
    w30 = _safe_window(features, "w30")
    w10 = _safe_window(features, "w10")
    w3 = _safe_window(features, "w3")

    score = 0.0
    reasons = []

    flow = _flow_score(w30, w10, w3)
    delta = _delta_score(w30, w10, w3)
    slope = _slope_score(w10, w3)
    depth = _depth_score(features)
    align = _alignment_bonus(w30, w10, w3)
    burst = _burst_bonus(w3)

    score += flow
    score += delta
    score += slope
    score += depth
    score += align
    score += burst

    penalty, penalty_reasons = _contradiction_penalty(features, w10, w3)
    score -= penalty

    if penalty_reasons:
        reasons.extend(penalty_reasons)

    direction = "UP" if score >= 0 else "DOWN"
    abs_score = abs(score)

    p_up, p_down, confidence = _sigmoid_conf(score)

    # Жёсткий no-trade фильтр
    # Если локальный edge слабый — не торгуем
    no_trade = False

    if abs_score < 2.2:
        no_trade = True
    if 47.5 <= w10["buy_pct"] <= 52.5:
        no_trade = True
    if abs(float(features.get("imbalance", 0.0))) < 0.06 and abs(w10["delta_ratio"]) < 0.06:
        no_trade = True

    if no_trade:
        display_signal = "NO TRADE"
        status = "NO TRADE"
    else:
        status, forced_display = classify_status(confidence, abs_score)
        display_signal = forced_display if forced_display else direction

    # Reasons for dashboard
    if w10["buy_pct"] >= 56:
        reasons.append(f"10s flow bullish ({w10['buy_pct']:.1f}% buy)")
    elif w10["sell_pct"] >= 56:
        reasons.append(f"10s flow bearish ({w10['sell_pct']:.1f}% sell)")

    if abs(w3["buy_pct"] - 50.0) >= 10:
        if w3["buy_pct"] > 50:
            reasons.append(f"3s tape supports buy ({w3['buy_pct']:.1f}% buy)")
        else:
            reasons.append(f"3s tape supports sell ({w3['sell_pct']:.1f}% sell)")

    imbalance = float(features.get("imbalance", 0.0))
    if abs(imbalance) >= 0.10:
        reasons.append(f"depth imbalance {imbalance:+.3f}")

    micro_bias_bps = float(features.get("micro_bias_bps", 0.0))
    if abs(micro_bias_bps) >= 0.7:
        reasons.append(f"microprice drift {micro_bias_bps:+.2f} bps")

    if abs(w10["delta_ratio"]) >= 0.08:
        reasons.append(f"10s delta ratio {w10['delta_ratio']:+.3f}")

    if abs(w10["slope"]) >= 1.5:
        reasons.append(f"10s slope {w10['slope']:+.2f}")

    if abs(w3["burst_ratio"] - 1.0) >= 0.5:
        reasons.append(f"burst x{w3['burst_ratio']:.2f}")

    divergence = features.get("divergence")
    if divergence:
        reasons.append(f"{divergence} divergence")

    # Для совместимости с app.py / dashboard
    ai_conf = 50.0
    ai_direction = "NEUTRAL"

    return {
        "display_signal": display_signal,
        "direction": direction,
        "status": status,
        "regime": "FLOW+DEPTH",
        "confidence": round(confidence, 1),
        "score": round(score, 2),
        "p_up": round(p_up, 1),
        "p_down": round(p_down, 1),
        "rule_conf": round(confidence, 1),
        "ai_conf": round(ai_conf, 1),
        "ai_direction": ai_direction,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_open_utc": next_5m_open().isoformat(),
        "reasons": reasons[:10],
    }
