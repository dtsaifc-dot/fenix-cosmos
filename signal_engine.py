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


def sigmoid_conf(score: float):
    x = min(25.0, max(-25.0, score / 6.0))
    p_up = 1.0 / (1.0 + math.exp(-x))
    p_down = 1.0 - p_up
    confidence = max(p_up, p_down) * 100.0
    return p_up * 100.0, p_down * 100.0, confidence


def _safe_window(features, name):
    w = features.get(name, {})
    return {
        "buy_pct": float(w.get("buy_pct", 50.0)),
        "sell_pct": float(w.get("sell_pct", 50.0)),
        "delta_ratio": float(w.get("delta_ratio", 0.0)),
        "cum_delta": float(w.get("cum_delta", 0.0)),
        "slope": float(w.get("slope", 0.0)),
        "burst_ratio": float(w.get("burst_ratio", 1.0)),
        "streak_bias": float(w.get("streak_bias", 0.0)),
    }


def _edge_from_window(w: dict) -> float:
    flow_edge = (w["buy_pct"] - 50.0) * 0.55
    delta_edge = w["delta_ratio"] * 30.0
    slope_edge = w["slope"] * 4.0
    burst_edge = (min(w["burst_ratio"], 4.0) - 1.0) * 1.8
    streak_edge = w["streak_bias"] * 0.6
    return flow_edge + delta_edge + slope_edge + burst_edge + streak_edge


def classify_status(regime: str, confidence: float, agreement_label: str = None, volatility_label: str = None):
    if confidence < NO_TRADE_THRESHOLD:
        return "NO TRADE", "NO TRADE"

    if regime == "FLAT" and confidence < BAD_SIGNAL_THRESHOLD:
        return "NO TRADE", "NO TRADE"

    if regime == "FLAT":
        return "BAD SIGNAL", None

    if agreement_label == "MIXED" and confidence < 60:
        return "BAD SIGNAL", None

    if volatility_label == "EXPLOSIVE" and confidence < 60:
        return "BAD SIGNAL", None

    if confidence < BAD_SIGNAL_THRESHOLD:
        return "BAD SIGNAL", None

    return "GOOD SIGNAL", None


def _apply_higher_tf_filter(direction: str, score: float, reasons: list, features: dict):
    global_1h = str(features.get("global_trend_1h", "FLAT"))
    bias_15m = str(features.get("global_bias_15m", "FLAT"))

    htf_bonus = 0.0

    if direction == "UP":
        if global_1h == "UP":
            htf_bonus += 7.0
            reasons.append("1h trend aligned up")
        elif global_1h == "UP BIAS":
            htf_bonus += 4.0
            reasons.append("1h bias supports up")
        elif global_1h == "DOWN":
            htf_bonus -= 9.0
            reasons.append("1h trend against up")
        elif global_1h == "DOWN BIAS":
            htf_bonus -= 5.0
            reasons.append("1h bias against up")

        if bias_15m == "UP":
            htf_bonus += 4.0
            reasons.append("15m bias aligned up")
        elif bias_15m == "DOWN":
            htf_bonus -= 5.0
            reasons.append("15m bias against up")

    elif direction == "DOWN":
        if global_1h == "DOWN":
            htf_bonus += 7.0
            reasons.append("1h trend aligned down")
        elif global_1h == "DOWN BIAS":
            htf_bonus += 4.0
            reasons.append("1h bias supports down")
        elif global_1h == "UP":
            htf_bonus -= 9.0
            reasons.append("1h trend against down")
        elif global_1h == "UP BIAS":
            htf_bonus -= 5.0
            reasons.append("1h bias against down")

        if bias_15m == "DOWN":
            htf_bonus += 4.0
            reasons.append("15m bias aligned down")
        elif bias_15m == "UP":
            htf_bonus -= 5.0
            reasons.append("15m bias against down")

    return score + htf_bonus


def build_signal(features: dict, ai_pred: dict | None = None) -> dict:
    w30 = _safe_window(features, "w30")
    w10 = _safe_window(features, "w10")
    w3 = _safe_window(features, "w3")

    s30 = _edge_from_window(w30) * 0.30
    s10 = _edge_from_window(w10) * 0.35
    s3 = _edge_from_window(w3) * 0.20

    depth_edge = float(features.get("imbalance", 0.0)) * 10.0
    micro_edge = float(features.get("micro_bias_bps", 0.0)) * 0.9

    flow_momentum_raw = features.get("flow_momentum", 0.0)
    if isinstance(flow_momentum_raw, (int, float)):
        momentum_edge = float(flow_momentum_raw) * 0.45
        flow_momentum_val = float(flow_momentum_raw)
    else:
        momentum_edge = 0.0
        flow_momentum_val = 0.0

    rule_score = s30 + s10 + s3 + depth_edge + micro_edge + momentum_edge

    regime = features.get("regime_live", "FLAT")
    divergence = features.get("divergence")
    volatility_label = features.get("volatility_label", "NORMAL")
    liquidity_state = features.get("liquidity_state", "NORMAL")
    agreement_label = features.get("agreement_label", None)
    absorption = features.get("absorption", "-")
    spoof_risk = features.get("spoof_risk", "-")
    tape_pressure = features.get("tape_pressure", "-")
    market_speed = features.get("market_speed", "-")
    liquidity_trap = features.get("liquidity_trap", "-")
    fake_breakout_risk = features.get("fake_breakout_risk", "-")
    smart_money_hint = features.get("smart_money_hint", "-")
    global_trend_1h = features.get("global_trend_1h", "FLAT")
    global_bias_15m = features.get("global_bias_15m", "FLAT")

    penalties = 0.0

    if regime == "FLAT":
        penalties += 2.0

    if volatility_label == "FAST":
        penalties += 0.5
    elif volatility_label == "EXPLOSIVE":
        penalties += 1.2

    if liquidity_state == "THIN":
        penalties += 1.0

    if agreement_label == "MIXED":
        penalties += 1.4

    if spoof_risk in ("HIGH", "UNKNOWN"):
        penalties += 1.0

    if fake_breakout_risk == "HIGH":
        penalties += 1.2
    elif fake_breakout_risk == "MEDIUM":
        penalties += 0.6

    if divergence == "bullish":
        rule_score += 2.5
    elif divergence == "bearish":
        rule_score -= 2.5

    if absorption == "BULLISH ABSORPTION":
        rule_score += 1.3
    elif absorption == "BEARISH ABSORPTION":
        rule_score -= 1.3

    if smart_money_hint == "SMART BUYER":
        rule_score += 1.2
    elif smart_money_hint == "SMART SELLER":
        rule_score -= 1.2

    if liquidity_trap == "BULL TRAP":
        rule_score -= 1.0
    elif liquidity_trap == "BEAR TRAP":
        rule_score += 1.0

    rule_score_after = rule_score - penalties if rule_score >= 0 else rule_score + penalties
    rule_p_up, rule_p_down, rule_conf = sigmoid_conf(rule_score_after)

    ai_p_up = 50.0
    ai_p_down = 50.0
    ai_conf = 50.0
    ai_direction = "UP"

    if ai_pred:
        ai_p_up = float(ai_pred.get("p_up", 50.0))
        ai_p_down = float(ai_pred.get("p_down", 50.0))
        ai_conf = float(ai_pred.get("confidence", 50.0))
        ai_direction = ai_pred.get("direction", "UP")

    final_p_up = (rule_p_up * 0.70) + (ai_p_up * 0.30)
    final_p_down = 100.0 - final_p_up
    final_conf = max(final_p_up, final_p_down)
    direction = "UP" if final_p_up >= 50.0 else "DOWN"

    reasons = []

    # Higher timeframe filter
    direction_score = rule_score_after if direction == "UP" else -rule_score_after
    direction_score = _apply_higher_tf_filter(direction, direction_score, reasons, features)
    adjusted_score = direction_score if direction == "UP" else -direction_score

    adj_p_up, adj_p_down, adj_rule_conf = sigmoid_conf(adjusted_score)

    final_p_up = (adj_p_up * 0.70) + (ai_p_up * 0.30)
    final_p_down = 100.0 - final_p_up
    final_conf = max(final_p_up, final_p_down)
    direction = "UP" if final_p_up >= 50.0 else "DOWN"

    status, forced_display = classify_status(
        regime=regime,
        confidence=final_conf,
        agreement_label=agreement_label,
        volatility_label=volatility_label,
    )

    if forced_display:
        display_signal = forced_display
    else:
        display_signal = direction

    if w10["buy_pct"] > 56:
        reasons.append(f"10s flow bullish ({w10['buy_pct']}% buy)")
    elif w10["sell_pct"] > 56:
        reasons.append(f"10s flow bearish ({w10['sell_pct']}% sell)")

    if abs(flow_momentum_val) > 5:
        reasons.append(f"flow momentum {flow_momentum_val:+.1f}")

    if w3["burst_ratio"] > 1.8:
        reasons.append(f"trade burst x{w3['burst_ratio']:.2f}")

    if abs(float(features.get("imbalance", 0.0))) > 0.12:
        reasons.append(f"depth imbalance {float(features.get('imbalance', 0.0)):+.3f}")

    if abs(float(features.get("micro_bias_bps", 0.0))) > 0.8:
        reasons.append(f"microprice drift {float(features.get('micro_bias_bps', 0.0)):+.2f} bps")

    if divergence:
        reasons.append(f"{divergence} divergence")

    if absorption and absorption != "-":
        reasons.append(str(absorption).lower())

    if smart_money_hint and smart_money_hint != "-":
        reasons.append(f"smart money: {smart_money_hint.lower()}")

    if liquidity_trap and liquidity_trap != "-":
        reasons.append(f"trap: {str(liquidity_trap).lower()}")

    if fake_breakout_risk and fake_breakout_risk != "-":
        reasons.append(f"fake breakout: {str(fake_breakout_risk).lower()}")

    reasons.append(f"market regime: {str(regime).lower()}")
    reasons.append(f"volatility: {str(volatility_label).lower()}")
    reasons.append(f"liquidity: {str(liquidity_state).lower()}")
    reasons.append(f"1h trend: {str(global_trend_1h).lower()}")
    reasons.append(f"15m bias: {str(global_bias_15m).lower()}")
    reasons.append(f"ai: {ai_direction} {ai_conf:.1f}%")

    return {
        "display_signal": display_signal,
        "direction": direction,
        "status": status,
        "regime": regime,
        "confidence": round(final_conf, 1),
        "score": round(adjusted_score, 2),
        "p_up": round(final_p_up, 1),
        "p_down": round(final_p_down, 1),
        "rule_conf": round(adj_rule_conf, 1),
        "ai_conf": round(ai_conf, 1),
        "ai_direction": ai_direction,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_open_utc": next_5m_open().isoformat(),
        "reasons": reasons[:14],
    }
