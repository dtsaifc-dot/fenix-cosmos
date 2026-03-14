def detect_regime(features: dict) -> dict:
    w60 = features.get("w60", {})
    w30 = features.get("w30", {})
    w10 = features.get("w10", {})
    w3 = features.get("w3", {})

    buy30 = float(w30.get("buy_pct", 50.0))
    buy10 = float(w10.get("buy_pct", 50.0))
    buy3 = float(w3.get("buy_pct", 50.0))

    slope30 = float(w30.get("slope", 0.0))
    slope10 = float(w10.get("slope", 0.0))
    slope3 = float(w3.get("slope", 0.0))

    range30 = float(w30.get("range", 0.0))
    range10 = float(w10.get("range", 0.0))
    burst3 = float(w3.get("burst_ratio", 1.0))

    imbalance = float(features.get("imbalance", 0.0))
    micro_bias = float(features.get("micro_bias_bps", 0.0))
    spoof_risk = str(features.get("spoof_risk", "UNKNOWN"))
    liquidity_state = str(features.get("liquidity_state", "NORMAL"))
    agreement = str(features.get("agreement_label", "MIXED"))
    tape_pressure = str(features.get("tape_pressure", "NEUTRAL"))
    smart_money = str(features.get("smart_money_hint", "NONE"))

    abs_flow10 = abs(buy10 - 50.0)
    abs_flow3 = abs(buy3 - 50.0)
    abs_slope10 = abs(slope10)
    abs_imb = abs(imbalance)
    abs_micro = abs(micro_bias)

    score_trend = 0.0
    score_flat = 0.0
    score_squeeze = 0.0
    score_vol_exp = 0.0
    score_inst = 0.0

    # TREND (Оценка силы направленного движения)
    if agreement in ("FULL BUY ALIGNMENT", "FULL SELL ALIGNMENT", "BUY BIAS", "SELL BIAS"):
        score_trend += 2.0
    if abs_flow10 > 7:
        score_trend += 1.5
    if abs_slope10 > 0.8:
        score_trend += 1.5
    if slope30 * slope10 > 0:
        score_trend += 1.0
    if tape_pressure in ("BUY PRESSURE", "SELL PRESSURE"):
        score_trend += 1.0

    # FLAT (Оценка затишья и отсутствия волатильности)
    if abs_flow10 < 4:
        score_flat += 2.0
    if abs_slope10 < 0.25:
        score_flat += 2.0
    if range10 < 2:
        score_flat += 1.0
    if agreement == "MIXED":
        score_flat += 1.0

    # SQUEEZE (Сжатие пружины перед импульсом)
    if range30 < 4 and range10 < 1.5:
        score_squeeze += 2.0
    if abs_flow10 < 5 and abs_slope10 < 0.4:
        score_squeeze += 1.5
    if liquidity_state in ("THIN", "NORMAL"):
        score_squeeze += 0.5

    # VOL EXPANSION (Взрывная волатильность)
    if range10 > 6:
        score_vol_exp += 1.5
    if burst3 > 1.8:
        score_vol_exp += 2.0
    if abs_flow3 > 10:
        score_vol_exp += 1.5
    if abs_micro > 0.8:
        score_vol_exp += 0.8

    # INSTITUTIONAL MOVE (Признаки работы крупного капитала)
    if abs_imb > 0.25:
        score_inst += 2.0
    if smart_money in ("SMART BUYER", "SMART SELLER", "BUYER ACTIVE", "SELLER ACTIVE"):
        score_inst += 2.0
    if burst3 > 2.0:
        score_inst += 1.2
    if tape_pressure in ("BUY PRESSURE", "SELL PRESSURE"):
        score_inst += 1.0
    if spoof_risk == "LOW":
        score_inst += 0.6

    scores = {
        "TREND": round(score_trend, 2),
        "FLAT": round(score_flat, 2),
        "SQUEEZE": round(score_squeeze, 2),
        "VOL_EXPANSION": round(score_vol_exp, 2),
        "INSTITUTIONAL_MOVE": round(score_inst, 2),
    }

    regime = max(scores, key=scores.get)
    confidence = scores[regime]

    direction = "NEUTRAL"
    if regime in ("TREND", "VOL_EXPANSION", "INSTITUTIONAL_MOVE"):
        if buy10 >= 50:
            direction = "UP"
        else:
            direction = "DOWN"

    return {
        "regime_type": regime,
        "regime_confidence": confidence,
        "regime_direction": direction,
        "regime_scores": scores,
    }
