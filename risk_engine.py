def _to_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def evaluate_risk(signal: dict, features: dict, regime: dict) -> dict:
    confidence = _to_float(signal.get("confidence"), 50.0)
    rule_conf = _to_float(signal.get("rule_conf"), confidence)
    ai_conf = _to_float(signal.get("ai_conf"), 50.0)

    status = str(signal.get("status", "NO TRADE"))
    direction = str(signal.get("direction", "UP"))
    regime_name = str(regime.get("regime_type", "FLAT"))
    regime_conf = _to_float(regime.get("regime_confidence"), 0.0)

    volatility = str(features.get("volatility_label", "NORMAL"))
    liquidity = str(features.get("liquidity_state", "NORMAL"))
    spoof_risk = str(features.get("spoof_risk", "UNKNOWN"))
    fake_breakout = str(features.get("fake_breakout_risk", "LOW"))
    liquidity_trap = str(features.get("liquidity_trap", "NONE"))
    smart_money = str(features.get("smart_money_hint", "NONE"))
    agreement = str(features.get("agreement_label", "MIXED"))
    tape_pressure = str(features.get("tape_pressure", "NEUTRAL"))

    risk_score = 0.0
    reasons = []

    # 1. Базовая оценка по уверенности (confidence)
    if confidence >= 85:
        risk_score += 3.0
    elif confidence >= 72:
        risk_score += 2.0
    elif confidence >= 60:
        risk_score += 1.0
    else:
        risk_score -= 2.0
        reasons.append("confidence too low")

    # 2. Согласованность Правил и AI
    if abs(rule_conf - ai_conf) <= 8:
        risk_score += 1.0
    elif abs(rule_conf - ai_conf) > 20:
        risk_score -= 1.0
        reasons.append("rule/ai disagreement")

    # 3. Статус сигнала
    if status == "GOOD SIGNAL":
        risk_score += 2.0
    elif status == "BAD SIGNAL":
        risk_score -= 1.5
        reasons.append("bad signal status")
    elif status == "NO TRADE":
        risk_score -= 4.0
        reasons.append("no trade status")

    # 4. Фильтрация по Рыночному Режиму (Regime)
    if regime_name == "TREND":
        risk_score += 2.0
    elif regime_name == "INSTITUTIONAL_MOVE":
        risk_score += 2.5
    elif regime_name == "VOL_EXPANSION":
        risk_score += 1.0
    elif regime_name == "SQUEEZE":
        risk_score -= 1.0
        reasons.append("squeeze regime")
    elif regime_name == "FLAT":
        risk_score -= 2.5
        reasons.append("flat regime")

    if regime_conf >= 4:
        risk_score += 1.0

    # 5. Состояние рынка (Market State)
    if volatility == "QUIET":
        risk_score -= 0.5
    elif volatility == "FAST":
        risk_score += 0.5
    elif volatility == "EXPLOSIVE":
        risk_score -= 0.8
        reasons.append("explosive volatility")

    if liquidity == "THIN":
        risk_score -= 1.2
        reasons.append("thin liquidity")
    elif liquidity == "THICK":
        risk_score += 0.6

    if spoof_risk == "HIGH":
        risk_score -= 1.5
        reasons.append("high spoof risk")
    elif spoof_risk == "MEDIUM":
        risk_score -= 0.6

    if fake_breakout == "HIGH":
        risk_score -= 1.4
        reasons.append("high fake breakout risk")
    elif fake_breakout == "MEDIUM":
        risk_score -= 0.7

    if liquidity_trap in ("BULL TRAP", "BEAR TRAP"):
        risk_score -= 0.8
        reasons.append(f"trap detected: {liquidity_trap.lower()}")

    # 6. Согласованность таймфреймов (Agreement)
    if agreement in ("FULL BUY ALIGNMENT", "FULL SELL ALIGNMENT"):
        risk_score += 1.5
    elif agreement in ("BUY BIAS", "SELL BIAS"):
        risk_score += 0.8
    elif agreement == "MIXED":
        risk_score -= 0.8
        reasons.append("mixed alignment")

    # 7. Давление ленты и Smart Money
    if tape_pressure in ("BUY PRESSURE", "SELL PRESSURE"):
        risk_score += 0.8

    if smart_money in ("SMART BUYER", "SMART SELLER"):
        risk_score += 1.4
    elif smart_money in ("BUYER ACTIVE", "SELLER ACTIVE"):
        risk_score += 0.7

    # 8. Проверка направления (Direction Sanity Check)
    if direction == "UP" and smart_money == "SMART SELLER":
        risk_score -= 1.2
        reasons.append("direction vs smart money conflict")
    if direction == "DOWN" and smart_money == "SMART BUYER":
        risk_score -= 1.2
        reasons.append("direction vs smart money conflict")

    # 9. Финальное решение
    if risk_score >= 6.0:
        decision = "TRADE"
        size_mode = "FULL"
        risk_label = "A+"
    elif risk_score >= 3.5:
        decision = "TRADE"
        size_mode = "REDUCED"
        risk_label = "B"
    elif risk_score >= 2.0:
        decision = "WATCH"
        size_mode = "MICRO"
        risk_label = "C"
    else:
        decision = "NO TRADE"
        size_mode = "NONE"
        risk_label = "D"

    return {
        "decision": decision,
        "size_mode": size_mode,
        "risk_label": risk_label,
        "risk_score": round(risk_score, 2),
        "risk_reasons": reasons[:8],
    }
