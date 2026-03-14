from collections import deque


def _safe_trades(state):
    try:
        return list(getattr(state, "trades", []) or [])
    except Exception:
        return []


def _safe_depth(state):
    try:
        d = getattr(state, "depth", {}) or {}
        bids = d.get("bids", []) or []
        asks = d.get("asks", []) or []
        return bids, asks
    except Exception:
        return [], []


def _last_n(seq, n):
    if not seq:
        return []
    return seq[-n:]


def _trend_label(x, eps=1e-9):
    if x > eps:
        return "growing"
    if x < -eps:
        return "falling"
    return "flat"


def _window_stats(trades, window_sec):
    if not trades:
        return {
            "buy_qty": 0.0,
            "sell_qty": 0.0,
            "buy_pct": 50.0,
            "sell_pct": 50.0,
            "cum_delta": 0.0,
            "delta_ratio": 0.0,
            "count": 0,
            "intensity": 0.0,
            "slope": 0.0,
            "range": 0.0,
            "burst_ratio": 1.0,
            "streak_bias": 0.0,
            "prices": [],
            "sides": [],
        }

    now_ts = trades[-1]["ts"]
    w = [t for t in trades if now_ts - t["ts"] <= window_sec]

    if not w:
        return {
            "buy_qty": 0.0,
            "sell_qty": 0.0,
            "buy_pct": 50.0,
            "sell_pct": 50.0,
            "cum_delta": 0.0,
            "delta_ratio": 0.0,
            "count": 0,
            "intensity": 0.0,
            "slope": 0.0,
            "range": 0.0,
            "burst_ratio": 1.0,
            "streak_bias": 0.0,
            "prices": [],
            "sides": [],
        }

    buy_qty = sum(float(t["qty"]) for t in w if not t["is_sell"])
    sell_qty = sum(float(t["qty"]) for t in w if t["is_sell"])
    total = buy_qty + sell_qty

    buy_pct = (buy_qty / total * 100.0) if total > 0 else 50.0
    sell_pct = 100.0 - buy_pct if total > 0 else 50.0
    cum_delta = buy_qty - sell_qty
    delta_ratio = (cum_delta / total) if total > 0 else 0.0

    prices = [float(t["price"]) for t in w]
    slope = prices[-1] - prices[0] if len(prices) >= 2 else 0.0
    prange = (max(prices) - min(prices)) if prices else 0.0

    count = len(w)
    intensity = count / max(window_sec, 1)

    last3 = [t for t in trades if now_ts - t["ts"] <= 3]
    burst_ratio = ((len(last3) / 3.0) / max(intensity, 0.0001)) if intensity > 0 else 1.0

    streak = 0
    last_side = None
    sides = []

    for t in w:
        sides.append("SELL" if t["is_sell"] else "BUY")

    for t in reversed(w):
        side = "SELL" if t["is_sell"] else "BUY"
        if last_side is None:
            last_side = side
            streak = 1
        elif side == last_side:
            streak += 1
        else:
            break

    streak_bias = 0.0
    if streak >= 2:
        streak_bias = float(streak if last_side == "BUY" else -streak)

    return {
        "buy_qty": round(buy_qty, 6),
        "sell_qty": round(sell_qty, 6),
        "buy_pct": round(buy_pct, 2),
        "sell_pct": round(sell_pct, 2),
        "cum_delta": round(cum_delta, 6),
        "delta_ratio": round(delta_ratio, 6),
        "count": count,
        "intensity": round(intensity, 4),
        "slope": round(slope, 4),
        "range": round(prange, 4),
        "burst_ratio": round(burst_ratio, 4),
        "streak_bias": round(streak_bias, 2),
        "prices": prices,
        "sides": sides,
    }


def _calc_aggressive_side(w3, w10):
    short_buy = w3["buy_pct"]
    med_buy = w10["buy_pct"]

    aggr_buy = max(0.0, min(100.0, short_buy * 0.7 + med_buy * 0.3))
    aggr_sell = 100.0 - aggr_buy
    return round(aggr_buy, 1), round(aggr_sell, 1)


def _calc_absorption(w10, slope):
    if w10["sell_pct"] > 60 and slope >= 0:
        return "BULLISH ABSORPTION"
    if w10["buy_pct"] > 60 and slope <= 0:
        return "BEARISH ABSORPTION"
    return "NONE"


def _calc_agreement_label(w60, w30, w10, w3):
    votes = []
    for w in (w60, w30, w10, w3):
        votes.append("BUY" if w["buy_pct"] >= 50 else "SELL")

    buy_votes = votes.count("BUY")
    sell_votes = votes.count("SELL")

    if buy_votes == 4:
        return "FULL BUY ALIGNMENT"
    if sell_votes == 4:
        return "FULL SELL ALIGNMENT"
    if buy_votes >= 3:
        return "BUY BIAS"
    if sell_votes >= 3:
        return "SELL BIAS"
    return "MIXED"


def _calc_market_speed(w3, w10, w30):
    intensity = max(w3["intensity"], w10["intensity"], w30["intensity"])
    if intensity >= 8:
        return "EXPLOSIVE"
    if intensity >= 4:
        return "FAST"
    if intensity >= 1.5:
        return "NORMAL"
    return "QUIET"


def _calc_volatility(w10):
    r = w10["range"]
    if r >= 20:
        return "EXPLOSIVE"
    if r >= 8:
        return "FAST"
    if r >= 2:
        return "NORMAL"
    return "QUIET"


def _calc_liquidity_state(bid_vol, ask_vol, levels):
    total = bid_vol + ask_vol
    if levels <= 0 or total < 5:
        return "THIN"
    if total < 20:
        return "NORMAL"
    return "THICK"


def _calc_spoof_risk(bids, asks):
    if not bids or not asks:
        return "UNKNOWN"

    bid_sizes = [float(q) for _, q in bids[:5]]
    ask_sizes = [float(q) for _, q in asks[:5]]

    if not bid_sizes or not ask_sizes:
        return "UNKNOWN"

    max_bid = max(bid_sizes)
    max_ask = max(ask_sizes)
    avg_bid = sum(bid_sizes) / len(bid_sizes)
    avg_ask = sum(ask_sizes) / len(ask_sizes)

    if max_bid > avg_bid * 4 or max_ask > avg_ask * 4:
        return "HIGH"
    if max_bid > avg_bid * 2.5 or max_ask > avg_ask * 2.5:
        return "MEDIUM"
    return "LOW"


def _calc_tape_pressure(w3, w10):
    if w3["buy_pct"] > 62 and w10["buy_pct"] > 55:
        return "BUY PRESSURE"
    if w3["sell_pct"] > 62 and w10["sell_pct"] > 55:
        return "SELL PRESSURE"
    return "NEUTRAL"


def _calc_liquidity_trap(w3, w10, slope):
    if w3["buy_pct"] > 70 and slope <= 0:
        return "BULL TRAP"
    if w3["sell_pct"] > 70 and slope >= 0:
        return "BEAR TRAP"
    return "NONE"


def _calc_fake_breakout_risk(w3, w10, divergence, spoof_risk):
    risk = 0
    if divergence in ("bullish", "bearish"):
        risk += 1
    if spoof_risk == "HIGH":
        risk += 2
    elif spoof_risk == "MEDIUM":
        risk += 1
    if w3["burst_ratio"] > 2.2 and abs(w10["slope"]) < 0.5:
        risk += 1

    if risk >= 3:
        return "HIGH"
    if risk >= 2:
        return "MEDIUM"
    return "LOW"


def _calc_smart_money_hint(absorption, tape_pressure, divergence):
    if absorption == "BULLISH ABSORPTION" and divergence == "bullish":
        return "SMART BUYER"
    if absorption == "BEARISH ABSORPTION" and divergence == "bearish":
        return "SMART SELLER"
    if tape_pressure == "BUY PRESSURE":
        return "BUYER ACTIVE"
    if tape_pressure == "SELL PRESSURE":
        return "SELLER ACTIVE"
    return "NONE"


def _calc_liquidity_sweep(w3, w10, depth_imbalance, micro_bias_bps, divergence, trap):
    score = 0
    direction = "NONE"
    label = "NONE"

    sell_shock = w3["sell_pct"] >= 72 and w3["burst_ratio"] >= 1.6
    buy_shock = w3["buy_pct"] >= 72 and w3["burst_ratio"] >= 1.6

    if sell_shock:
        score += 1
        if depth_imbalance > 0.18:
            score += 2
        if micro_bias_bps > 0.2:
            score += 1
        if divergence == "bullish":
            score += 1
        if trap == "BEAR TRAP":
            score += 2

        if score >= 4:
            direction = "UP"
            label = "BULLISH SWEEP"

    if buy_shock:
        score = 0
        score += 1
        if depth_imbalance < -0.18:
            score += 2
        if micro_bias_bps < -0.2:
            score += 1
        if divergence == "bearish":
            score += 1
        if trap == "BULL TRAP":
            score += 2

        if score >= 4:
            direction = "DOWN"
            label = "BEARISH SWEEP"

    return {
        "liquidity_sweep": label,
        "liquidity_sweep_dir": direction,
    }


def _calc_divergence(w10):
    if w10["slope"] > 0 and w10["delta_ratio"] < 0:
        return "bearish"
    if w10["slope"] < 0 and w10["delta_ratio"] > 0:
        return "bullish"
    return None


def _calc_regime(w10, w30):
    abs_flow = abs(w10["buy_pct"] - 50.0)
    abs_delta = abs(w10["delta_ratio"])
    abs_slope = abs(w10["slope"])

    if abs_slope < 0.4 and abs_delta < 0.08 and abs_flow < 4:
        return "FLAT"

    combo_slope = w10["slope"] + w30["slope"] * 0.35
    if combo_slope > 0:
        return "TREND UP"
    return "TREND DOWN"


def _calc_depth_features(bids, asks):
    bid_vol = sum(float(q) for _, q in bids)
    ask_vol = sum(float(q) for _, q in asks)

    depth_total = bid_vol + ask_vol
    imbalance = ((bid_vol - ask_vol) / depth_total) if depth_total > 0 else 0.0
    imbalance_pct = ((imbalance + 1.0) / 2.0 * 100.0) if depth_total > 0 else 50.0
    bid_ask_ratio = (bid_vol / ask_vol) if ask_vol > 0 else 1.0

    best_bid = float(bids[0][0]) if bids else None
    best_ask = float(asks[0][0]) if asks else None

    mid = None
    microprice = None
    micro_bias_bps = 0.0

    if best_bid is not None and best_ask is not None:
        mid = (best_bid + best_ask) / 2.0
        if depth_total > 0:
            microprice = ((best_ask * bid_vol) + (best_bid * ask_vol)) / depth_total
            if mid:
                micro_bias_bps = ((microprice - mid) / mid) * 10000.0

    bid_wall_active = False
    ask_wall_active = False
    bid_wall_price = None
    ask_wall_price = None

    if bids:
        bid_sizes = [float(q) for _, q in bids[:10]]
        if bid_sizes:
            avg_bid = sum(bid_sizes) / len(bid_sizes)
            top_bid = max(bids[:10], key=lambda x: float(x[1]))
            if float(top_bid[1]) > avg_bid * 2.5:
                bid_wall_active = True
                bid_wall_price = float(top_bid[0])

    if asks:
        ask_sizes = [float(q) for _, q in asks[:10]]
        if ask_sizes:
            avg_ask = sum(ask_sizes) / len(ask_sizes)
            top_ask = max(asks[:10], key=lambda x: float(x[1]))
            if float(top_ask[1]) > avg_ask * 2.5:
                ask_wall_active = True
                ask_wall_price = float(top_ask[0])

    return {
        "bid_vol": round(bid_vol, 4),
        "ask_vol": round(ask_vol, 4),
        "imbalance": round(imbalance, 4),
        "imbalance_pct": round(imbalance_pct, 2),
        "bid_ask_ratio": round(bid_ask_ratio, 4),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid": mid,
        "microprice": microprice,
        "micro_bias_bps": round(micro_bias_bps, 4),
        "levels": len(bids),
        "bid_wall_active": bid_wall_active,
        "ask_wall_active": ask_wall_active,
        "bid_wall_price": bid_wall_price,
        "ask_wall_price": ask_wall_price,
    }


def _calc_global_trend(trades):
    if not trades:
        return "FLAT"

    now_ts = trades[-1]["ts"]

    w15 = [t for t in trades if now_ts - t["ts"] <= 15 * 60]
    w60 = [t for t in trades if now_ts - t["ts"] <= 60 * 60]

    if len(w60) < 5:
        return "FLAT"

    p15_first = float(w15[0]["price"]) if w15 else float(trades[-1]["price"])
    p15_last = float(w15[-1]["price"]) if w15 else float(trades[-1]["price"])
    p60_first = float(w60[0]["price"])
    p60_last = float(w60[-1]["price"])

    slope15 = p15_last - p15_first
    slope60 = p60_last - p60_first

    if abs(slope60) < 3 and abs(slope15) < 1.2:
        return "FLAT"

    if slope60 > 0 and slope15 >= 0:
        return "UP"
    if slope60 < 0 and slope15 <= 0:
        return "DOWN"

    if slope60 > 0:
        return "UP BIAS"
    if slope60 < 0:
        return "DOWN BIAS"

    return "FLAT"


def _calc_global_bias_15m(trades):
    if not trades:
        return "FLAT"

    now_ts = trades[-1]["ts"]
    w15 = [t for t in trades if now_ts - t["ts"] <= 15 * 60]

    if len(w15) < 5:
        return "FLAT"

    first_p = float(w15[0]["price"])
    last_p = float(w15[-1]["price"])
    slope = last_p - first_p

    if abs(slope) < 1.0:
        return "FLAT"
    return "UP" if slope > 0 else "DOWN"


def build_features(state):
    trades = _safe_trades(state)
    bids, asks = _safe_depth(state)

    last_price = None
    if trades:
        try:
            last_price = float(trades[-1]["price"])
        except Exception:
            last_price = None

    w60 = _window_stats(trades, 60)
    w30 = _window_stats(trades, 30)
    w10 = _window_stats(trades, 10)
    w3 = _window_stats(trades, 3)

    depth = _calc_depth_features(bids, asks)

    divergence = _calc_divergence(w10)
    regime_live = _calc_regime(w10, w30)
    volatility_label = _calc_volatility(w10)
    market_speed = _calc_market_speed(w3, w10, w30)
    liquidity_state = _calc_liquidity_state(depth["bid_vol"], depth["ask_vol"], depth["levels"])
    agreement_label = _calc_agreement_label(w60, w30, w10, w3)
    absorption = _calc_absorption(w10, w10["slope"])
    spoof_risk = _calc_spoof_risk(bids, asks)
    tape_pressure = _calc_tape_pressure(w3, w10)
    liquidity_trap = _calc_liquidity_trap(w3, w10, w10["slope"])
    fake_breakout_risk = _calc_fake_breakout_risk(w3, w10, divergence, spoof_risk)
    smart_money_hint = _calc_smart_money_hint(absorption, tape_pressure, divergence)

    sweep_info = _calc_liquidity_sweep(
        w3=w3,
        w10=w10,
        depth_imbalance=depth["imbalance"],
        micro_bias_bps=depth["micro_bias_bps"],
        divergence=divergence,
        trap=liquidity_trap,
    )

    flow_momentum = round(w3["buy_pct"] - w10["buy_pct"], 2)
    aggr_buy_pct, aggr_sell_pct = _calc_aggressive_side(w3, w10)

    delta_hist_7 = deque(maxlen=7)
    delta_hist_20 = deque(maxlen=20)

    recent_trades = _last_n(trades, 20)
    for idx, t in enumerate(recent_trades):
        side = -float(t["qty"]) if t["is_sell"] else float(t["qty"])
        delta_hist_20.append(side)
        if idx >= max(0, len(recent_trades) - 7):
            delta_hist_7.append(side)

    delta_trend_7 = _trend_label(sum(delta_hist_7))
    delta_trend_20 = _trend_label(sum(delta_hist_20))

    global_trend_1h = _calc_global_trend(trades)
    global_bias_15m = _calc_global_bias_15m(trades)

    return {
        "last_price": last_price,

        "buy_qty": w30["buy_qty"],
        "sell_qty": w30["sell_qty"],
        "buy_pct": w30["buy_pct"],
        "sell_pct": w30["sell_pct"],
        "cum_delta": w30["cum_delta"],
        "trades_count": w30["count"],
        "slope": w30["slope"],

        "w60": w60,
        "w30": w30,
        "w10": w10,
        "w3": w3,

        "flow_momentum": flow_momentum,
        "aggr_buy_pct": aggr_buy_pct,
        "aggr_sell_pct": aggr_sell_pct,

        "divergence": divergence,
        "regime_live": regime_live,
        "volatility_label": volatility_label,
        "market_speed": market_speed,
        "liquidity_state": liquidity_state,
        "agreement_label": agreement_label,
        "absorption": absorption,
        "spoof_risk": spoof_risk,
        "tape_pressure": tape_pressure,
        "liquidity_trap": liquidity_trap,
        "fake_breakout_risk": fake_breakout_risk,
        "smart_money_hint": smart_money_hint,

        "liquidity_sweep": sweep_info["liquidity_sweep"],
        "liquidity_sweep_dir": sweep_info["liquidity_sweep_dir"],

        "delta_trend_7": delta_trend_7,
        "delta_trend_20": delta_trend_20,

        "global_trend_1h": global_trend_1h,
        "global_bias_15m": global_bias_15m,

        **depth,
    }
