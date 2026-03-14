import json
import math
from pathlib import Path

from config import AI_STATE_FILE


FEATURE_NAMES = [
    "flow30",
    "flow10",
    "flow3",
    "delta10",
    "delta3",
    "slope10",
    "imbalance",
    "micro_bias",
    "agreement",
]


DEFAULT_STATE = {
    "bias": 0.0,
    "lr": 0.04,
    "samples": 0,
    "weights": {
        "flow30": 0.45,
        "flow10": 0.95,
        "flow3": 0.75,
        "delta10": 0.80,
        "delta3": 0.55,
        "slope10": 0.40,
        "imbalance": 0.30,
        "micro_bias": 0.35,
        "agreement": 0.55,
    }
}


def _sigmoid(x: float) -> float:
    x = max(-30.0, min(30.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _clip(x: float, lo: float = -3.0, hi: float = 3.0) -> float:
    return max(lo, min(hi, x))


def save_ai_state(state: dict):
    Path(AI_STATE_FILE).write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def load_ai_state() -> dict:
    p = Path(AI_STATE_FILE)

    if not p.exists():
        save_ai_state(DEFAULT_STATE)
        return json.loads(json.dumps(DEFAULT_STATE))

    try:
        data = json.loads(p.read_text(encoding="utf-8"))

        if "weights" not in data:
            raise ValueError("weights missing")

        for name in FEATURE_NAMES:
            data["weights"].setdefault(name, DEFAULT_STATE["weights"][name])

        data.setdefault("bias", 0.0)
        data.setdefault("lr", 0.04)
        data.setdefault("samples", 0)

        return data
    except Exception:
        save_ai_state(DEFAULT_STATE)
        return json.loads(json.dumps(DEFAULT_STATE))


def build_ai_vector(features: dict) -> dict:
    w30 = features.get("w30", {})
    w10 = features.get("w10", {})
    w3 = features.get("w3", {})

    flow30 = _clip((float(w30.get("buy_pct", 50.0)) - 50.0) / 10.0)
    flow10 = _clip((float(w10.get("buy_pct", 50.0)) - 50.0) / 10.0)
    flow3 = _clip((float(w3.get("buy_pct", 50.0)) - 50.0) / 10.0)

    delta10 = _clip(float(w10.get("delta_ratio", 0.0)) * 8.0)
    delta3 = _clip(float(w3.get("delta_ratio", 0.0)) * 8.0)

    slope10 = _clip(float(w10.get("slope", 0.0)) / 4.0)
    imbalance = _clip(float(features.get("imbalance", 0.0)) * 4.0)
    micro_bias = _clip(float(features.get("micro_bias_bps", 0.0)) / 4.0)

    agreement_votes = 0
    agreement_votes += 1 if float(w30.get("buy_pct", 50.0)) >= 50.0 else -1
    agreement_votes += 1 if float(w10.get("buy_pct", 50.0)) >= 50.0 else -1
    agreement_votes += 1 if float(w3.get("buy_pct", 50.0)) >= 50.0 else -1
    agreement = _clip(agreement_votes / 2.0, -1.5, 1.5)

    return {
        "flow30": flow30,
        "flow10": flow10,
        "flow3": flow3,
        "delta10": delta10,
        "delta3": delta3,
        "slope10": slope10,
        "imbalance": imbalance,
        "micro_bias": micro_bias,
        "agreement": agreement,
    }


def predict_ai(features: dict) -> dict:
    state = load_ai_state()
    vec = build_ai_vector(features)

    z = float(state["bias"])
    for name in FEATURE_NAMES:
        z += float(state["weights"].get(name, 0.0)) * float(vec.get(name, 0.0))

    p_up = _sigmoid(z)
    p_down = 1.0 - p_up
    conf = max(p_up, p_down) * 100.0
    direction = "UP" if p_up >= 0.5 else "DOWN"

    return {
        "direction": direction,
        "p_up": round(p_up * 100.0, 1),
        "p_down": round(p_down * 100.0, 1),
        "confidence": round(conf, 1),
        "raw_score": round(z, 4),
        "vector": vec,
        "samples": int(state.get("samples", 0)),
    }


def train_on_resolved_history(history: list):
    state = load_ai_state()
    changed = False

    for row in history:
        if row.get("ai_trained"):
            continue

        if row.get("result") not in ("WIN", "LOSS"):
            continue

        actual = row.get("actual")
        if actual not in ("UP", "DOWN"):
            continue

        vec = row.get("feature_snapshot")
        if not isinstance(vec, dict) or not vec:
            row["ai_trained"] = True
            changed = True
            continue

        target = 1.0 if actual == "UP" else 0.0

        z = float(state["bias"])
        for name in FEATURE_NAMES:
            z += float(state["weights"].get(name, 0.0)) * float(vec.get(name, 0.0))

        pred = _sigmoid(z)
        err = target - pred
        lr = float(state.get("lr", 0.04))

        state["bias"] += lr * err
        for name in FEATURE_NAMES:
            state["weights"][name] += lr * err * float(vec.get(name, 0.0))

        state["samples"] = int(state.get("samples", 0)) + 1
        row["ai_trained"] = True
        changed = True

    if changed:
        save_ai_state(state)

    return changed, state
