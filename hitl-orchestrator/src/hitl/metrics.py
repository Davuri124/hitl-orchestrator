import re, string
from collections import Counter


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred), normalize_answer(gold)
    if p in ("yes", "no", "noanswer") or g in ("yes", "no", "noanswer"):
        return float(p == g)
    pt, gt = p.split(), g.split()
    common = Counter(pt) & Counter(gt)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    prec, rec = ns / len(pt), ns / len(gt)
    return 2 * prec * rec / (prec + rec)


def em_score(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))
