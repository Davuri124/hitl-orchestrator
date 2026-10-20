"""Real data only: HotpotQA (distractor setting) has gold answers AND gold supporting paragraphs,
which gives ground-truth correctness for both the retrieval step and the final answer."""
import json, random
from dataclasses import dataclass


@dataclass
class Example:
    qid: str
    question: str
    answer: str
    paragraphs: list      # [(title, text)] -- 10 per question (2 gold + 8 distractors)
    gold_titles: list
    qtype: str
    level: str


def _from_official(it):
    paras = [(t, " ".join(s).strip()) for t, s in it["context"]]
    gold = sorted({t for t, _ in it["supporting_facts"]})
    return Example(it["_id"], it["question"], it["answer"], paras, gold, it.get("type", ""), it.get("level", ""))


def _from_hf(r):
    c = r["context"]
    paras = [(t, " ".join(s).strip()) for t, s in zip(c["title"], c["sentences"])]
    gold = sorted(set(r["supporting_facts"]["title"]))
    return Example(r["id"], r["question"], r["answer"], paras, gold, r.get("type", ""), r.get("level", ""))


def load_hotpot(split, path=None):
    """path: official json (hotpot_train_v1.1.json / hotpot_dev_distractor_v1.json) or None -> HF hub."""
    if path:
        with open(path, encoding="utf-8") as f:
            return [_from_official(x) for x in json.load(f)]
    from datasets import load_dataset
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    return [_from_hf(r) for r in ds]


def all_examples(cfg, train_path=None, dev_path=None):
    rng = random.Random(cfg.seed)
    tr, dv = load_hotpot("train", train_path), load_hotpot("validation", dev_path)
    rng.shuffle(tr); rng.shuffle(dv)
    tr, te = tr[: cfg.n_train + cfg.n_val], dv[cfg.dev_offset: cfg.dev_offset + cfg.n_test]
    ex = tr + te
    idx = {"train": list(range(cfg.n_train)),
           "val": list(range(cfg.n_train, cfg.n_train + cfg.n_val)),
           "test": list(range(cfg.n_train + cfg.n_val, len(ex)))}
    return ex, idx
