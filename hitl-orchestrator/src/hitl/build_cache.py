"""GPU stage (Kaggle). Runs the REAL pipeline once for every query and every branch the policy could take,
and caches outputs + uncertainty signals. RL then trains on this replay table (no synthetic data anywhere).

Cells per query: 3 evidence variants (narrow / wide / gold) x 2 answerers (base / specialist) = 6.

  python -m hitl.build_cache retrieve
  python -m hitl.build_cache summarise
  python -m hitl.build_cache answer --which base
  python -m hitl.build_cache answer --which spec
  python -m hitl.build_cache critic
  python -m hitl.build_cache verify
  python -m hitl.build_cache assemble
Each LLM stage is resumable (jsonl append) and loads exactly one model.
"""
import argparse, json
import numpy as np
from .config import Cfg
from .data import all_examples
from collections import Counter
from .metrics import f1_score, em_score, normalize_answer
from . import agents


class Store:
    def __init__(self, path):
        self.path, self.rows = path, {}
        if path.exists():
            for line in open(path, encoding="utf-8"):
                r = json.loads(line); self.rows[tuple(r["key"])] = r

    def has(self, k): return tuple(k) in self.rows

    def add(self, rows):
        with open(self.path, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n"); self.rows[tuple(r["key"])] = r


def chunks(xs, n):
    for i in range(0, len(xs), n): yield xs[i:i + n]


def setup(cfg, a):
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    ex, idx = all_examples(cfg, a.train_path, a.dev_path)
    json.dump({"idx": idx, "qids": [e.qid for e in ex]}, open(cfg.cache_dir / "meta.json", "w"))
    return ex


def cmd_retrieve(cfg, a):
    from .retrieval import retrieve
    ex = setup(cfg, a)
    np.savez(cfg.cache_dir / "retrieval.npz", **retrieve(ex, cfg))


def cmd_summarise(cfg, a):
    ex = setup(cfg, a); sel = np.load(cfg.cache_dir / "retrieval.npz")["sel"]
    st = Store(cfg.cache_dir / "summaries.jsonl"); llm = agents.VLLMAgent(cfg.base_model)
    todo = [(i, r) for i in range(len(ex)) for r in range(3) if not st.has((i, r))]
    for ch in chunks(todo, 1024):
        ps = [llm.chat(agents.SUMM_SYS, f"Question: {ex[i].question}\n\nEvidence:\n{agents.evidence_text(ex[i], sel[i, r])}") for i, r in ch]
        outs = llm.generate(ps, 160)
        st.add([{"key": [i, r], "text": o.outputs[0].text.strip()} for (i, r), o in zip(ch, outs)])


def cmd_answer(cfg, a):
    ex = setup(cfg, a); sel = np.load(cfg.cache_dir / "retrieval.npz")["sel"]
    m = 0 if a.which == "base" else 1
    summ = Store(cfg.cache_dir / "summaries.jsonl"); st = Store(cfg.cache_dir / "answers.jsonl")
    llm = agents.VLLMAgent(cfg.base_model if m == 0 else cfg.spec_model)
    todo = [(i, r) for i in range(len(ex)) for r in range(3) if not st.has((i, r, m))]
    for ch in chunks(todo, 1024):
        ps = [llm.chat(agents.ANS_SYS, f"Question: {ex[i].question}\n\nEvidence:\n{agents.evidence_text(ex[i], sel[i, r])}"
                                       f"\n\nNotes:\n{summ.rows[(i, r)]['text']}\n\nAnswer:") for i, r in ch]
        outs = llm.generate(ps, 24, logprobs=20)                                   # greedy answer + token uncertainty
        souts = llm.generate(ps, 24, n=cfg.n_samples, temperature=0.7)             # K sampled answers (prefix-cached)
        st.add([{"key": [i, r, m], "text": o.outputs[0].text.strip(), **agents.token_stats(o),
                 "samples": [x.text.strip() for x in so.outputs]}
                for (i, r), o, so in zip(ch, outs, souts)])


def cmd_critic(cfg, a):
    ex = setup(cfg, a); sel = np.load(cfg.cache_dir / "retrieval.npz")["sel"]
    ans = Store(cfg.cache_dir / "answers.jsonl"); st = Store(cfg.cache_dir / "critic.jsonl")
    llm = agents.VLLMAgent(cfg.base_model)          # the critic is always the base-size agent
    todo = [k for k in ans.rows if not st.has(k)]
    for ch in chunks(todo, 1024):
        ps = [llm.chat(agents.CRIT_SYS, f"Question: {ex[i].question}\n\nEvidence:\n{agents.evidence_text(ex[i], sel[i, r])}"
                                        f"\n\nProposed answer: {ans.rows[(i, r, m)]['text']}\n\nIs the proposed answer correct and supported? Yes or No:")
              for (i, r, m) in ch]
        outs = llm.generate(ps, 1, logprobs=20)
        st.add([{"key": list(k), "p_yes": agents.p_yes(o)} for k, o in zip(ch, outs)])


def cmd_verify(cfg, a):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from sentence_transformers import SentenceTransformer
    ex = setup(cfg, a); sel = np.load(cfg.cache_dir / "retrieval.npz")["sel"]
    ans = Store(cfg.cache_dir / "answers.jsonl")
    tok = AutoTokenizer.from_pretrained(cfg.nli_model)
    nli = AutoModelForSequenceClassification.from_pretrained(cfg.nli_model).cuda().half().eval()
    lab = {v.lower(): k for k, v in nli.config.id2label.items()}
    ie, ic = lab["entailment"], lab["contradiction"]
    emb = SentenceTransformer(cfg.embed_model, device="cuda")
    keys = sorted(ans.rows)
    N = len(ex); pe = np.zeros((N, 3, 2), np.float32); pc = np.zeros((N, 3, 2), np.float32)
    aemb = np.zeros((N, 3, 2, emb.get_sentence_embedding_dimension()), np.float32)
    for ch in chunks(keys, 64):
        prem = [agents.evidence_text(ex[i], sel[i, r]) for i, r, m in ch]
        hyp = [f"The answer to the question '{ex[i].question}' is {ans.rows[(i, r, m)]['text']}." for i, r, m in ch]
        enc = tok(prem, hyp, truncation="only_first", max_length=512, padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad(): p = torch.softmax(nli(**enc).logits.float(), -1).cpu().numpy()
        e = emb.encode([f"{ex[i].question} {ans.rows[(i, r, m)]['text']}" for i, r, m in ch], normalize_embeddings=True)
        for (i, r, m), pp, ee in zip(ch, p, e):
            pe[i, r, m], pc[i, r, m], aemb[i, r, m] = pp[ie], pp[ic], ee
    np.savez(cfg.cache_dir / "verify.npz", nli_ent=pe, nli_con=pc, aemb=aemb)


def hb(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def sem_feats(samples, greedy):
    """Sampling-based semantic uncertainty: mean F1 of K sampled answers vs the greedy answer, and entropy over
    clusters of identical normalised sampled answers. (Standard semantic-entropy idea; not a novel contribution.)"""
    norm = [normalize_answer(x) for x in samples]
    p = np.array(list(Counter(norm).values()), dtype=float) / len(norm)
    return float(np.mean([f1_score(x, greedy) for x in samples])), float(-(p * np.log(p)).sum())


def cmd_assemble(cfg, a):
    ex = setup(cfg, a); R = np.load(cfg.cache_dir / "retrieval.npz"); V = np.load(cfg.cache_dir / "verify.npz")
    ans = Store(cfg.cache_dir / "answers.jsonl"); cr = Store(cfg.cache_dir / "critic.jsonl")
    N = len(ex); f1 = np.zeros((N, 3, 2), np.float32); em = np.zeros_like(f1); af = np.zeros((N, 3, 2, 13), np.float32)
    missing = 0
    for i in range(N):
        for r in range(3):
            for m in range(2):
                k = (i, r, m)
                if k not in ans.rows or k not in cr.rows: missing += 1; continue
                x = ans.rows[k]; y = cr.rows[k]["p_yes"]
                if "samples" not in x: raise SystemExit("answers.jsonl has no sampled answers: wipe cache and rebuild")
                sa, se = sem_feats(x["samples"], x["text"])
                # cross-model agreement: observable only AFTER the specialist has been paid for (m=1); 0 otherwise
                xm = f1_score(ans.rows[(i, r, 0)]["text"], x["text"]) if m == 1 else 0.0
                ne, nc = float(V["nli_ent"][i, r, m]), float(V["nli_con"][i, r, m])
                f1[i, r, m] = f1_score(x["text"], ex[i].answer); em[i, r, m] = em_score(x["text"], ex[i].answer)
                af[i, r, m] = [x["mean_lp"], x["min_lp"], x["mean_ent"], x["max_ent"], x["n_tok"], y, ne, nc,
                               abs(ne - y), float(hb((ne + y) / 2)), sa, se, xm]
    assert missing == 0, f"{missing} cells missing: finish all stages first"
    np.savez(cfg.cache_dir / "cache.npz", f1=f1, em=em, afeat=af, aemb=V["aemb"], rfeat=R["rfeat"], qemb=R["qemb"],
             remb=R["remb"], gold_recall=R["gold_recall"])
    print("cache.npz written | mean F1 per (variant x model):\n", f1.mean(0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["retrieve", "summarise", "answer", "critic", "verify", "assemble"])
    ap.add_argument("--which", default="base", choices=["base", "spec"])
    ap.add_argument("--train_path"); ap.add_argument("--dev_path")
    ap.add_argument("--cache_dir", default="cache")
    ap.add_argument("--n_train", type=int, default=3000); ap.add_argument("--n_val", type=int, default=1000)
    ap.add_argument("--n_test", type=int, default=2000); ap.add_argument("--dev_offset", type=int, default=0)
    a = ap.parse_args()
    cfg = Cfg(n_train=a.n_train, n_val=a.n_val, n_test=a.n_test, dev_offset=a.dev_offset)
    cfg.cache_dir = __import__("pathlib").Path(a.cache_dir)
    globals()[f"cmd_{a.stage}"](cfg, a)
