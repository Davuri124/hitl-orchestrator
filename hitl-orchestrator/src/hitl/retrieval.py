"""Retrieval agent. Three evidence variants per query (all computed, then cached):
 0 narrow : top-k_narrow by dense similarity (cheap default)
 1 wide   : "re-retrieve with modified constraints" = bigger k + hop-2 query expansion with the top-1 paragraph
 2 gold   : what a human reviewer would supply (gold supporting paragraphs)
"""
import numpy as np

QPREF = "Represent this sentence for searching relevant passages: "


def _feats(s, sel):
    srt = np.sort(s)[::-1]
    p = np.exp((s - s.max()) / 0.05); p /= p.sum()
    ent = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(s)))
    return [srt[0], srt[0] - srt[1], ent, s[sel].mean(), s[sel].min(), float(len(sel))]


def retrieve(examples, cfg, device="cuda"):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(cfg.embed_model, device=device)
    enc = lambda t: m.encode(t, batch_size=256, normalize_embeddings=True, show_progress_bar=True)
    texts, offs = [], []
    for e in examples:
        offs.append(len(texts)); texts += [f"{t}. {x}" for t, x in e.paragraphs]
    P = enc(texts)
    qe = enc([QPREF + e.question for e in examples])
    s1 = [P[offs[i]:offs[i] + len(e.paragraphs)] @ qe[i] for i, e in enumerate(examples)]
    qe2 = enc([QPREF + e.question + " " + e.paragraphs[int(np.argmax(s1[i]))][1][:300] for i, e in enumerate(examples)])
    N, KW = len(examples), cfg.k_wide
    sel = -np.ones((N, 3, KW), dtype=int)
    rfeat = np.zeros((N, 3, 6), dtype=np.float32)
    remb = np.zeros((N, 3, qe.shape[1]), dtype=np.float32)
    grec = np.zeros((N, 3), dtype=np.float32)
    for i, e in enumerate(examples):
        Pi = P[offs[i]:offs[i] + len(e.paragraphs)]
        a = s1[i]; b = np.maximum(a, Pi @ qe2[i])
        gold = [j for j, (t, _) in enumerate(e.paragraphs) if t in set(e.gold_titles)]
        variants = [(np.argsort(-a)[:cfg.k_narrow], a), (np.argsort(-b)[:KW], b), (np.array(gold[:KW]), a)]
        for r, (ix, sc) in enumerate(variants):
            sel[i, r, :len(ix)] = ix
            rfeat[i, r] = _feats(sc, ix)
            remb[i, r] = Pi[ix].mean(0)
            grec[i, r] = len({e.paragraphs[j][0] for j in ix} & set(e.gold_titles)) / max(1, len(e.gold_titles))
    return dict(sel=sel, rfeat=rfeat, qemb=qe.astype(np.float32), remb=remb, gold_recall=grec)
