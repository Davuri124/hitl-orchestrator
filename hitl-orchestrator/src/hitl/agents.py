"""LLM agents (summariser, answerer, critic) on vLLM + prompt templates + token-level uncertainty."""
import numpy as np

SUMM_SYS = ("You are a careful research assistant. Summarise the evidence into at most 5 short factual "
            "bullet points relevant to the question. Use only the evidence.")
ANS_SYS = ("Answer the question using only the evidence and notes. Reply with the shortest possible answer "
           "(a few words, or yes/no). No explanation.")
CRIT_SYS = ("You are a strict reviewer. Decide whether the proposed answer is correct and fully supported by "
            "the evidence. Reply with exactly one word: Yes or No.")


def evidence_text(e, sel):
    return "\n\n".join(f"[{e.paragraphs[j][0]}]\n{e.paragraphs[j][1]}" for j in sel if j >= 0)


class VLLMAgent:
    def __init__(self, model, max_len=4096, gpu_util=0.88):
        from vllm import LLM
        quant = "awq" if "awq" in model.lower() else None
        self.llm = LLM(model=model, dtype="half", max_model_len=max_len, quantization=quant,
                       gpu_memory_utilization=gpu_util, enable_prefix_caching=True)
        self.tok = self.llm.get_tokenizer()

    def chat(self, system, user):
        return self.tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                            tokenize=False, add_generation_prompt=True)

    def generate(self, prompts, max_tokens, logprobs=None, n=1, temperature=0.0):
        from vllm import SamplingParams
        return self.llm.generate(prompts, SamplingParams(temperature=temperature, max_tokens=max_tokens,
                                                         logprobs=logprobs, n=n), use_tqdm=True)


def token_stats(out):
    """Mean/min sampled-token logprob and top-20 entropy over the generated answer tokens."""
    o = out.outputs[0]
    if not o.token_ids or o.logprobs is None:
        return dict(mean_lp=-5.0, min_lp=-5.0, mean_ent=3.0, max_ent=3.0, n_tok=0)
    lps, ents = [], []
    for tid, d in zip(o.token_ids, o.logprobs):
        lps.append(d[tid].logprob)
        p = np.exp([v.logprob for v in d.values()])
        ents.append(float(-(p * np.log(p + 1e-12)).sum()))
    return dict(mean_lp=float(np.mean(lps)), min_lp=float(np.min(lps)),
                mean_ent=float(np.mean(ents)), max_ent=float(np.max(ents)), n_tok=len(lps))


def p_yes(out):
    """Critic confidence read from the Yes/No next-token distribution (not self-reported text)."""
    py = pn = 0.0
    for lp in out.outputs[0].logprobs[0].values():
        t = (lp.decoded_token or "").strip().lower()
        if t == "yes": py += float(np.exp(lp.logprob))
        elif t == "no": pn += float(np.exp(lp.logprob))
    return py / (py + pn) if py + pn > 0 else 0.5
