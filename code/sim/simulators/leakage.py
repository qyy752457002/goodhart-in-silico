"""Leakage controls for LLM simulators (methodology section 2.2).

The comment letters on ED SR1 closed on 2024-03-01 and were then published, so a model trained
later may have *read* the answers it is asked to predict. Three controls:

  1. training cut-off     compare a model whose cut-off precedes the letters with a recent one;
  2. named vs anonymous   if naming the organisation helps a lot, the model may be recalling it
                          (or legitimately knows its public positions);
  3. verbatim recall      ask the model to continue a passage of a real submission and compare the
                          continuation with the true text, against a topic-matched control.

Controls 1 and 2 are contrasts between simulator outputs: `metrics.paired_accuracy_contrast`.
This module implements control 3.
"""
from __future__ import annotations

import difflib
import json
import os
import re

import numpy as np
import pandas as pd

from .backends import ChatBackend

RECALL_SYSTEM = "You reproduce documents exactly. Continue the given text word for word; output only the continuation."


def load_passages(responses_dir: str, min_words: int = 80) -> pd.DataFrame:
    """Passages from a directory of per-organisation JSON files, each a list of
    {"question_id": "Q7", "organization": "...", "response_text": "..."} records.

    Note: the probe is only as verbatim as the texts are. If `response_text` holds extracted or
    summarised answers rather than the letters' original wording, supply original excerpts instead.
    """
    rows = []
    for fname in sorted(os.listdir(responses_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(responses_dir, fname), encoding="utf-8") as f:
            for rec in json.load(f):
                text = " ".join(str(rec.get("response_text") or "").split())
                if len(text.split()) >= min_words:
                    rows.append({"org": re.sub(r"_(edsr1|aasb)_.*$", "", str(rec.get("organization", fname))),
                                 "qid": str(rec.get("question_id", "")), "text": text})
    return pd.DataFrame(rows, columns=["org", "qid", "text"])


def split_passage(text: str, prefix_words: int = 60, suffix_words: int = 40) -> tuple[str, str]:
    words = text.split()
    return " ".join(words[:prefix_words]), " ".join(words[prefix_words: prefix_words + suffix_words])


def _tokens(text: str) -> list:
    return re.findall(r"[a-z0-9]+", text.lower())


def overlap_scores(candidate: str, reference: str) -> dict:
    """Similarity of a continuation to a reference: token-sequence ratio in [0, 1] and the longest
    run of consecutive shared tokens (long runs are the signature of memorisation)."""
    a, b = _tokens(candidate), _tokens(reference)
    if not a or not b:
        return {"ratio": 0.0, "longest_run": 0}
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    longest = matcher.find_longest_match(0, len(a), 0, len(b)).size
    return {"ratio": float(matcher.ratio()), "longest_run": int(longest)}


def recall_probe(passages: pd.DataFrame, backend: ChatBackend, n: int = 20, rng: np.random.Generator = None,
                 prefix_words: int = 60, suffix_words: int = 40, max_tokens: int = 120,
                 consultation: str = "AASB Exposure Draft ED SR1") -> pd.DataFrame:
    """Verbatim-recall probe.

    For each sampled passage the model sees the first `prefix_words` words and is asked to continue
    the text exactly. The continuation is scored against the true next `suffix_words` words and
    against a control: the same span of a *different* organisation's answer to the same question
    (same topic, so shared vocabulary does not count as recall). A true-minus-control gap in
    `longest_run` or `ratio` that is clearly positive indicates memorised text.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    if passages.empty:
        raise ValueError("no passages to probe")
    picks = rng.choice(len(passages), size=min(n, len(passages)), replace=False)
    rows = []
    for j in picks:
        rec = passages.iloc[int(j)]
        prefix, suffix = split_passage(rec["text"], prefix_words, suffix_words)
        others = passages[(passages["qid"] == rec["qid"]) & (passages["org"] != rec["org"])]
        control = split_passage(others.iloc[int(rng.integers(len(others)))]["text"], prefix_words,
                                suffix_words)[1] if len(others) else ""
        user = (f"The following is the beginning of the written response of {rec['org']} to question "
                f"{rec['qid']} of the {consultation} consultation. Continue it exactly as it was written.\n\n"
                f"{prefix}")
        reply = backend.complete([{"role": "system", "content": RECALL_SYSTEM}, {"role": "user", "content": user}],
                                 temperature=0.0, max_tokens=max_tokens, seed=0, json_mode=False)
        true_s, ctrl_s = overlap_scores(reply.text, suffix), overlap_scores(reply.text, control)
        rows.append({"org": rec["org"], "qid": rec["qid"], "ratio_true": true_s["ratio"],
                     "ratio_control": ctrl_s["ratio"], "run_true": true_s["longest_run"],
                     "run_control": ctrl_s["longest_run"], "continuation": reply.text[:400]})
    out = pd.DataFrame(rows)
    out["ratio_gap"] = out["ratio_true"] - out["ratio_control"]
    out["run_gap"] = out["run_true"] - out["run_control"]
    return out
