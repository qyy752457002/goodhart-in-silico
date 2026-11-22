"""Prompts for LLM persona simulators.

Methodology map (03_methodology.md): sections 2.1 and 2.3.
  R1a named persona      organisation name + stakeholder type + proposal text
  R1b anonymous persona  stakeholder type + one-sentence description (no name) + proposal text
  R2  context persona    as above, plus the organisation's positions on other questions

The model is asked to *verbalise a probability distribution* over the stance classes as JSON
(Meister et al., 2025), conditional on the organisation having answered the question. The order in
which the classes are listed is shuffled per call (Dominguez-Olmedo et al., 2024).

Leakage rules, enforced by `PromptSource` implementations and checked in the test-suite:
  * a prompt never contains the organisation's own answer (or its extracted summary) to the
    question being predicted, nor to a question that shares its decision variable;
  * the anonymous variant never contains the organisation's name or acronym (`redact`).

Bump PROMPT_VERSION whenever the wording changes: it is part of the cache key.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np

PROMPT_VERSION = "v1"
KEYS_MARKER = "Keys: "

SYSTEM_PROMPT = (
    "You are an expert analyst of public consultations on accounting and sustainability reporting "
    "standards. Your task is to estimate how a specific respondent answered a consultation question, "
    "given who the respondent is. Base the estimate on the respondent's interests and the positions "
    "that respondents of its kind typically take. Answer with a single JSON object and nothing else."
)

# Stance classes of Dataset A, described for the model. The wording is ours (no annotation codebook
# is distributed with the dataset): edit it if the AASB staff definitions become available.
STANCE_DESCRIPTIONS = {
    "agree": "supports the proposal as drafted",
    "qualified_agree": "supports the proposal overall, but with conditions, reservations or requested changes",
    "mixed": "expresses both support and opposition, with no clear overall position",
    "qualified_disagree": "opposes the proposal overall, while accepting parts of it or acknowledging some merit",
    "alternative_proposal": "does not accept the proposal and puts forward a different approach instead",
    "disagree": "opposes the proposal",
}

STANCE_WORDS = {
    "agree": "agreed", "qualified_agree": "agreed with qualifications", "mixed": "gave a mixed view",
    "qualified_disagree": "disagreed with qualifications",
    "alternative_proposal": "proposed an alternative approach", "disagree": "disagreed",
}

# One-line description of each fine stakeholder type (used by the named and anonymous variants).
FINE_TYPE_DESCRIPTIONS = {
    "CORP": "a company that prepares financial reports and would have to apply the proposed "
            "climate-related disclosure requirements",
    "ASSOC": "an industry or business association representing companies that would have to prepare "
             "the proposed disclosures",
    "INV": "an institutional investor, asset owner, asset manager or investor body, i.e. a user of "
           "climate-related financial disclosures",
    "ASSUR": "an audit and assurance firm or a professional body for accountants, actuaries or "
             "governance practitioners",
    "SERV": "a professional services provider (advisory, legal, data or ESG ratings)",
    "PUBLIC": "a public-sector body (regulator, government agency, treasury or local council)",
    "ACAD": "an academic or a university research group",
    "CIVIL": "a civil-society organisation (environmental group, NGO or trade union)",
    "INDIV": "an individual responding in a personal capacity",
}


@dataclass(frozen=True)
class Option:
    key: str      # JSON key the model must use
    text: str     # description shown to the model
    stance: str   # canonical stance class it maps to


@dataclass(frozen=True)
class ContextItem:
    question: str
    stance_text: str
    rationale: str = ""


class PromptSource(Protocol):
    """What a testbed must provide so that the persona runner can build prompts."""
    consultation: str

    def persona(self, org_index: int, variant: str) -> str: ...
    def question(self, qid: int) -> str: ...
    def background(self, qid: int) -> str: ...
    def options(self, qid: int) -> list: ...
    def context(self, org_index: int, qid: int, anonymous: bool) -> list: ...


def stance_options() -> list:
    """The six stance classes as answer options (AASB testbed, synthetic testbed)."""
    return [Option(key=s, text=text, stance=s) for s, text in STANCE_DESCRIPTIONS.items()]


def stable_rng(*parts) -> np.random.Generator:
    """A generator seeded by a platform-independent hash of the arguments."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def shuffled(options: list, *seed_parts) -> list:
    """Options in a pseudo-random order that is a deterministic function of `seed_parts`."""
    order = stable_rng("option-order", *seed_parts).permutation(len(options))
    return [options[j] for j in order]


def org_aliases(org: str) -> list:
    """Name variants to hide in anonymous prompts: the full name, the name without a trailing
    acronym, and the acronym itself, e.g. 'Minerals Council of Australia (MCA)' -> 3 aliases."""
    aliases = {org.strip()}
    m = re.match(r"^(.*?)\s*\(([^)]+)\)\s*$", org)
    if m:
        aliases.update({m.group(1).strip(), m.group(2).strip()})
    return sorted((a for a in aliases if len(a) >= 2), key=len, reverse=True)


def redact(text: str, org: str, replacement: str = "the respondent") -> str:
    """Remove every alias of `org` from `text` (whole-word, case-insensitive for multi-word names,
    case-sensitive for short acronyms so that ordinary words are left alone)."""
    for alias in org_aliases(org):
        flags = 0 if (len(alias) <= 5 and alias.isupper()) else re.IGNORECASE
        text = re.sub(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", replacement, text, flags=flags)
    return text


def contains_alias(text: str, org: str) -> bool:
    return redact(text, org, replacement="\x00") != text


def _truncate(text: str, max_chars: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= max_chars else text[: max_chars - 3].rstrip() + "..."


def context_block(items: list, max_items: int = 12, max_chars: int = 220) -> str:
    if not items:
        return ""
    lines = ["The same respondent's positions on other questions of this consultation:"]
    for j, item in enumerate(items[:max_items], 1):
        line = f"{j}. On \"{_truncate(item.question, max_chars)}\" the respondent {item.stance_text}."
        if item.rationale:
            line += f" Reasons: {_truncate(item.rationale, max_chars)}"
        lines.append(line)
    return "\n".join(lines) + "\n\n"


def build_messages(source: PromptSource, org_index: int, qid: int, variant: str, options: list,
                   identity: str = "named", max_context_items: int = 12) -> list:
    """Chat messages for one (organisation, question) cell.

    variant:  'named' | 'anonymous' | 'context'
    identity: for the context variant, whether the persona itself is named or anonymous.
    options:  answer options in the order they should be listed (already shuffled).
    """
    if variant not in ("named", "anonymous", "context"):
        raise ValueError(f"unknown persona variant {variant!r}")
    persona_variant = identity if variant == "context" else variant
    if persona_variant not in ("named", "anonymous"):
        raise ValueError(f"unknown identity {identity!r}")
    parts = [f"Consultation: {source.consultation}\n",
             f"Respondent: {source.persona(org_index, persona_variant)}\n"]
    background = source.background(qid).strip()
    if background:
        parts.append(f"Background to the question, from the consultation document:\n\"\"\"\n{background}\n\"\"\"\n")
    parts.append(f"Question put to respondents:\n\"\"\"\n{source.question(qid).strip()}\n\"\"\"\n")
    if variant == "context":
        items = source.context(org_index, qid, anonymous=(persona_variant == "anonymous"))
        parts.append(context_block(items, max_context_items).rstrip("\n") + "\n" if items else
                     "No other positions of this respondent are available.\n")
    listing = "\n".join(f"- \"{o.key}\": {o.text}" for o in options)
    parts.append(
        "This respondent did answer this question in its written submission. Estimate the probability "
        "that its answer falls into each of the following categories:\n"
        f"{listing}\n\n"
        "Return one JSON object that maps every category key to a probability between 0 and 1. "
        "The probabilities must sum to 1. Use exactly these keys and no others.\n"
        f"{KEYS_MARKER}{json.dumps([o.key for o in options])}"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "\n".join(parts)}]


def keys_in_prompt(messages: list) -> list:
    """Recover the option keys from a prompt built by `build_messages` (used by the mock backend)."""
    for line in reversed(messages[-1]["content"].splitlines()):
        if line.startswith(KEYS_MARKER):
            return json.loads(line[len(KEYS_MARKER):])
    raise ValueError("the prompt does not list its answer keys")
