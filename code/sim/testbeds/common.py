"""What a testbed hands to the experiments, and a prompt source that works for any `Problem`."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from ..data import Problem
from ..simulators import prompts


@dataclass(eq=False)
class Testbed:
    """A consultation ready for the experiments.

    prob          the decision problem (true stances are used only for pilots and evaluation);
    source        builds prompts for the LLM persona simulators (None if not applicable);
    context       (n_org, n_context_questions) support on questions outside the decision set,
                  NaN where unanswered -- available to relational simulators in scenario S2;
    board         the regulator's actual package, if known (diagnostic only).
    """
    name: str
    prob: Problem
    source: object = None
    context: np.ndarray | None = None
    context_qids: list = field(default_factory=list)
    board: np.ndarray | None = None
    meta: dict = field(default_factory=dict)


def strip_question_number(text: str) -> str:
    return re.sub(r"^\s*\[Question \d+\]\s*", "", str(text))


class ProblemPromptSource:
    """Generic prompt source: personas are '<organisation>, a member of group <group>' and the
    context of a cell is the organisation's stances on its *other* decisions (leave-one-out).

    Used for the synthetic and toy testbeds and as the base class of the AASB source.
    """
    consultation = "A public consultation on a set of policy proposals."

    def __init__(self, prob: Problem):
        self.prob = prob
        self._cells_of_org = [np.flatnonzero(prob.cell_org == i) for i in range(prob.n_org)]

    def persona(self, org_index: int, variant: str) -> str:
        group = self.prob.groups[org_index]
        if variant == "anonymous":
            return f"an organisation in the stakeholder group \"{group}\" (name withheld)."
        return f"{self.prob.orgs[org_index]}, an organisation in the stakeholder group \"{group}\"."

    def question(self, qid: int) -> str:
        q = (self.prob.questions or {}).get(int(qid))
        return strip_question_number(q["question"]) if q else f"Do you agree with proposal {qid}?"

    def background(self, qid: int) -> str:
        q = (self.prob.questions or {}).get(int(qid))
        return q.get("background", "") if q else ""

    def options(self, qid: int) -> list:
        return prompts.stance_options()

    def context(self, org_index: int, qid: int, anonymous: bool = False) -> list:
        """Stances of the same organisation on other decisions. Never includes the target question
        or any question that shares its decision variable."""
        prob = self.prob
        cells = self._cells_of_org[org_index]
        target_dec = {int(prob.cell_dec[k]) for k in cells if int(prob.cell_q[k]) == int(qid)}
        items = []
        for k in sorted(cells, key=lambda k: int(prob.cell_q[k])):
            if int(prob.cell_q[k]) == int(qid) or int(prob.cell_dec[k]) in target_dec:
                continue
            items.append(prompts.ContextItem(question=self.question(int(prob.cell_q[k])),
                                             stance_text=prompts.STANCE_WORDS[prob.cell_stance[k]]))
        return items
