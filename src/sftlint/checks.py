"""Checks that look at one record at a time."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable

from .formats import ROLES, TOOL_ROLES, Record, Turn

# code: (level, what it means)
CHECKS = {
    "bad-json": ("error", "the line is not valid JSON"),
    "bad-encoding": ("error", "the line is not valid UTF-8"),
    "not-object": ("error", "the record is not a JSON object"),
    "unknown-format": ("error", "the record is in none of the supported formats"),
    "bad-turn": ("error", "a turn is malformed"),
    "empty-turn": ("error", "a turn has no text"),
    "unknown-role": ("error", "a turn has a role chat templates do not know"),
    "no-assistant": ("error", "there is no assistant turn to learn from"),
    "eval-leak": ("error", "an eval record also appears in the training data"),
    "same-answer": ("error", "the chosen and rejected answers are the same"),
    "prompt-mismatch": ("error", "the chosen and rejected answers continue different prompts"),
    "role-order": ("warning", "turns are out of order"),
    "ends-with-user": ("warning", "the last turn is not from the assistant"),
    "template-token": ("warning", "the text contains chat-template tokens"),
    "boilerplate": ("warning", "an assistant turn (or a chosen answer) has AI boilerplate or a refusal"),
    "too-long": ("warning", "the record is longer than --max-tokens"),
    "duplicate": ("warning", "the record repeats an earlier one"),
    "near-duplicate": ("warning", "the record is almost the same as an earlier one"),
}

TEMPLATE_TOKENS = [
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<|begin_of_text|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
    "<start_of_turn>",
    "<end_of_turn>",
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
    "<|end|>",
    "[INST]",
    "[/INST]",
    "<</SYS>>",
    "</s>",
]

BOILERPLATE = re.compile(
    r"as an ai language model|as a large language model|as an ai(?: assistant| model)?,? (?:developed|created|trained) by"
    r"|i am an ai (?:developed|created|trained) by|my knowledge cutoff|i don't have personal (?:opinions|feelings|experiences)"
    r"|i'm sorry, but i (?:can't|cannot) (?:help|assist|fulfill|comply)|i cannot fulfill (?:this|that|your) request",
    re.IGNORECASE,
)

_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]")


@dataclass
class Issue:
    code: str
    where: str
    message: str

    @property
    def level(self) -> str:
        return CHECKS[self.code][0]


def approx_tokens(text: str) -> int:
    """Rough token count: about four characters per token, one per CJK character."""
    cjk = len(_CJK.findall(text))
    return math.ceil((len(text) - cjk) / 4 + cjk)


def _role_order(turns: list[Turn]) -> str | None:
    roles = [t.role for t in turns]
    start = 0
    while start < len(roles) and roles[start] in ("system", "developer"):
        start += 1
    prev = None
    for i, role in enumerate(roles[start:], start + 1):
        if role in ("system", "developer"):
            return f"turn {i} is a {role} turn after the conversation started"
        if role in TOOL_ROLES:
            if prev not in ("assistant", "tool"):
                return f"turn {i} is a tool result that does not follow an assistant turn"
            prev = "tool"
            continue
        if prev is None and role == "assistant":
            return "the conversation starts with an assistant turn"
        if role == prev and role in ("user", "assistant"):
            return f"turns {i - 1} and {i} are both {role} turns"
        prev = role
    return None


def _turn_issues(turns: list[Turn], where: str, first: int = 1, label: str = "turn", boilerplate: bool = True) -> list[Issue]:
    """Checks on single turns, numbered from `first`."""
    issues = []
    for i, t in enumerate(turns, first):
        if t.role not in ROLES:
            issues.append(Issue("unknown-role", where, f"{label} {i} has role {t.role!r}"))
        if not t.text.strip():
            issues.append(Issue("empty-turn", where, f"{label} {i} ({t.role}) is empty"))
    found = next(((i, t, tok) for i, t in enumerate(turns, first) for tok in TEMPLATE_TOKENS if tok in t.text), None)
    if found:
        i, t, tok = found
        issues.append(Issue("template-token", where, f"{label} {i} ({t.role}) contains {tok}; the chat template adds these itself"))
    if boilerplate:
        for i, t in enumerate(turns, first):
            m = BOILERPLATE.search(t.text) if t.role == "assistant" else None
            if m:
                issues.append(Issue("boilerplate", where, f"{label} {i} says {m.group(0)!r}"))
                break
    return issues


def _structure_issues(turns: list[Turn], where: str, malformed: bool) -> list[Issue]:
    """Checks on the conversation as a whole."""
    if malformed:
        return []  # with a malformed turn left out, the shape of the rest says nothing
    if "assistant" not in [t.role for t in turns]:
        return [Issue("no-assistant", where, "no assistant turn")]
    issues = []
    order = _role_order(turns)
    if order:
        issues.append(Issue("role-order", where, order))
    last = next((t.role for t in reversed(turns) if t.role not in ("system", "developer")), None)
    if last != "assistant":
        issues.append(Issue("ends-with-user", where, f"the last turn is a {last} turn, so nothing after it is learned"))
    return issues


def check_record(rec: Record, max_tokens: int | None = None, count: Callable[[str], int] = approx_tokens) -> list[Issue]:
    issues = [Issue(code, rec.where, message) for code, message in rec.problems]
    if rec.fmt is None:
        return issues
    malformed = any(code == "bad-turn" for code, _ in rec.problems)

    if rec.fmt == "text":
        # A pre-formatted text field is supposed to hold template tokens, so it only gets the length checks.
        if not rec.text.strip():
            issues.append(Issue("empty-turn", rec.where, "the text field is empty"))
    elif rec.rejected is None:
        issues += _turn_issues(rec.turns, rec.where)
        issues += _structure_issues(rec.turns, rec.where, malformed)
    elif len(rec.turns) == 1 and rec.turns[0].role == "text":  # a preference pair of pre-formatted dialogues
        for side, turns in (("chosen", rec.turns), ("rejected", rec.rejected)):
            if not turns[0].text.strip():
                issues.append(Issue("empty-turn", rec.where, f"the {side} text is empty"))
        if rec.turns[0].text.strip() == rec.rejected[0].text.strip() != "":
            issues.append(Issue("same-answer", rec.where, "chosen and rejected are the same, so the pair teaches nothing"))
    else:
        # The prompt and the chosen answer get every check. A rejected answer may well be a refusal,
        # so it skips the boilerplate check, and only problems that differ from the chosen side are reported.
        prompt = rec.turns[: rec.prompt_turns]
        issues += _turn_issues(rec.turns, rec.where)
        issues += _turn_issues(rec.rejected, rec.where, rec.prompt_turns + 1, "rejected turn", boilerplate=False)
        chosen = _structure_issues(rec.turns, rec.where, malformed)
        seen = {(i.code, i.message) for i in chosen}
        issues += chosen
        for i in _structure_issues(prompt + rec.rejected, rec.where, malformed):
            if (i.code, i.message) not in seen:
                issues.append(Issue(i.code, i.where, f"rejected side: {i.message}"))
        answers = [[(t.role, t.text.strip()) for t in side] for side in (rec.turns[rec.prompt_turns :], rec.rejected)]
        if answers[0] == answers[1] and answers[0]:
            issues.append(Issue("same-answer", rec.where, "chosen and rejected are the same, so the pair teaches nothing"))

    if max_tokens:
        n = max(count(text) for text in rec.sequences)
        if n > max_tokens:
            issues.append(Issue("too-long", rec.where, f"{n:,} tokens, over the limit of {max_tokens:,}"))
    return issues
