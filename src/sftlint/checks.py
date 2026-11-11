"""Checks that look at one record at a time."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable

from .formats import ROLES, TOOL_ROLES, Record

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
    "role-order": ("warning", "turns are out of order"),
    "ends-with-user": ("warning", "the last turn is not from the assistant"),
    "template-token": ("warning", "the text contains chat-template tokens"),
    "boilerplate": ("warning", "an assistant turn has AI boilerplate or a refusal"),
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


def _role_order(rec: Record) -> str | None:
    roles = [t.role for t in rec.turns]
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


def check_record(rec: Record, max_tokens: int | None = None, count: Callable[[str], int] = approx_tokens) -> list[Issue]:
    issues = [Issue(code, rec.where, message) for code, message in rec.problems]
    if rec.fmt is None:
        return issues

    if rec.fmt == "text":
        if not rec.text.strip():
            issues.append(Issue("empty-turn", rec.where, "the text field is empty"))
    else:
        for i, t in enumerate(rec.turns, 1):
            if t.role not in ROLES:
                issues.append(Issue("unknown-role", rec.where, f"turn {i} has role {t.role!r}"))
            if not t.text.strip():
                issues.append(Issue("empty-turn", rec.where, f"turn {i} ({t.role}) is empty"))

        roles = [t.role for t in rec.turns]
        if "assistant" not in roles:
            issues.append(Issue("no-assistant", rec.where, "no assistant turn"))
        else:
            # With a malformed turn left out, the order of the rest says nothing.
            order = None if any(code == "bad-turn" for code, _ in rec.problems) else _role_order(rec)
            if order:
                issues.append(Issue("role-order", rec.where, order))
            last = next((t.role for t in reversed(rec.turns) if t.role not in ("system", "developer")), None)
            if last != "assistant":
                issues.append(Issue("ends-with-user", rec.where, f"the last turn is a {last} turn, so nothing after it is learned"))

        # A pre-formatted text field is supposed to hold template tokens, so only look in chat records.
        found = next(((i, t, tok) for i, t in enumerate(rec.turns, 1) for tok in TEMPLATE_TOKENS if tok in t.text), None)
        if found:
            i, t, tok = found
            issues.append(Issue("template-token", rec.where, f"turn {i} ({t.role}) contains {tok}; the chat template adds these itself"))

        for i, t in enumerate(rec.turns, 1):
            m = BOILERPLATE.search(t.text) if t.role == "assistant" else None
            if m:
                issues.append(Issue("boilerplate", rec.where, f"turn {i} says {m.group(0)!r}"))
                break

    if max_tokens:
        n = count(rec.text)
        if n > max_tokens:
            issues.append(Issue("too-long", rec.where, f"{n:,} tokens, over the limit of {max_tokens:,}"))
    return issues
