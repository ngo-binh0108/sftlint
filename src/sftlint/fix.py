"""Remove chat-template tokens that were pasted into a dataset's text."""

from __future__ import annotations

import re

from .checks import TEMPLATE_TOKENS

_ROLE = r"(?:system|developer|user|human|assistant|model|gpt|tool|ipython)"
# A token that opens a turn header takes the role name and line break after it along: "<|im_start|>assistant\n",
# "<|start_header_id|>user<|end_header_id|>\n\n" and "<start_of_turn>model\n" all go as a whole.
_HEADER = rf"(?:<\|im_start\|>|<\|start_header_id\|>|<start_of_turn>)[ \t]*{_ROLE}?[ \t]*(?:<\|end_header_id\|>)?\n*"
_ANY = "|".join(re.escape(t) for t in TEMPLATE_TOKENS)
_START = re.compile(rf"\A\s*(?:{_HEADER}|{_ANY})")
_END = re.compile(rf"(?:{_ANY})\s*\Z")


def strip_text(text: str) -> str:
    """The text without template tokens at its start or end.

    Tokens in the middle mark turn boundaries: the string holds a whole templated conversation, and taking the
    tokens out would run the turns together. Such text is returned unchanged.
    """
    s = text
    while m := _START.match(s):
        s = s[m.end() :]
    while m := _END.search(s):
        s = s[: m.start()]
    if s == text or any(t in s for t in TEMPLATE_TOKENS):
        return text
    return s.strip()


def strip_tokens(value):
    """A copy of a record with template tokens taken off every string in it."""
    if isinstance(value, str):
        return strip_text(value)
    if isinstance(value, list):
        return [strip_tokens(v) for v in value]
    if isinstance(value, dict):
        return {k: strip_tokens(v) for k, v in value.items()}
    return value
