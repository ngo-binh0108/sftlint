"""Remove chat-template tokens that were pasted into a dataset's text."""

from __future__ import annotations

import re

from .checks import END_TOKENS, TEMPLATE_TOKENS, tokens_in

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
    if not tokens_in(text) or len(tokens_in(text)) < sum(t in text for t in TEMPLATE_TOKENS):
        return text  # no tokens, or a markup </s> that taking tokens off would break
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


ASSISTANT_SPEAKERS = {"assistant", "gpt", "chatgpt", "model"}


def cut_text(text: str) -> str:
    """The text up to its first end token, if more text follows it; otherwise the text unchanged."""
    cuts = [text.find(t) for t in END_TOKENS if t in tokens_in(text)]
    if not cuts:
        return text
    head, tail = text[: min(cuts)], text[min(cuts) :]
    for t in END_TOKENS:
        if tail.startswith(t):
            tail = tail[len(t) :]
            break
    return head.rstrip() if tail.strip() else text


def _cut_content(content):
    if isinstance(content, str):
        return cut_text(content)
    if isinstance(content, list):  # content parts
        return [{**p, "text": cut_text(p["text"])} if isinstance(p, dict) and isinstance(p.get("text"), str) else p for p in content]
    return content


def _cut_messages(messages, role_key="role", content_key="content"):
    if not isinstance(messages, list):
        return messages
    out = []
    for m in messages:
        if isinstance(m, dict) and str(m.get(role_key, "")).lower() in ASSISTANT_SPEAKERS and content_key in m:
            m = {**m, content_key: _cut_content(m[content_key])}
        out.append(m)
    return out


def cut_run_on(obj, fmt: str | None):
    """A copy of a record whose answers stop at their first end token, when the text runs on after it.

    Only answers are cut: in a prompt, an end token in the middle marks a turn boundary, not a model that did not stop.
    """
    if not isinstance(obj, dict):
        return obj
    obj = dict(obj)
    if fmt == "messages":
        obj["messages"] = _cut_messages(obj["messages"])
    elif fmt == "sharegpt":
        obj["conversations"] = [
            _cut_messages([m], "from", "value")[0] if isinstance(m, dict) and "from" in m else _cut_messages([m])[0]
            for m in obj["conversations"]
        ]
    elif fmt == "alpaca" and isinstance(obj.get("output"), str):
        obj["output"] = cut_text(obj["output"])
    elif fmt == "prompt-completion":
        c = obj["completion"]
        obj["completion"] = _cut_messages(c) if isinstance(c, list) else cut_text(c) if isinstance(c, str) else c
    elif fmt == "preference":
        for key in ("chosen", "rejected"):
            v = obj[key]
            if isinstance(v, list):
                obj[key] = _cut_messages(v)
            elif isinstance(v, str) and obj.get("prompt") is not None:  # a bare string answer; text pairs are left alone
                obj[key] = cut_text(v)
    return obj
