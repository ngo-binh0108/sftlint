"""Read dataset files and turn each record into a list of chat turns."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# Roles the common trainers and chat templates understand.
ROLES = {"system", "developer", "user", "assistant", "tool", "function", "ipython"}
TOOL_ROLES = {"tool", "function", "ipython"}

SHAREGPT_ROLES = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "chatgpt": "assistant",
    "function_call": "assistant",
    "observation": "tool",
    "tool": "tool",
}


@dataclass
class Turn:
    role: str
    text: str


@dataclass
class Record:
    path: str
    where: str  # "file:line" for JSONL, "file[index]" for a JSON array
    data: object = None
    fmt: str | None = None
    turns: list[Turn] = field(default_factory=list)
    problems: list[tuple[str, str]] = field(default_factory=list)  # (check code, message) found while reading

    @property
    def text(self) -> str:
        return "\n".join(t.text for t in self.turns)


def _content_text(content) -> str | None:
    """Text of a message's content: a string, or OpenAI-style content parts. None if it is neither."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    parts.append(part["text"])
                elif part.get("type") not in (None, "text"):
                    parts.append(f"[{part['type']}]")  # an image or audio part still counts as content
        return "\n".join(parts)
    return None


def _messages(rec: Record, messages: list) -> None:
    for i, m in enumerate(messages, 1):
        if not isinstance(m, dict):
            rec.problems.append(("bad-turn", f"turn {i} is not an object"))
            continue
        text = _content_text(m.get("content"))
        if text is None:
            rec.problems.append(("bad-turn", f"turn {i} has content that is not text"))
            continue
        calls = m.get("tool_calls") or m.get("function_call")
        if not text.strip() and calls:
            text = json.dumps(calls, ensure_ascii=False)
        rec.turns.append(Turn(str(m.get("role")), text))


def _sharegpt(rec: Record, conversations: list) -> None:
    for i, m in enumerate(conversations, 1):
        if isinstance(m, dict) and "role" in m and "from" not in m:
            _messages(rec, [m])
            continue
        if not isinstance(m, dict) or not isinstance(m.get("value"), str):
            rec.problems.append(("bad-turn", f"turn {i} needs a 'from' and a string 'value'"))
            continue
        speaker = str(m.get("from"))
        rec.turns.append(Turn(SHAREGPT_ROLES.get(speaker.lower(), speaker), m["value"]))


def parse(rec: Record, obj) -> None:
    """Work out the record's format and fill in its turns."""
    rec.data = obj
    if not isinstance(obj, dict):
        rec.problems.append(("not-object", f"record is a JSON {type(obj).__name__}, not an object"))
        return
    if isinstance(obj.get("messages"), list):
        rec.fmt = "messages"
        _messages(rec, obj["messages"])
    elif isinstance(obj.get("conversations"), list):
        rec.fmt = "sharegpt"
        _sharegpt(rec, obj["conversations"])
    elif "instruction" in obj and "output" in obj:
        rec.fmt = "alpaca"
        prompt = str(obj.get("instruction") or "")
        if obj.get("input"):
            prompt += "\n\n" + str(obj["input"])
        if obj.get("system"):
            rec.turns.append(Turn("system", str(obj["system"])))
        rec.turns += [Turn("user", prompt), Turn("assistant", str(obj.get("output") or ""))]
    elif "prompt" in obj and "completion" in obj:
        rec.fmt = "prompt-completion"
        for role, key in (("user", "prompt"), ("assistant", "completion")):
            value = obj[key]
            if isinstance(value, list):  # TRL's conversational prompt-completion
                _messages(rec, value)
            else:
                rec.turns.append(Turn(role, "" if value is None else str(value)))
    elif isinstance(obj.get("text"), str):
        rec.fmt = "text"
        rec.turns.append(Turn("text", obj["text"]))
    else:
        keys = ", ".join(list(obj)[:6]) or "none"
        rec.problems.append(
            (
                "unknown-format",
                f"no messages, conversations, instruction/output, prompt/completion or text field (keys: {keys})",
            )
        )


def load(path: str) -> list[Record]:
    """Read a .jsonl file (one record per line) or a .json file holding an array of records."""
    raw = Path(path).read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]

    if Path(path).suffix.lower() == ".json":
        try:
            data = json.loads(raw.decode("utf-8"))
        except UnicodeDecodeError:
            return [Record(path, path, problems=[("bad-encoding", "file is not valid UTF-8")])]
        except json.JSONDecodeError as e:
            if e.msg != "Extra data":  # "Extra data" means it is really JSON Lines
                return [Record(path, f"{path}:{e.lineno}", problems=[("bad-json", f"not valid JSON ({e.msg})")])]
        else:
            records = []
            for i, obj in enumerate(data if isinstance(data, list) else [data]):
                rec = Record(path, f"{path}[{i}]")
                parse(rec, obj)
                records.append(rec)
            return records

    records = []
    for n, line in enumerate(raw.split(b"\n"), 1):
        if not line.strip():
            continue
        rec = Record(path, f"{path}:{n}")
        records.append(rec)
        try:
            obj = json.loads(line.decode("utf-8"))
        except UnicodeDecodeError:
            rec.problems.append(("bad-encoding", "line is not valid UTF-8"))
            continue
        except json.JSONDecodeError as e:
            rec.problems.append(("bad-json", f"not valid JSON ({e.msg} at column {e.colno})"))
            continue
        parse(rec, obj)
    return records
