import json
import random

import pytest

from sftlint import cli, dedupe
from sftlint.checks import CHECKS, approx_tokens, check_record
from sftlint.formats import Record, load, parse


def rec(obj, where="t.jsonl:1"):
    r = Record("t.jsonl", where)
    parse(r, obj)
    return r


def codes(obj, **kw):
    return [i.code for i in check_record(rec(obj), **kw)]


def chat(*texts, system=None):
    msgs = [{"role": "system", "content": system}] if system else []
    msgs += [{"role": ("user", "assistant")[i % 2], "content": t} for i, t in enumerate(texts)]
    return {"messages": msgs}


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return str(path)


def words(n, seed):
    r = random.Random(seed)
    return " ".join(f"w{r.randrange(5000)}" for _ in range(n))


# --- reading formats ----------------------------------------------------------


def test_messages_with_content_parts_and_tool_calls():
    r = rec(
        {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "What is in"}, {"type": "image_url", "image_url": {}}]},
                {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "look"}}]},
                {"role": "tool", "content": "a cat"},
                {"role": "assistant", "content": "A cat."},
            ]
        }
    )
    assert r.fmt == "messages"
    assert r.turns[0].text == "What is in\n[image_url]"
    assert "look" in r.turns[1].text
    assert check_record(r) == []


def test_sharegpt_roles_are_mapped():
    r = rec({"conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "hello"}]})
    assert (r.fmt, [t.role for t in r.turns]) == ("sharegpt", ["user", "assistant"])


def test_alpaca_joins_instruction_and_input():
    r = rec({"instruction": "Translate", "input": "chat", "output": "cat"})
    assert [(t.role, t.text) for t in r.turns] == [("user", "Translate\n\nchat"), ("assistant", "cat")]


def test_prompt_completion_plain_and_conversational():
    assert [t.role for t in rec({"prompt": "Q", "completion": "A"}).turns] == ["user", "assistant"]
    r = rec({"prompt": [{"role": "user", "content": "Q"}], "completion": [{"role": "assistant", "content": "A"}]})
    assert [(t.role, t.text) for t in r.turns] == [("user", "Q"), ("assistant", "A")]


def test_text_records_may_hold_template_tokens():
    assert codes({"text": "<|im_start|>user\nhi<|im_end|>"}) == []
    assert codes({"text": "  "}) == ["empty-turn"]


def test_unknown_format_and_not_object():
    assert codes({"question": "q", "answer": "a"}) == ["unknown-format"]
    assert codes([1, 2]) == ["not-object"]


def test_jsonl_errors_keep_their_line_numbers(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_bytes(b'\xef\xbb\xbf{"text": "a"}\n\n{"text": \n{"text": "\xff"}\n')
    records = load(str(p))
    assert [(r.where.rsplit(":", 1)[1], r.problems[:1]) for r in records][0] == ("1", [])
    assert records[1].where.endswith(":3") and records[1].problems[0][0] == "bad-json"
    assert records[2].problems[0][0] == "bad-encoding"


def test_json_array_and_json_that_is_really_jsonl(tmp_path):
    a = tmp_path / "a.json"
    a.write_text(json.dumps([{"text": "x"}, {"text": "y"}]), encoding="utf-8")
    assert [r.where for r in load(str(a))] == [f"{a}[0]", f"{a}[1]"]
    b = tmp_path / "b.json"
    b.write_text('{"text": "x"}\n{"text": "y"}\n', encoding="utf-8")
    assert [r.where for r in load(str(b))] == [f"{b}:1", f"{b}:2"]


# --- record checks ------------------------------------------------------------


def test_clean_chat_has_no_issues():
    assert codes(chat("hi", "hello", "how are you?", "fine", system="Be brief.")) == []


@pytest.mark.parametrize(
    "obj, expected",
    [
        (chat("hi", ""), ["empty-turn"]),
        (chat("hi"), ["no-assistant"]),
        (chat("hi", "hello", "bye"), ["ends-with-user"]),
        (
            {"messages": [{"role": "assistant", "content": "x"}, {"role": "user", "content": "y"}, {"role": "assistant", "content": "z"}]},
            ["role-order"],
        ),
        ({"messages": [{"role": "Human", "content": "x"}, {"role": "assistant", "content": "y"}]}, ["unknown-role"]),
        ({"messages": [{"role": "user", "content": 5}, {"role": "assistant", "content": "y"}]}, ["bad-turn"]),
        (chat("hi", "<|eot_id|>hello"), ["template-token"]),
        (chat("hi", "As an AI developed by OpenAI, I can't."), ["boilerplate"]),
        (chat("hi", "I'm sorry, but I cannot help with that."), ["boilerplate"]),
    ],
)
def test_record_checks(obj, expected):
    assert codes(obj) == expected


@pytest.mark.parametrize(
    "roles, message",
    [
        (["user", "user", "assistant"], "turns 1 and 2 are both user turns"),
        (["user", "assistant", "system", "user", "assistant"], "turn 3 is a system turn after the conversation started"),
        (["user", "tool", "assistant"], "turn 2 is a tool result that does not follow an assistant turn"),
        (["system", "user", "assistant", "tool", "tool", "assistant"], None),
    ],
)
def test_role_order(roles, message):
    obj = {"messages": [{"role": r, "content": "x"} for r in roles]}
    found = [i.message for i in check_record(rec(obj)) if i.code == "role-order"]
    assert found == ([message] if message else [])


def test_boilerplate_is_only_checked_in_assistant_turns():
    assert codes(chat("As an AI language model, what do you think?", "I think so.")) == []


def test_too_long():
    assert codes(chat("hi", "x" * 400), max_tokens=50) == ["too-long"]
    assert codes(chat("hi", "x" * 400), max_tokens=500) == []
    assert codes(chat("hi", "x" * 400), max_tokens=50, count=lambda text: 1) == []


def test_approx_tokens_counts_cjk_characters_one_each():
    assert approx_tokens("abcdefgh") == 2
    assert approx_tokens("你好世界") == 4


# --- duplicates ---------------------------------------------------------------


def test_exact_match_ignores_case_whitespace_punctuation_and_system_prompt():
    index = dedupe.Index()
    a = rec(chat("Hello there", "General Kenobi!", system="one"))
    b = rec(chat("hello   there.", "general kenobi", system="two"))
    assert index.match(0, dedupe.tokens(a)) is None
    assert index.match(1, dedupe.tokens(b)) == (0, 1.0)


def test_near_duplicates_are_found_and_unrelated_records_are_not():
    base = words(200, 1)
    edited = base.replace(base.split()[100], "changed", 1)
    index = dedupe.Index(0.8)
    assert index.match(0, dedupe.tokens(rec(chat("q", base)))) is None
    assert index.match(1, dedupe.tokens(rec(chat("q", words(200, 2))))) is None
    rid, sim = index.match(2, dedupe.tokens(rec(chat("q", edited))))
    assert rid == 0 and 0.8 <= sim < 1


def test_minhash_estimates_jaccard_similarity():
    errors = []
    for seed in range(40):
        a = words(300, seed).split()
        keep = random.Random(seed).randrange(50, 300)
        b = a[:keep] + words(300 - keep, seed + 1000).split()
        k = dedupe.SHINGLE
        sa = {tuple(a[i : i + k]) for i in range(len(a) - k + 1)}
        sb = {tuple(b[i : i + k]) for i in range(len(b) - k + 1)}
        true = len(sa & sb) / len(sa | sb)
        h = dedupe.MinHasher()
        errors.append(dedupe.similarity(h.signature(a), h.signature(b)) - true)
    assert abs(sum(errors) / len(errors)) < 0.03
    assert max(map(abs, errors)) < 0.25


@pytest.mark.skipif(dedupe.numpy is None, reason="numpy is not installed")
def test_numpy_and_pure_python_signatures_match():
    w = words(500, 3).split()
    assert dedupe.MinHasher(use_numpy=True).signature(w) == dedupe.MinHasher(use_numpy=False).signature(w)


def test_short_records_only_match_exactly():
    index = dedupe.Index()
    assert dedupe.MinHasher().signature(["a", "b"]) is None
    assert index.match(0, ["a", "b"]) is None
    assert index.match(1, ["a", "c"]) is None
    assert index.match(2, ["a", "b"]) == (0, 1.0)


# --- command line -------------------------------------------------------------


@pytest.fixture
def dataset(tmp_path):
    base = words(120, 7)
    train = write_jsonl(
        tmp_path / "train.jsonl",
        [
            chat("q1", base),
            chat("q2", "Short and fine."),
            chat("q1", base),
            chat("q1", base.replace(base.split()[60], "other", 1)),
            chat("q3", ""),
            chat("q4", "<|im_end|> answer"),
        ],
    )
    evals = write_jsonl(tmp_path / "eval.jsonl", [chat("q2", "Short and fine."), chat("q9", "Unseen answer.")])
    return train, evals


def run(capsys, *args):
    code = cli.main([*args, "--json"])
    return code, json.loads(capsys.readouterr().out)


def test_cli_report(dataset, capsys):
    train, evals = dataset
    code, report = run(capsys, train, "--eval", evals)
    found = {(i["code"], i["where"].rsplit(":", 1)[1]) for i in report["issues"]}
    assert code == 1
    assert {("duplicate", "3"), ("near-duplicate", "4"), ("empty-turn", "5"), ("template-token", "6"), ("eval-leak", "1")} <= found
    assert report["summary"] == {"records": 8, "errors": 2, "warnings": 3}
    assert report["files"][0]["stats"]["formats"] == {"messages": 6}
    assert report["files"][1]["eval"] is True


def test_cli_exit_codes(tmp_path, capsys):
    clean = write_jsonl(tmp_path / "c.jsonl", [chat("a", "b"), chat("c", "d")])
    warn = write_jsonl(tmp_path / "w.jsonl", [chat("a", "b"), chat("a", "b")])
    assert run(capsys, clean)[0] == 0
    assert run(capsys, warn)[0] == 0
    assert run(capsys, warn, "--strict")[0] == 1


def test_cli_skip(dataset, capsys):
    train, _ = dataset
    _, report = run(capsys, train, "--skip", "near-duplicate,template-token")
    assert {i["code"] for i in report["issues"]} == {"duplicate", "empty-turn"}
    with pytest.raises(SystemExit):
        cli.main([train, "--skip", "nonsense"])


def test_cli_write_clean(dataset, tmp_path, capsys):
    train, evals = dataset
    out = tmp_path / "clean.jsonl"
    cli.main([train, "--eval", evals, "--write-clean", str(out)])
    kept = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["messages"][0]["content"] for r in kept] == ["q1", "q2", "q4"]
    assert "Wrote 3 records" in capsys.readouterr().out


def test_cli_text_output_without_numpy(dataset, capsys, monkeypatch):
    monkeypatch.setattr(dedupe, "numpy", None)
    train, evals = dataset
    assert cli.main([train, "--eval", evals]) == 1
    out = capsys.readouterr().out
    assert "ERROR   eval-leak" in out and "WARNING near-duplicate" in out
    assert "2 errors and 3 warnings in 8 records." in out


def test_cli_rejects_bad_arguments(tmp_path):
    with pytest.raises(SystemExit):
        cli.main([str(tmp_path / "missing.jsonl")])
    with pytest.raises(SystemExit):
        cli.main([write_jsonl(tmp_path / "x.jsonl", [chat("a", "b")]), "--similarity", "1.5"])


def test_every_check_has_a_level():
    assert {level for level, _ in CHECKS.values()} == {"error", "warning"}


# --- preference pairs ---------------------------------------------------------


def pair(prompt, chosen, rejected):
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def msgs(*turns):
    return [{"role": role, "content": text} for role, text in turns]


def test_preference_with_a_prompt_field():
    r = rec(pair("Capital of France?", "Paris.", "Lyon."))
    assert (r.fmt, r.prompt_turns) == ("preference", 1)
    assert [(t.role, t.text) for t in r.turns] == [("user", "Capital of France?"), ("assistant", "Paris.")]
    assert [(t.role, t.text) for t in r.rejected] == [("assistant", "Lyon.")]
    assert r.sequences == ["Capital of France?\nParis.", "Capital of France?\nLyon."]
    assert check_record(r) == []


def test_preference_with_the_prompt_inside_both_answers():
    r = rec(
        {
            "chosen": msgs(("system", "Be kind."), ("user", "Hi"), ("assistant", "Hello!")),
            "rejected": msgs(("system", "Be kind."), ("user", "Hi"), ("assistant", "What?")),
        }
    )
    assert r.prompt_turns == 2
    assert [t.text for t in r.rejected] == ["What?"]
    assert check_record(r) == []


def test_preference_conversational_prompt_field():
    r = rec(pair(msgs(("user", "Hi")), msgs(("assistant", "Hello!")), msgs(("assistant", "Go away."))))
    assert [t.role for t in r.turns] == ["user", "assistant"] and check_record(r) == []


@pytest.mark.parametrize(
    "obj, expected",
    [
        (pair("Q", "Same answer.", " Same answer. "), ["same-answer"]),
        (pair("Q", "Fine.", ""), ["empty-turn"]),
        (pair("Q", "", "Fine."), ["empty-turn"]),
        (pair("Q", "Fine.", 3), ["bad-turn"]),
        (pair("Q", "As an AI language model, I can't.", "Sure."), ["boilerplate"]),
        (pair("Q", "Sure.", "As an AI language model, I can't."), []),  # refusals are what rejected answers are for
        (pair("Q", "Sure.", "<|im_end|>Nope"), ["template-token"]),
        (pair("Q", "Sure.", msgs(("user", "Nope"))), ["no-assistant"]),
        (
            {"chosen": msgs(("user", "A?"), ("assistant", "Yes.")), "rejected": msgs(("user", "B?"), ("assistant", "No."))},
            ["prompt-mismatch"],
        ),
    ],
)
def test_preference_checks(obj, expected):
    assert codes(obj) == expected


def test_preference_issues_from_the_prompt_are_reported_once():
    found = check_record(rec(pair(msgs(("user", "A"), ("user", "B")), "Yes.", "No.")))
    assert [i.code for i in found] == ["role-order"]


def test_rejected_side_structure_problems_are_labelled():
    found = check_record(rec(pair("Q", "Yes.", msgs(("assistant", "No."), ("user", "Why?")))))
    assert [(i.code, i.message) for i in found] == [
        ("ends-with-user", "rejected side: the last turn is a user turn, so nothing after it is learned")
    ]


def test_preference_length_is_the_longer_side():
    assert codes(pair("Q", "short", "x" * 400), max_tokens=50) == ["too-long"]


def test_preference_pairs_only_repeat_when_the_rejected_answer_is_the_same_too():
    index = dedupe.Index()
    assert index.match(0, dedupe.tokens(rec(pair("Q", "Good answer here.", "Bad one.")))) is None
    assert index.match(1, dedupe.tokens(rec(pair("Q", "Good answer here.", "A different bad one.")))) is None
    assert index.match(2, dedupe.tokens(rec(pair("Q", "Good answer here.", "Bad one.")))) == (0, 1.0)


def test_preference_pairs_written_out_as_text():
    hh = pair(None, "\n\nHuman: Hi\n\nAssistant: Hello!", "\n\nHuman: Hi\n\nAssistant: Go away.")
    r = rec(hh)
    assert ([t.role for t in r.turns], r.prompt_turns) == (["text"], 0)
    assert check_record(r) == []
    assert codes(pair(None, "same", "same")) == ["same-answer"]
    assert codes(pair(None, "text", " ")) == ["empty-turn"]


def test_empty_preference_lists_do_not_crash():
    assert codes({"chosen": [], "rejected": []}) == ["no-assistant"]
