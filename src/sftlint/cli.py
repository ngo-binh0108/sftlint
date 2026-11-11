"""sftlint command line."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from . import __version__, hub
from .checks import CHECKS, Issue, approx_tokens, check_record
from .dedupe import Index, tokens
from .formats import Record, load

DEDUPE_CODES = {"duplicate", "near-duplicate", "eval-leak"}


def token_counter(name: str | None):
    """A function counting tokens with a Hugging Face tokenizer, or the rough estimate if no name is given."""
    if not name:
        return approx_tokens, "~"
    try:
        from tokenizers import Tokenizer
    except ImportError:
        raise SystemExit("--tokenizer needs the tokenizers package: pip install tokenizers") from None
    tok = Tokenizer.from_file(name) if Path(name).is_file() else Tokenizer.from_pretrained(name)
    return (lambda text: len(tok.encode(text, add_special_tokens=False).ids)), ""


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def file_stats(records: list[Record], count) -> dict:
    parsed = [r for r in records if r.fmt]
    lengths = [max(count(text) for text in r.sequences) for r in parsed]
    turns = [len(r.turns) for r in parsed if r.turns and r.turns[0].role != "text"]  # text has no turns to count
    systems = Counter(t.text for r in parsed for t in r.turns[:1] if t.role in ("system", "developer"))
    return {
        "records": len(records),
        "formats": dict(Counter(r.fmt for r in parsed)),
        "turns": {"min": min(turns), "mean": round(sum(turns) / len(turns), 1), "max": max(turns)} if turns else None,
        "tokens": {"p50": percentile(lengths, 0.5), "p95": percentile(lengths, 0.95), "max": max(lengths, default=0)},
        "system_prompts": {"records": sum(systems.values()), "distinct": len(systems)},
    }


def find_duplicates(train: list[Record], evals: list[Record], threshold: float, skip: set[str]) -> tuple[list[Issue], set[int]]:
    """Issues for repeated records, and the ids (positions in train + evals) of records that repeat an earlier one."""
    index = Index(threshold if "near-duplicate" not in skip else 1.0)
    everything = train + evals
    issues, repeats = [], set()
    for rid, rec in enumerate(everything):
        if rec.fmt is None:
            continue
        found = index.match(rid, tokens(rec))
        if found is None:
            continue
        original, sim = found
        repeats.add(rid)
        first = everything[original]
        if rid >= len(train) and original < len(train):
            code = "eval-leak"
            what = "the same as" if sim == 1 else f"~{sim:.2f} similar to"
            issues.append(Issue(code, rec.where, f"{what} training record {first.where}"))
        elif sim == 1:
            issues.append(Issue("duplicate", rec.where, f"repeats {first.where}"))
        else:
            issues.append(Issue("near-duplicate", rec.where, f"~{sim:.2f} similar to {first.where}"))
    return [i for i in issues if i.code not in skip], repeats


def render(report: dict, show_all: bool) -> str:
    out = []
    for f in report["files"]:
        s = f["stats"]
        kinds = ", ".join(f"{n:,} {k}" for k, n in sorted(s["formats"].items(), key=lambda kv: -kv[1])) or "none readable"
        out.append(f"{f['path']}{' (eval)' if f['eval'] else ''}: {s['records']:,} records ({kinds})")
        parts = []
        if s["turns"]:
            parts.append(f"turns per record: {s['turns']['min']}-{s['turns']['max']}, mean {s['turns']['mean']}")
        tok = s["tokens"]
        parts.append(f"tokens per record: p50 {report['approx']}{tok['p50']:,}, p95 {tok['p95']:,}, max {tok['max']:,}")
        out.append("  " + "   ".join(parts))
        if s["system_prompts"]["records"]:
            out.append(f"  system prompt in {s['system_prompts']['records']:,} records ({s['system_prompts']['distinct']:,} distinct)")
    out.append("")

    by_code: dict[str, list[dict]] = {}
    for issue in report["issues"]:
        by_code.setdefault(issue["code"], []).append(issue)
    order = sorted(by_code, key=lambda c: (CHECKS[c][0] != "error", list(CHECKS).index(c)))
    for code in order:
        items = by_code[code]
        level, meaning = CHECKS[code]
        out.append(f"{level.upper():7} {code}: {meaning} ({len(items):,})")
        shown = items if show_all else items[:5]
        out += [f"  {i['where']}  {i['message']}" for i in shown]
        if len(items) > len(shown):
            out.append(f"  ... and {len(items) - len(shown):,} more (--all lists them)")
        out.append("")

    t = report["summary"]
    if not report["issues"]:
        out.append(f"No problems found in {t['records']:,} records.")
    else:
        out.append(f"{t['errors']:,} errors and {t['warnings']:,} warnings in {t['records']:,} records.")
    if "written" in t:
        out.append(f"Wrote {t['written']:,} records to {t['clean_path']}, leaving out records with errors and repeats.")
    return "\n".join(out)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="sftlint",
        description="Find broken records, duplicates and eval leaks in a fine-tuning dataset before you train on it.",
    )
    ap.add_argument(
        "files",
        nargs="+",
        metavar="DATA",
        help="training data: .jsonl, .json, .gz or .parquet files, or a Hugging Face dataset as org/name or org/name/path/file",
    )
    ap.add_argument("--eval", action="append", default=[], metavar="DATA", help="eval or test set to check for leaks (repeatable)")
    ap.add_argument("--split", default="train", help="split to read from a Hugging Face dataset (default train)")
    ap.add_argument("--eval-split", default="test", help="split to read from a Hugging Face dataset given to --eval (default test)")
    ap.add_argument("--revision", default="main", help="branch, tag or commit of Hugging Face datasets (default main)")
    ap.add_argument("--max-tokens", type=int, metavar="N", help="warn about records longer than N tokens")
    ap.add_argument("--tokenizer", metavar="NAME", help="count tokens with this Hugging Face tokenizer (repo id or tokenizer.json)")
    ap.add_argument("--similarity", type=float, default=0.8, metavar="X", help="near-duplicate threshold, 0 to 1 (default 0.8)")
    ap.add_argument("--skip", default="", metavar="CODES", help="comma-separated checks to turn off, e.g. boilerplate,near-duplicate")
    ap.add_argument("--write-clean", metavar="OUT", help="write the training records without errors or repeats to OUT as JSONL")
    ap.add_argument("--strict", action="store_true", help="exit with 1 on warnings too, not just errors")
    ap.add_argument("--all", action="store_true", help="list every issue instead of the first five of each kind")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    skip = {c.strip() for c in args.skip.split(",") if c.strip()}
    unknown = skip - set(CHECKS)
    if unknown:
        ap.error(f"unknown check(s) in --skip: {', '.join(sorted(unknown))}. Known: {', '.join(CHECKS)}")
    if not 0 < args.similarity <= 1:
        ap.error("--similarity must be above 0 and at most 1")

    sources = []  # (name shown in the report, local path, is eval)
    for ref, is_eval in [(f, False) for f in args.files] + [(f, True) for f in args.eval]:
        if Path(ref).is_file():
            sources.append((ref, ref, is_eval))
        elif hub.looks_like_repo(ref):
            try:
                found = hub.resolve(ref, args.eval_split if is_eval else args.split, args.revision)
            except hub.HubError as e:
                ap.error(str(e))
            sources += [(name, str(local), is_eval) for name, local in found]
        else:
            ap.error(f"no such file: {ref}")

    count, approx = token_counter(args.tokenizer)
    loaded = [(name, is_eval, load(local, name)) for name, local, is_eval in sources]
    train = [rec for _, is_eval, records in loaded if not is_eval for rec in records]
    evals = [rec for _, is_eval, records in loaded if is_eval for rec in records]

    issues: list[Issue] = []
    bad: set[int] = set()
    for rid, rec in enumerate(train + evals):
        found = [i for i in check_record(rec, args.max_tokens, count) if i.code not in skip]
        issues += found
        if any(i.level == "error" for i in found):
            bad.add(rid)
    repeats: set[int] = set()
    if DEDUPE_CODES - skip:
        dupes, repeats = find_duplicates(train, evals, args.similarity, skip)
        issues += dupes

    files = [{"path": name, "eval": is_eval, "stats": file_stats(records, count)} for name, is_eval, records in loaded]
    errors = sum(i.level == "error" for i in issues)
    report = {
        "approx": approx,
        "files": files,
        "issues": [{"level": i.level, "code": i.code, "where": i.where, "message": i.message} for i in issues],
        "summary": {"records": len(train) + len(evals), "errors": errors, "warnings": len(issues) - errors},
    }

    if args.write_clean:
        kept = [r for rid, r in enumerate(train) if rid not in bad and rid not in repeats]
        with open(args.write_clean, "w", encoding="utf-8", newline="\n") as f:
            for rec in kept:
                f.write(json.dumps(rec.data, ensure_ascii=False) + "\n")
        report["summary"].update(written=len(kept), clean_path=args.write_clean)

    if args.json:
        report["token_count"] = "estimate" if approx else args.tokenizer
        del report["approx"]
        print(json.dumps(report, indent=2))
    else:
        if hasattr(sys.stdout, "reconfigure"):  # quoted dataset text may not fit a Windows console's code page
            sys.stdout.reconfigure(errors="replace")
        print(render(report, args.all))

    if errors or (args.strict and issues):
        return 1
    return 0
