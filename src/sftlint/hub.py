"""Find and download the data files of a Hugging Face dataset repo."""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import __version__

ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
DATA_SUFFIXES = (".jsonl", ".json", ".jsonl.gz", ".json.gz", ".parquet")
_REF = re.compile(r"[A-Za-z0-9][\w.-]*/[\w.-]+(/.+)?")


class HubError(Exception):
    pass


def looks_like_repo(ref: str) -> bool:
    """True for "org/name" or "org/name/path/in/repo", when no local file has that name."""
    return not Path(ref).exists() and bool(_REF.fullmatch(ref))


def _request(url: str):
    headers = {"User-Agent": f"sftlint/{__version__}"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):  # the Hub also answers 401 for repos that do not exist
            raise HubError("not found, or gated or private (set HF_TOKEN to a token that can read it)") from None
        if e.code == 404:
            raise HubError("not found") from None
        raise HubError(f"HTTP {e.code} from {url}") from None
    except urllib.error.URLError as e:
        raise HubError(f"could not reach {ENDPOINT}: {e.reason}") from None


def list_files(repo: str, revision: str) -> list[tuple[str, int]]:
    """(path, size) of every file in a dataset repo."""
    url = f"{ENDPOINT}/api/datasets/{repo}/tree/{urllib.parse.quote(revision, safe='')}?recursive=true"
    files = []
    while url:
        with _request(url) as r:
            files += [(f["path"], f.get("size", 0)) for f in json.load(r) if f.get("type") == "file"]
            link = r.headers.get("Link") or ""
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else None
    return files


def pick_files(files: list[tuple[str, int]], split: str) -> list[tuple[str, int]]:
    """The data files of one split, found by name: train.jsonl, data/train-00000-of-00002.parquet, train/part1.json ..."""
    data = [(p, s) for p, s in files if p.lower().endswith(DATA_SUFFIXES)]
    word = re.compile(rf"(^|[/_.-]){re.escape(split)}([/_.-]|$)", re.IGNORECASE)
    chosen = [(p, s) for p, s in data if word.search(p)]
    if not chosen and len(data) == 1:
        chosen = data  # a single data file is the whole dataset
    if not chosen:
        names = ", ".join(p for p, _ in data[:10]) or "none"
        raise HubError(f"no data file for split {split!r} (data files: {names}); name one file as org/name/path")
    # Several files for one split usually share one format; mixing parquet and json copies would double-count.
    parquet = [(p, s) for p, s in chosen if p.endswith(".parquet")]
    other = [(p, s) for p, s in chosen if not p.endswith(".parquet")]
    return other or parquet


def cache_dir() -> Path:
    if os.environ.get("SFTLINT_CACHE"):
        return Path(os.environ["SFTLINT_CACHE"])
    base = os.environ.get("XDG_CACHE_HOME") or os.environ.get("LOCALAPPDATA") or Path.home() / ".cache"
    return Path(base) / "sftlint"


def download(repo: str, path: str, size: int, revision: str) -> Path:
    """The local copy of one file, downloaded unless the cache already holds a copy of the same size."""
    dest = cache_dir() / repo / revision / path
    if dest.is_file() and (size == 0 or dest.stat().st_size == size):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {repo}/{path} ({size / 2**20:,.1f} MiB)", file=sys.stderr)
    url = f"{ENDPOINT}/datasets/{repo}/resolve/{urllib.parse.quote(revision, safe='')}/{urllib.parse.quote(path)}"
    part = dest.with_name(dest.name + ".part")
    with _request(url) as r, open(part, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    part.replace(dest)
    return dest


def resolve(ref: str, split: str, revision: str) -> list[tuple[str, Path]]:
    """(name to show, local path) for each data file a reference points at."""
    parts = ref.split("/")
    repo, inner = "/".join(parts[:2]), "/".join(parts[2:])
    try:
        files = list_files(repo, revision)
    except HubError as e:
        raise HubError(f"{repo}: {e}") from None
    if inner:
        chosen = [(p, s) for p, s in files if p == inner]
        if not chosen:
            raise HubError(f"{repo} has no file {inner!r}")
    else:
        try:
            chosen = pick_files(files, split)
        except HubError as e:
            raise HubError(f"{repo}: {e}") from None
    return [(f"{repo}/{p}", download(repo, p, s, revision)) for p, s in chosen]
