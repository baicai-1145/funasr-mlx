"""Port notes: docs/port-notes/download.py.md"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

#: The 14 published repos (``baicai1145/*`` on both platforms).
CATALOG: dict[str, dict] = {
    "sensevoice-mlx-fp16": {"variants": ["."]},
    "paraformer-mlx-fp16": {"variants": ["."]},
    "whisper-tiny-mlx-fp16": {"variants": ["."]},
    "whisper-large-v3-mlx-fp16": {"variants": ["."]},
    "whisper-large-v3-turbo-mlx-fp16": {"variants": ["."]},
    "ct-punc-mlx-fp16": {"variants": ["."]},
    "emotion2vec-mlx-fp16": {"variants": ["."]},
    "fsmn-vad-mlx-fp16": {"variants": ["."]},
    "campplus-mlx-fp16": {"variants": ["."]},
    "qwen3-asr-0.6b-mlx": {"variants": ["fp16", "w4a16"]},
    "qwen3-asr-1.7b-mlx": {"variants": ["fp16", "w4a16"]},
    "glm-asr-nano-mlx": {"variants": ["fp16", "w4a16"]},
    "moss-transcribe-diarize-mlx": {"variants": ["fp16", "w4a16"]},
    "fun-asr-nano-mlx": {"variants": ["fp16", "w4a16"]},
}

OWNER = "baicai1145"
_SKIP_SUFFIXES = (".lock", ".metadata", ".md5")
_SKIP_NAMES = {".gitattributes", ".ms_upload_cache"}


def _hf_list(repo: str) -> list[str]:
    url = f"https://huggingface.co/api/models/{OWNER}/{repo}"
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.loads(r.read())
    return [s["rfilename"] for s in data.get("siblings", [])]


def _ms_list(repo: str) -> list[str]:
    url = f"https://www.modelscope.cn/api/v1/models/{OWNER}/{repo}/repo/files?Recursive=true"
    req = urllib.request.Request(url, headers={"User-Agent": "funasr-mlx/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read())
    return [f["Path"] for f in data["Data"]["Files"] if f.get("Type") != "tree"]


def _fetch(url: str, dest: Path, *, quiet: bool = False) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "funasr-mlx/0.1"})
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as fh:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if not quiet and total and done % (1 << 28) < (1 << 20):
                    pct = done / total * 100
                    print(f"\r  {dest.name}: {done/1e9:.2f}/{total/1e9:.2f} GB ({pct:.0f}%)",
                          end="", file=sys.stderr, flush=True)
        if not quiet and total:
            print(file=sys.stderr)
        return True
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        if dest.exists():
            dest.unlink()
        print(f"  [{url.split('/')[2]}] {exc.__class__.__name__}: {exc}", file=sys.stderr)
        return False


def _fetch_all(urls: list[str], dest: Path) -> bool:
    for url in urls:
        if _fetch(url, dest, quiet=True):
            return True
    return False


def download(
    repo: str,
    *,
    variant: str | None = None,
    dest: str | Path = ".",
    owner: str = OWNER,
    skip_existing: bool = True,
) -> Path:
    global OWNER
    old_owner, OWNER = OWNER, owner
    try:
        if repo not in CATALOG:
            known = ", ".join(sorted(CATALOG))
            raise ValueError(f"Unknown repo {repo!r}. Known: {known}")
        variants = CATALOG[repo]["variants"]
        if variant is not None and variant not in variants and variant != ".":
            raise ValueError(f"Unknown variant {variant!r} for {repo}; choose from {variants}")

        listing = None
        # ModelScope primary (measured 20.4 MB/s vs HF 5.6 MB/s on the same
        # shard set from this network); HF is the per-file fallback.
        for lister, host in ((_ms_list, "modelscope.cn"), (_hf_list, "huggingface.co")):
            try:
                listing = lister(repo)
                primary = host
                break
            except Exception as exc:  # noqa: BLE001
                print(f"  [{host}] listing failed: {exc}", file=sys.stderr)
        if listing is None:
            raise RuntimeError("could not list the repo on either platform")

        files = [
            path for path in listing
            if not path.endswith(_SKIP_SUFFIXES) and path not in _SKIP_NAMES
        ]
        if variant and variant != ".":
            files = [p for p in files if p == variant or p.startswith(variant + "/") or not p.startswith(("fp16/", "w4a16/"))]
            root_files = [p for p in files if "/" not in p]
            sub_files = [p for p in files if p.startswith(variant + "/")]
            files = root_files + sub_files
        base = Path(dest) / repo
        todo = []
        for path in files:
            target = base / path
            if skip_existing and target.is_file() and target.stat().st_size > 0:
                continue
            urls = (
                [f"https://www.modelscope.cn/models/{owner}/{repo}/resolve/master/{path}",
                 f"https://huggingface.co/{owner}/{repo}/resolve/main/{path}"]
            )
            todo.append((path, target, urls))

        done = 0
        failed = []
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(_fetch_all, urls, target): (path, target) for path, target, urls in todo}
            for fut in as_completed(futures):
                path, target = futures[fut]
                if fut.result():
                    done += 1
                else:
                    failed.append(path)
        if failed:
            raise RuntimeError(f"failed to download {len(failed)} file(s): {failed[:3]}...")
        print(f"downloaded {done} new file(s) -> {base}", file=sys.stderr)
        return base / variant if variant and variant != "." else base
    finally:
        OWNER = old_owner
