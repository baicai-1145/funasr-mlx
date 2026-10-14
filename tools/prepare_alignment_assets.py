"""Acquire pinned public evaluation assets; no ML framework imports."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request

MODELS = {
    "sensevoice": "FunAudioLLM/SenseVoiceSmall",
    "paraformer": "funasr/paraformer-zh",
    "ct-punc": "funasr/ct-punc",
    "fsmn-vad": "funasr/fsmn-vad",
    "campplus": "funasr/campplus",
    "emotion2vec": "emotion2vec/emotion2vec_plus_large",
}
ALLOWED = {
    "model.pt",
    "pytorch_model.bin",
    "campplus_cn_common.bin",
    "config.yaml",
    "configuration.json",
    "config.json",
    "tokens.json",
    "tokens.txt",
    "chn_jpn_yue_eng_ko_spectok.bpe.model",
    "am.mvn",
    "seg_dict",
    "README.md",
    "LICENSE",
    "model.py",
    "vocab.txt",
}
USER_AGENT = "funasr-mlx-official-alignment/1.0"
LIMIT = 20_000_000_000


def response(url, timeout=30):
    return urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=timeout
    )


def metadata(url):
    with response(url) as stream:
        data = stream.read(5_000_001)
    if len(data) > 5_000_000:
        raise ValueError("Metadata exceeds the 5 MB bound")
    return json.loads(data)


def hub_tree(url):
    entries = []
    for _ in range(12):
        with response(url) as stream:
            data = stream.read(5_000_001)
            links = urllib.request.parse_http_list(stream.headers.get("Link", ""))
        if len(data) > 5_000_000:
            raise ValueError("Hub directory page exceeds metadata bound")
        entries.extend(json.loads(data))
        next_url = None
        for link in links:
            target, _, parameters = link.partition(";")
            if any(
                parameter.strip().removeprefix("rel=").strip('"') == "next"
                for parameter in parameters.split(";")
            ):
                next_url = target.strip().removeprefix("<").removesuffix(">")
        if next_url is None:
            return entries
        if urllib.parse.urlparse(next_url).netloc != "huggingface.co":
            raise ValueError("Unexpected Hub pagination host")
        url = next_url
    raise ValueError("Hub directory pagination exceeds bound")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n")


def hub_model(family, repository):
    detail = metadata(f"https://huggingface.co/api/models/{repository}")
    revision = detail["sha"]
    listing = hub_tree(
        f"https://huggingface.co/api/models/{repository}/tree/{revision}?recursive=true"
    )
    files = []
    for item in listing:
        name = item["path"]
        if item["type"] != "file" or name not in ALLOWED:
            continue
        files.append(
            {
                "path": name,
                "bytes": item["size"],
                "sha256": item.get("lfs", {}).get("oid"),
                "git_oid": item["oid"] if not item.get("lfs") else None,
                "url": f"https://huggingface.co/{repository}/resolve/{revision}/{name}",
            }
        )
    return {
        "family": family,
        "repository": repository,
        "revision": revision,
        "card": detail.get("cardData", {}),
        "files": files,
    }


def catalog():
    result = {"models": {}, "datasets": {}, "errors": {}, "created": time.time()}
    for family, repository in MODELS.items():
        try:
            result["models"][family] = hub_model(family, repository)
            print(
                json.dumps({"model": family, "files": result["models"][family]["files"]}),
                flush=True,
            )
        except (urllib.error.URLError, ValueError, KeyError) as error:
            result["errors"][family] = str(error)
    try:
        source = "https://raw.githubusercontent.com/openai/whisper/v20250625/whisper/__init__.py"
        with response(source) as stream:
            tree = ast.parse(stream.read(300_000).decode())
        value = next(
            node.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "_MODELS" for target in node.targets
            )
        )
        models = ast.literal_eval(value)
        url_tiny = models["tiny"]
        result["models"]["whisper"] = {
            "family": "whisper",
            "variant": "tiny",
            "source": source,
            "files": [{"path": "tiny.pt", "url": url_tiny, "sha256": url_tiny.split("/")[-2]}],
        }
        for variant, name in (
            ("whisper-large-v3", "large-v3"),
            ("whisper-large-v3-turbo", "large-v3-turbo"),
        ):
            url = models[name]
            result["models"][variant] = {
                "family": "whisper",
                "variant": name,
                "source": source,
                "files": [{"path": f"{name}.pt", "url": url, "sha256": url.split("/")[-2]}],
            }
    except (urllib.error.URLError, ValueError, KeyError, StopIteration) as error:
        result["errors"]["whisper"] = str(error)
    result["datasets"]["librispeech-test-clean"] = {
        "repository": "OpenSLR/12",
        "license": "CC BY 4.0",
        "files": [
            {
                "path": "test-clean.tar.gz",
                "url": "https://www.openslr.org/resources/12/test-clean.tar.gz",
                "bytes": 346_663_984,
            }
        ],
    }
    try:
        repository = "AudioLLMs/aishell_1_zh_test"
        detail = metadata(f"https://huggingface.co/api/datasets/{repository}")
        revision = detail["sha"]
        listing = hub_tree(
            f"https://huggingface.co/api/datasets/{repository}/tree/{revision}?recursive=true"
        )
        files = [
            {
                "path": entry["path"],
                "bytes": entry["size"],
                "sha256": entry.get("lfs", {}).get("oid"),
                "git_oid": entry["oid"] if not entry.get("lfs") else None,
                "url": f"https://huggingface.co/datasets/{repository}/resolve/{revision}/{entry['path']}",
            }
            for entry in listing
            if entry["type"] == "file"
            and (
                ("test" in entry["path"].lower() and entry["path"].endswith(".parquet"))
                or (
                    entry["path"].startswith("data_aishell/wav/S")
                    and entry["path"].endswith(".tar.gz")
                    and int(Path(entry["path"]).name[1:5]) >= 764
                )
                or entry["path"]
                in {"README.md", "LICENSE", "data_aishell/transcript/aishell_transcript_v0.8.txt"}
            )
        ]
        result["datasets"]["aishell-test"] = {
            "repository": repository,
            "revision": revision,
            "card": detail.get("cardData", {}),
            "files": files,
            "selection_note": "Public AISHELL-1-derived test mirror, 6920 records, not the complete canonical 7176-record split",
        }
    except (urllib.error.URLError, ValueError, KeyError) as error:
        result["errors"]["aishell-test"] = str(error)
    try:
        record = metadata("https://zenodo.org/api/records/1188976")
        files = [
            {
                "path": entry["key"],
                "bytes": entry["size"],
                "checksum": entry["checksum"],
                "url": entry["links"]["self"],
            }
            for entry in record["files"]
            if entry["key"] == "Audio_Speech_Actors_01-24.zip"
        ]
        result["datasets"]["ravdess-speech"] = {
            "record": "1188976",
            "doi": record["doi"],
            "license": record["metadata"]["license"],
            "files": files,
        }
    except (urllib.error.URLError, ValueError, KeyError) as error:
        result["errors"]["ravdess-speech"] = str(error)
    return result


def download(item, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = item.get("bytes")
    if destination.exists() and (expected is None or destination.stat().st_size == expected):
        print(f"existing {destination}", flush=True)
    else:
        partial = destination.with_suffix(destination.suffix + ".partial")
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": USER_AGENT}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(item["url"], headers=headers)
        with urllib.request.urlopen(request, timeout=60) as stream:
            append = offset and stream.status == 206
            if not append:
                offset = 0
            total = offset + int(stream.headers.get("Content-Length", "0"))
            if total > LIMIT or (expected and total and total != expected):
                raise ValueError(f"Unexpected download size for {destination}: {total}")
            if shutil.disk_usage(destination.parent).free < max(total - offset, 0) + 5_000_000_000:
                raise ValueError("Insufficient disk headroom")
            with partial.open("ab" if append else "wb") as output:
                written = offset
                while chunk := stream.read(1_048_576):
                    output.write(chunk)
                    written += len(chunk)
                    if written > LIMIT:
                        raise ValueError("Download exceeded the per-file bound")
            if expected is not None and written != expected:
                raise ValueError(f"Incomplete download: {written} != {expected}")
        os.replace(partial, destination)
    digest = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    git_digest = hashlib.sha1(usedforsecurity=False)
    git_digest.update(f"blob {destination.stat().st_size}\0".encode())
    with destination.open("rb") as stream:
        while chunk := stream.read(1_048_576):
            digest.update(chunk)
            md5.update(chunk)
            git_digest.update(chunk)
    actual = digest.hexdigest()
    if item.get("sha256") and actual != item["sha256"]:
        raise ValueError(f"SHA256 mismatch: {destination}")
    if item.get("git_oid") and git_digest.hexdigest() != item["git_oid"]:
        raise ValueError(f"Git blob integrity mismatch: {destination}")
    if item.get("checksum", "").startswith("md5:"):
        if md5.hexdigest() != item["checksum"].partition(":")[2]:
            raise ValueError(f"MD5 integrity mismatch: {destination}")
    print(
        json.dumps(
            {"downloaded": str(destination), "bytes": destination.stat().st_size, "sha256": actual}
        ),
        flush=True,
    )
    return {
        **item,
        "local": str(destination),
        "actual_sha256": actual,
        "actual_bytes": destination.stat().st_size,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scan-only", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--from-catalog", type=Path)
    parser.add_argument("--group", choices=["all", "models", "datasets"], default="all")
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    catalog_path = args.root / "catalog.json"
    previous = json.loads(catalog_path.read_text()) if catalog_path.is_file() else None
    if args.from_catalog:
        assets = json.loads(args.from_catalog.read_text())
    else:
        assets = previous if previous is not None else catalog()
    if previous is not None and previous != assets:
        raise ValueError("Pinned campaign catalog changed; use a new campaign root")
    write_json(catalog_path, assets)
    if args.scan_only:
        print(json.dumps({"errors": assets["errors"], "root": str(args.root)}, indent=2))
        return
    record_path = args.root / "downloaded.json"
    existing = json.loads(record_path.read_text()) if record_path.is_file() else []
    records = {record["local"]: record for record in existing}
    completed = 0
    for group in ("datasets", "models"):
        if args.group not in {"all", group}:
            continue
        for name, package in assets[group].items():
            for item in package["files"]:
                if args.metadata_only and item.get("bytes", LIMIT) > 20_000_000:
                    continue
                try:
                    record = download(item, args.root / group / name / item["path"])
                    prior = records.get(record["local"])
                    if prior and (prior["url"] != record["url"] or
                                  prior["actual_sha256"] != record["actual_sha256"]):
                        raise ValueError("Pinned asset identity changed; use a new campaign root")
                    records[record["local"]] = record
                    completed += 1
                    write_json(record_path, list(records.values()))
                except (urllib.error.URLError, ValueError, OSError) as error:
                    print(json.dumps({"failed": item["url"], "error": str(error)}), flush=True)
                    raise
    print(json.dumps({"completed_files": completed, "recorded_files": len(records),
                      "root": str(args.root)}))


if __name__ == "__main__":
    main()
