"""Prepare bounded real-audio inputs with source hashes and corpus coverage."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import zipfile

MAX_MEMBER = 32_000_000
MAX_EXTRACTED = 4_000_000_000


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1_048_576):
            value.update(chunk)
    return value.hexdigest()


def safe_target(root, name):
    if "\\" in name:
        raise ValueError("Archive member contains a platform-dependent path")
    target = (root / name).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Archive path escapes extraction root")
    return target


def copy_bounded(source, destination, expected):
    copied = 0
    while chunk := source.read(min(1_048_576, expected - copied + 1)):
        if copied + len(chunk) > expected:
            raise ValueError("Archive member exceeds its declared byte count")
        destination.write(chunk)
        copied += len(chunk)
    if copied != expected:
        raise ValueError("Archive member is shorter than its declared byte count")
    return copied


def extract_archive(source, target):
    target.mkdir(parents=True, exist_ok=True)
    total = 0
    if source.suffix == ".zip":
        with zipfile.ZipFile(source) as archive:
            for member in archive.infolist():
                path = safe_target(target, member.filename)
                if member.is_dir():
                    continue
                if (
                    member.file_size > MAX_MEMBER
                    or (member.external_attr >> 16) & 0o170000 == 0o120000
                ):
                    raise ValueError("Unsupported archive member")
                total += member.file_size
                if total > MAX_EXTRACTED:
                    raise ValueError("Archive exceeds extracted-byte bound")
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as input_file, path.open("wb") as output_file:
                    copy_bounded(input_file, output_file, member.file_size)
    else:
        with tarfile.open(source, "r|gz") as archive:
            for member in archive:
                path = safe_target(target, member.name)
                if member.isdir():
                    continue
                if not member.isfile() or member.size > MAX_MEMBER:
                    raise ValueError("Unsupported archive member")
                total += member.size
                if total > MAX_EXTRACTED:
                    raise ValueError("Archive exceeds extracted-byte bound")
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as input_file, path.open("wb") as output_file:
                    copy_bounded(input_file, output_file, member.size)
    return total


def record(path, *, key, corpus, transcript="", speaker="", emotion=None, language=""):
    import soundfile as sf

    info = sf.info(path)
    if info.channels != 1 or info.samplerate != 16000:
        raise ValueError(f"Prepared audio is not mono 16 kHz: {path}")
    return {
        "key": key,
        "audio": str(path.resolve()),
        "corpus": corpus,
        "language": language,
        "duration_seconds": info.frames / info.samplerate,
        "sample_rate": info.samplerate,
        "transcript": transcript,
        "speaker_id": speaker,
        "emotion_label": emotion,
        "sha256": digest(path),
    }


def prepare_librispeech(root, output):
    source = root / "datasets/librispeech-test-clean/test-clean.tar.gz"
    destination = output / "librispeech-test-clean"
    extracted = extract_archive(source, destination)
    transcripts = {}
    for path in destination.rglob("*.trans.txt"):
        for line in path.read_text().splitlines():
            key, _, text = line.partition(" ")
            transcripts[key] = text
    entries = [
        record(
            path,
            key="libri_" + path.stem,
            corpus="librispeech-test-clean",
            language="en",
            transcript=transcripts[path.stem],
            speaker=path.stem.split("-")[0],
        )
        for path in sorted(destination.rglob("*.flac"))
    ]
    return entries, {
        "count": len(entries),
        "archive_sha256": digest(source),
        "extracted_bytes": extracted,
        "license": "CC BY 4.0",
    }


def prepare_aishell(root, output):
    import pyarrow.parquet as pq
    import soundfile as sf

    entries = []
    destination = output / "aishell-test"
    destination.mkdir(parents=True, exist_ok=True)
    shards = sorted((root / "datasets/aishell-test/data").glob("*.parquet"))
    if len(shards) != 3:
        raise ValueError("AISHELL-derived test mirror must contain three shards")
    for shard in shards:
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(batch_size=1, columns=["context", "answer"]):
            row = batch.to_pylist()[0]
            encoded = row["context"]["bytes"]
            if not encoded or len(encoded) > MAX_MEMBER:
                raise ValueError("AISHELL audio record exceeds byte bound")
            info = sf.info(io.BytesIO(encoded))
            if info.samplerate != 16000 or info.channels != 1:
                raise ValueError("Unexpected AISHELL audio format")
            key = f"aishell_{len(entries):05d}"
            path = destination / (key + ".wav")
            path.write_bytes(encoded)
            entries.append(
                record(
                    path,
                    key=key,
                    corpus="aishell-derived-test-mirror",
                    transcript=row["answer"],
                    language="zh",
                )
            )
    return entries, {
        "count": len(entries),
        "expected_mirror_count": 6920,
        "canonical_split_count": 7176,
        "is_canonical_complete_split": False,
        "license": "not-declared-by-selected-mirror",
        "license_note": "Redistribution and commercial use have not been verified",
        "shards": [{"path": str(p), "sha256": digest(p)} for p in shards],
    }


def mono_16k(samples, rate):
    import numpy as np
    from scipy.signal import resample_poly

    if samples.ndim != 2 or not 1 <= samples.shape[1] <= 8:
        raise ValueError("Waveform must have between one and eight channels")
    if samples.shape[0] > 10 * rate or not np.isfinite(samples).all():
        raise ValueError("Waveform exceeds its bound or contains non-finite samples")
    mono = samples.mean(axis=1, dtype=np.float32)
    if rate == 48000:
        return resample_poly(mono, 1, 3).astype(np.float32)
    if rate != 16000:
        raise ValueError("Unexpected RAVDESS sample rate")
    return mono


def prepare_ravdess(root, output):
    import soundfile as sf

    source = root / "datasets/ravdess-speech/Audio_Speech_Actors_01-24.zip"
    originals = output / "ravdess-original"
    extracted = extract_archive(source, originals)
    destination = output / "ravdess-16k"
    destination.mkdir(parents=True, exist_ok=True)
    emotions = {
        1: "neutral",
        2: "calm",
        3: "happy",
        4: "sad",
        5: "angry",
        6: "fearful",
        7: "disgust",
        8: "surprised",
    }
    entries = []
    for path in sorted(originals.rglob("*.wav")):
        fields = path.stem.split("-")
        if len(fields) != 7 or fields[:2] != ["03", "01"]:
            raise ValueError("Unexpected RAVDESS speech filename")
        samples, rate = sf.read(path, dtype="float32", always_2d=True)
        channels = samples.shape[1]
        samples = mono_16k(samples, rate)
        prepared = destination / path.name
        sf.write(prepared, samples, 16000, subtype="FLOAT")
        entry = record(
            prepared,
            key="ravdess_" + path.stem,
            corpus="ravdess-speech",
            language="en",
            speaker=fields[-1],
            emotion=emotions[int(fields[2])],
        )
        entry.update(
            {
                "source_sha256": digest(path),
                "source_sample_rate": rate,
                "source_channels": channels,
                "channel_handling": "mean(axis=1,dtype=float32)",
                "resampling": "scipy.signal.resample_poly(up=1,down=3)"
                if rate == 48000
                else "none",
            }
        )
        entries.append(entry)
    return entries, {
        "count": len(entries),
        "archive_sha256": digest(source),
        "extracted_bytes": extracted,
        "license": "CC BY-NC-SA 4.0",
        "note": "calm has no dedicated emotion2vec label; numerical parity is separate from accuracy",
    }


def select(entries, count):
    eligible = [e for e in entries if 1.0 <= e["duration_seconds"] <= 8.0]
    selected, used = [], set()
    for entry in eligible:
        group = entry.get("emotion_label") or entry.get("speaker_id") or entry["key"]
        if group in used:
            continue
        used.add(group)
        selected.append(entry)
        if len(selected) == count:
            return selected
    selected.extend(e for e in eligible if e not in selected)
    return selected[:count]


def write_json(path, document):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(document, ensure_ascii=True, indent=2))
    temporary.replace(path)


def source_coverage(root, name):
    packages = {
        "english": "librispeech-test-clean",
        "chinese": "aishell-test",
        "emotion": "ravdess-speech",
    }
    licenses = {
        "english": "CC BY 4.0",
        "chinese": "not-declared-by-selected-mirror",
        "emotion": "CC BY-NC-SA 4.0",
    }
    result = {"license": licenses[name]}
    if name == "chinese":
        result.update(
            expected_mirror_count=6920,
            canonical_split_count=7176,
            is_canonical_complete_split=False,
            license_note="The selected mirror has no license declaration; "
            "redistribution and commercial use remain unverified",
        )
    catalog_path = root / "catalog.json"
    if catalog_path.is_file():
        package = json.loads(catalog_path.read_text())["datasets"][packages[name]]
        for field in ("repository", "revision", "doi", "record"):
            if package.get(field):
                result[field] = package[field]
    records_path = root / "downloaded.json"
    if records_path.is_file():
        records = json.loads(records_path.read_text())
        folder = root / "datasets" / packages[name]
        sources = [
            {
                "path": row["local"],
                "sha256": row["actual_sha256"],
                "bytes": row["actual_bytes"],
                "url": row["url"],
            }
            for row in records
            if Path(row["local"]).is_relative_to(folder)
        ]
        result["source_files"] = sources
        if name == "chinese":
            result["shards"] = [row for row in sources if row["path"].endswith(".parquet")]
        elif sources:
            result["archive_sha256"] = sources[0]["sha256"]
    if name == "english":
        folder = root / "prepared/librispeech-test-clean"
        if folder.is_dir():
            result["on_disk_extracted_bytes"] = sum(
                p.stat().st_size for p in folder.rglob("*") if p.is_file()
            )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument(
        "--corpus", choices=("all", "english", "chinese", "emotion", "selection"), default="all"
    )
    arguments = parser.parse_args()
    if not 1 <= arguments.samples <= 128:
        raise ValueError("Selected sample count must be between 1 and 128")
    output = arguments.root / "prepared"
    output.mkdir(parents=True, exist_ok=True)
    coverage_path = output / "coverage.json"
    coverage = json.loads(coverage_path.read_text()) if coverage_path.is_file() else {}
    functions = {
        "english": prepare_librispeech,
        "chinese": prepare_aishell,
        "emotion": prepare_ravdess,
    }
    for name, function in functions.items():
        metadata = {**coverage.get(name, {}), **source_coverage(arguments.root, name)}
        corpus = output / f"corpus-{name}.json"
        if arguments.corpus not in ("all", name):
            if not corpus.is_file():
                if arguments.corpus == "selection":
                    raise FileNotFoundError(f"Corpus has not been prepared: {corpus}")
                continue
            entries = json.loads(corpus.read_text())
            metadata.update(count=len(entries), previously_prepared=True)
        else:
            entries, prepared_metadata = function(arguments.root, output)
            metadata.update(prepared_metadata)
            write_json(corpus, entries)
        if not metadata.get("license"):
            raise ValueError(f"Corpus {name} has no license status")
        for entry in entries:
            if not entry.get("language"):
                entry["language"] = "zh" if name == "chinese" else "en"
        selected = select(entries, arguments.samples)
        if not selected:
            raise ValueError(f"Corpus {name} has no eligible diagnostic samples")
        metadata["selected_count"] = len(selected)
        coverage[name] = metadata
        write_json(output / f"manifest-{name}.json", {"samples": selected})
        write_json(coverage_path, coverage)
        print(json.dumps({"corpus": name, **metadata}), flush=True)


if __name__ == "__main__":
    main()
