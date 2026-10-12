"""Static and unit tests for the GPTQ calibration-manifest builder.

The manifest is the input to every GPTQ run, so its contract has to be checked
without the (large, external) audio being present: these tests exercise the
selection logic -- language balance, the round-robin speaker spread, the
calibration/acceptance disjointness assertion -- on synthetic candidate pools,
and check the module's declared constants against the task brief.

No network, no FLEURS download, no checkpoint.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tools import build_calib_manifest as manifest


# --------------------------------------------------------------------------- #
# declared contract
# --------------------------------------------------------------------------- #
def test_the_ten_languages_are_the_briefs_list():
    assert manifest.LANGUAGES == ("zh", "en", "ja", "ko", "es", "fr", "de", "ru", "th", "vi")


def test_prompt_language_names_match_the_qwen3_guide_strings():
    assert manifest.PROMPT_LANGUAGE == {
        "zh": "Chinese",
        "en": "English",
        "ja": "Japanese",
        "ko": "Korean",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "ru": "Russian",
        "th": "Thai",
        "vi": "Vietnamese",
    }


def test_every_language_has_a_source_and_a_fleurs_config_or_local_corpus():
    for language in manifest.LANGUAGES:
        assert language in manifest.SOURCES
    # The eight downloaded languages map to FLEURS configs; zh/en do not.
    assert set(manifest.FLEURS_LANGUAGES) == set(manifest.LANGUAGES) - {"zh", "en"}


def test_the_audio_contract_is_16k_mono_bounded():
    assert manifest.SAMPLE_RATE == 16000
    assert manifest.MIN_SECONDS == 3.0
    assert manifest.MAX_SECONDS == 15.0
    assert manifest.MIN_SECONDS < manifest.MAX_SECONDS


def test_the_split_sizes_give_48_calibration_and_8_acceptance_per_language():
    assert manifest.N_CALIB == 48
    assert manifest.N_ACCEPT == 8
    assert manifest.N_SPARE == 4


def test_the_hessian_language_guard_is_satisfied_by_construction():
    """Equal per-language calibration counts keep every language at 10%."""
    languages = len(manifest.LANGUAGES)
    share = manifest.N_CALIB / (manifest.N_CALIB * languages)
    assert share <= 0.3, "a language would exceed the 30% Hessian-mass guard"


# --------------------------------------------------------------------------- #
# round-robin speaker spread
# --------------------------------------------------------------------------- #
def test_round_robin_spreads_across_speakers():
    """A talker-heavy pool must not fill the selection from a few voices."""
    candidates = [{"speaker": f"s{i % 5}", "id": i} for i in range(50)]
    picked = manifest.round_robin(candidates, 10)
    assert len(picked) == 10
    speakers = [entry["speaker"] for entry in picked]
    assert len(set(speakers)) == 5
    # Perfectly even: two per speaker.
    assert all(speakers.count(name) == 2 for name in set(speakers))


def test_round_robin_is_deterministic():
    candidates = [{"speaker": f"s{i % 7}", "id": i} for i in range(40)]
    first = manifest.round_robin(candidates, 12)
    second = manifest.round_robin(candidates, 12)
    assert [entry["id"] for entry in first] == [entry["id"] for entry in second]


def test_round_robin_returns_everything_when_the_pool_is_small():
    candidates = [{"speaker": "a", "id": 0}, {"speaker": "b", "id": 1}]
    picked = manifest.round_robin(candidates, 10)
    assert len(picked) == 2


def test_round_robin_prefers_one_per_speaker_before_repeating():
    candidates = [{"speaker": "a", "id": i} for i in range(20)]
    candidates += [{"speaker": "b", "id": 100}]
    picked = manifest.round_robin(candidates, 4)
    assert {entry["speaker"] for entry in picked} == {"a", "b"}
    assert [entry["id"] for entry in picked][:2] == [0, 100]


# --------------------------------------------------------------------------- #
# eligibility
# --------------------------------------------------------------------------- #
def _write_wav(path, seconds: float, rate: int = 16000, channels: int = 1):
    import soundfile as sf

    data = np.zeros((int(seconds * rate), channels), dtype=np.float32)
    sf.write(str(path), data, rate)
    return path


def test_eligible_accepts_a_16k_mono_utterance_in_range(tmp_path):
    path = _write_wav(tmp_path / "ok.wav", 5.0)
    info = manifest.eligible(path)
    assert info is not None
    assert info[1] == 16000 and info[2] == 1


@pytest.mark.parametrize(
    "seconds,rate,channels",
    [
        (1.0, 16000, 1),  # too short
        (20.0, 16000, 1),  # too long
        (5.0, 8000, 1),  # wrong rate
        (5.0, 16000, 2),  # not mono
    ],
)
def test_eligible_rejects_out_of_contract_audio(tmp_path, seconds, rate, channels):
    path = _write_wav(tmp_path / "bad.wav", seconds, rate=rate, channels=channels)
    assert manifest.eligible(path) is None


def test_eligible_rejects_a_file_that_is_not_audio(tmp_path):
    path = tmp_path / "not-audio.wav"
    path.write_bytes(b"this is not a wav file")
    assert manifest.eligible(path) is None
    assert manifest.audio_info(path) is None


def test_eligible_rejects_a_missing_file(tmp_path):
    assert manifest.eligible(tmp_path / "absent.wav") is None


# --------------------------------------------------------------------------- #
# local corpus filtering
# --------------------------------------------------------------------------- #
def test_local_candidates_filters_and_assigns_stride_speakers(tmp_path):
    keep = _write_wav(tmp_path / "keep.wav", 5.0)
    short = _write_wav(tmp_path / "short.wav", 1.0)
    corpus = [
        {"audio": str(keep), "transcript": "a", "speaker_id": "S1"},
        {"audio": str(short), "transcript": "b", "speaker_id": "S1"},
        {"audio": str(keep), "transcript": "c", "speaker_id": ""},
    ]
    rows = manifest.local_candidates(corpus, "zh")
    assert len(rows) == 2
    assert rows[0]["speaker"] == "S1"
    # A missing speaker id becomes a deterministic stride bucket.
    assert rows[1]["speaker"].startswith("stride-")
    assert all(row["lang"] == "zh" for row in rows)


def test_english_speaker_reads_the_librispeech_path():
    sample = {"audio": "/x/prepared/librispeech-test-clean/LibriSpeech/test-clean/1089/134686/f.flac"}
    assert manifest.english_speaker(sample) == "1089/134686"
    assert manifest.english_speaker({"audio": "short.flac"}) == ""


# --------------------------------------------------------------------------- #
# the manifest writer end to end, on synthetic corpora
# --------------------------------------------------------------------------- #
def test_main_writes_a_balanced_disjoint_manifest(tmp_path, monkeypatch):
    """Drive ``main`` with synthetic pools and check the written manifest."""
    alignment = tmp_path / "alignment"
    prepared = alignment / "prepared"
    prepared.mkdir(parents=True)
    root = tmp_path / "calib"
    root.mkdir()

    # Chinese and English local corpora: 20 utterances each, 2 speakers.
    for name, language in (("corpus-chinese.json", "zh"), ("corpus-english.json", "en")):
        entries = []
        for index in range(20):
            path = _write_wav(root / f"{language}-{index}.wav", 5.0)
            entries.append(
                {
                    "audio": str(path),
                    "transcript": f"{language} text {index}",
                    "speaker_id": f"S{index % 2}",
                }
            )
        (prepared / name).write_text(json.dumps(entries))

    # FLEURS: one tsv + audio dir per (language, split).
    for language, config in manifest.FLEURS_LANGUAGES.items():
        for split, directory in (("dev", "fleurs"), ("test", "fleurs-test")):
            audio_dir = root / directory / config
            audio_dir.mkdir(parents=True)
            rows = []
            for index in range(20):
                name = f"{split}-{index}.wav"
                _write_wav(audio_dir / name, 5.0)
                rows.append(f"client{index % 4}\t{name}\ttranscript {index}\textra\tmore\tMALE")
            (root / "raw").mkdir(exist_ok=True)
            (root / "raw" / f"fleurs-{config}-{split}.tsv").write_text("\n".join(rows) + "\n")

    monkeypatch.setattr(
        "sys.argv",
        [
            "build_calib_manifest.py",
            "--root",
            str(root),
            "--alignment-root",
            str(alignment),
            "--calib",
            "8",
            "--accept",
            "4",
            "--spare",
            "2",
        ],
    )
    assert manifest.main() == 0
    written = json.loads((root / "calib-manifest.json").read_text())

    assert written["languages"] == list(manifest.LANGUAGES)
    assert written["totals"]["calibration"] == 8 * len(manifest.LANGUAGES)
    assert written["totals"]["acceptance"] == 4 * len(manifest.LANGUAGES)
    assert written["totals"]["samples"] == (8 + 4 + 2) * len(manifest.LANGUAGES)

    samples = written["samples"]
    calibration = {s["audio_path"] for s in samples if s["split"] == "calibration"}
    acceptance = {s["audio_path"] for s in samples if s["split"] == "acceptance"}
    assert not (calibration & acceptance)
    assert len({s["audio_path"] for s in samples}) == len(samples)

    for sample in samples:
        assert sample["sample_rate"] == 16000
        assert sample["sha256"]
        assert sample["prompt_language"] == manifest.PROMPT_LANGUAGE[sample["lang"]]
        assert len(sample["sha256"]) == 64

    # The eight FLEURS languages must draw acceptance from the test split and
    # calibration from dev, which is what makes them disjoint by construction.
    for sample in samples:
        if sample["lang"] in manifest.FLEURS_LANGUAGES:
            if sample["split"] == "acceptance":
                assert "/fleurs-test/" in sample["audio_path"]
            elif sample["split"] == "calibration":
                assert "/fleurs/" in sample["audio_path"]


def test_main_refuses_when_a_language_pool_is_too_small(tmp_path, monkeypatch):
    alignment = tmp_path / "alignment"
    prepared = alignment / "prepared"
    prepared.mkdir(parents=True)
    root = tmp_path / "calib"
    root.mkdir()
    for name, language in (("corpus-chinese.json", "zh"), ("corpus-english.json", "en")):
        entries = [
            {
                "audio": str(_write_wav(root / f"{language}-{i}.wav", 5.0)),
                "transcript": "t",
                "speaker_id": "S",
            }
            for i in range(3)
        ]
        (prepared / name).write_text(json.dumps(entries))
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_calib_manifest.py",
            "--root",
            str(root),
            "--alignment-root",
            str(alignment),
        ],
    )
    with pytest.raises(SystemExit, match="eligible"):
        manifest.main()


def test_digest_matches_hashlib(tmp_path):
    import hashlib

    path = tmp_path / "x.bin"
    path.write_bytes(b"calibration")
    assert manifest.digest(path) == hashlib.sha256(b"calibration").hexdigest()


# --------------------------------------------------------------------------- #
# the conversion tool's config pass-through
# --------------------------------------------------------------------------- #
def test_convert_tool_passes_the_source_config_with_the_quantization_section(
    tmp_path, monkeypatch
):
    """The emitted checkpoint must carry a loadable ``config.json``.

    ``pipeline.load_model`` refuses to guess a configuration, and the converter
    writes the destination's ``config.json`` from whatever it is handed.  An
    earlier version of the tool passed ``config=None`` and produced a checkpoint
    with no ``config.json`` at all: the conversion reported success and the
    checkpoint could not be loaded.  A later version then let the GPTQ
    hyper-parameters (also called ``config``) overwrite the model config, so
    ``config.json`` got a ``GptqConfig`` object.  Both are guarded here.
    """
    import json

    from tools import gptq_calibrate

    source = tmp_path / "native-llm" / "qwen3-asr-0.6b"
    source.mkdir(parents=True)
    (source / "config.json").write_text(json.dumps({"thinker_config": {"a": 1}}))

    seen = {}

    class Info:
        files: tuple = ()
        tensor_count = 0
        tensor_bytes = 0
        largest_tensor_bytes = 0
        dtypes: dict = {}

    def fake_convert(source_path, destination, **kwargs):
        seen.update(kwargs)
        seen["source"] = source_path
        return Info()

    class FakeGptq:
        def __init__(self, hessians, configs, method="gptq"):
            self.hessians = hessians
            self.configs = configs
            self.method = method
            self.outcomes = {"thinker.model.embed_tokens.weight": {"method": "gptq"}}

    class FakeAccumulator:
        @classmethod
        def load(cls, path):
            return cls()

        def hessian(self, balance=True, language_weights=None):
            return np.eye(4, dtype=np.float32)

        def language_mass_with(self, language_weights, balance=True):
            return {}

    monkeypatch.setattr("funasr_mlx.checkpoints.convert_checkpoint", fake_convert)
    monkeypatch.setattr("funasr_mlx.checkpoints.GptqWeight", FakeGptq)
    monkeypatch.setattr(
        "funasr_mlx.models.llm.gptq.HessianAccumulator.load", FakeAccumulator.load
    )
    # The tool imports it inside the function, so patch it on the module the
    # tool imports FROM (funasr_mlx.memory), which is what the local import sees.
    monkeypatch.setattr("funasr_mlx.memory.guarded_child_active", lambda: True)

    class Args:
        family = "qwen3-asr-0.6b"
        phase = "convert"
        native_root = tmp_path / "native-llm"
        source_dir = None
        manifest = tmp_path / "unused.json"
        hessian = tmp_path / "h.npz"
        destination = tmp_path / "out"
        report = tmp_path / "report.json"
        config = None
        dtype = "bfloat16"
        storage = "bfloat16"
        quantization = {
            "bits": 4,
            "group_size": 64,
            "mode": "affine",
            "targets": ["thinker.model.embed_tokens"],
            "method": "gptq",
        }
        method = "gptq"
        language_weights = None
        max_language_mass = 0.30
        damp_percent = 0.01
        block_size = 128
        no_act_order = False
        tensor_limit = 1_000_000

    assert gptq_calibrate.convert(Args()) == 0

    # The model's own config came through untouched ...
    assert seen["config"]["thinker_config"] == {"a": 1}
    # ... and is a plain JSON-able mapping, not a GptqConfig object.
    assert isinstance(seen["config"], dict)
    json.dumps(seen["config"])
    # ... with the quantization section the loader needs.
    assert seen["config"]["quantization"]["method"] == "gptq"
    # ... and the GPTQ hyper-parameters went to GptqWeight, not config.json.
    assert "bits" in seen["gptq"].configs["thinker.model.embed_tokens.weight"]
    assert "quantization" not in seen["gptq"].configs["thinker.model.embed_tokens.weight"]

    # The report must actually be written.  An earlier version referenced the
    # renamed local here and crashed *after* a successful conversion, so the
    # checkpoint existed but the run reported failure.
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["gptq"]["thinker.model.embed_tokens.weight"]["method"] == "gptq"
    assert report["gptq_config"]["bits"] == 4
    assert report["targets"] == ["thinker.model.embed_tokens"]
