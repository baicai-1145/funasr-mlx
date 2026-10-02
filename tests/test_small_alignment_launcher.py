"""Launcher reservation checks are not model acceptance."""

from pathlib import Path

from tools.run_small_alignment import model_arguments, model_policy


def test_model_policy_preserves_project_ceiling_and_system_reserve():
    for family in ("fsmn-vad", "campplus", "whisper"):
        for mode in ("reference", "native"):
            policy = model_policy(family, mode)
            assert policy.project_limit == 8_000_000_000
            assert policy.min_available == 3_000_000_000
            assert policy.headroom == 500_000_000


def test_only_tiny_native_has_calibrated_model_reservation():
    assert model_policy("whisper", "native").process_limit == 1_200_000_000
    assert model_policy("whisper", "reference").process_limit == 1_600_000_000
    assert model_policy("campplus", "native").process_limit == 1_600_000_000


def test_both_backends_receive_identical_prepared_whisper_inputs():
    root = Path("/campaign")
    original = model_arguments(root, "whisper", "reference")
    native = model_arguments(root, "whisper", "native")
    assert original[3:] == native[3:]
    assert "manifest-english.json" in " ".join(native)
    assert "manifest-chinese.json" in " ".join(native)
