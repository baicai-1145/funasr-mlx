import json

import yaml

from tools.prepare_large_alignment import main


def _write_campaign(tmp_path, tensors, family="emotion2vec", config=None, declared=None):
    model_dir = tmp_path / "models" / family
    model_dir.mkdir(parents=True)
    (model_dir / "config.yaml").write_text(
        yaml.safe_dump(
            config
            if config is not None
            else {
                "model": "Emotion2vec",
                "model_conf": {
                    "embed_dim": 32,
                    "depth": 2,
                    "num_heads": 2,
                    "mlp_ratio": 2.0,
                    "normalize": True,
                    "modalities": {"audio": {"feature_encoder_spec": "[(512,10,5)]"}},
                },
                "scope_map": ["d2v_model.", "none"],
            }
        )
    )
    (model_dir / "tokens.txt").write_text("happy\nsad\n")
    (model_dir / "weight-inventory.json").write_text(
        json.dumps(
            {
                "tensors": {
                    name: {
                        "shape": list(value.shape),
                        "dtype": str(value.dtype).replace("torch.", ""),
                        "bytes": value.numel() * value.element_size(),
                    }
                    for name, value in (declared if declared is not None else tensors).items()
                },
                "checkpoint": str(model_dir / "model.pt"),
            }
        )
    )
    import torch

    torch.save(tensors, model_dir / "model.pt")
    return model_dir


def _labels():
    return ["d2v_model.modality_encoders.AUDIO.extra_tokens", "d2v_model.blocks.0.attn.proj.weight"]


def test_emotion_conversion_strips_scope_prefix_and_records_disposition(
    tmp_path, monkeypatch, capsys
):
    import numpy as np
    import torch

    tensors = {
        "d2v_model.modality_encoders.AUDIO.extra_tokens": torch.arange(
            64, dtype=torch.float32
        ).reshape(1, 2, 32),
        "d2v_model.blocks.0.attn.proj.weight": torch.eye(32, dtype=torch.float32).reshape(
            1, 32, 32
        ),
        "proj.weight": torch.ones(32, 2),
        "proj.bias": torch.zeros(2),
        "d2v_model.decoder.proj.weight": torch.ones(2, 2),
    }
    declared = {
        name: value for name, value in tensors.items() if name != "d2v_model.decoder.proj.weight"
    }
    _write_campaign(tmp_path, tensors, declared=declared)
    destination = tmp_path / "native-full" / "emotion2vec"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare-large",
            "--root",
            str(tmp_path),
            "--family",
            "emotion2vec",
            "--destination",
            str(destination),
        ],
    )
    main()
    captured = json.loads(capsys.readouterr().out)
    assert captured["family"] == "emotion2vec"
    assert captured["parameters"] == 4
    from funasr_mlx.checkpoints import inspect_checkpoint, safetensor_header

    info = inspect_checkpoint(destination)
    assert info.tensor_count == 4
    header = safetensor_header(destination / info.files[0])
    assert "modality_encoders.AUDIO.extra_tokens" in header
    assert "d2v_model" not in " ".join(header)
    np.testing.assert_allclose(header["modality_encoders.AUDIO.extra_tokens"]["shape"], [1, 2, 32])
    disposition = json.loads((destination / "config-disposition.json").read_text())
    assert "scope_map" in disposition
    assert (destination / "source-config.yaml").is_file()
    index = json.loads((destination / "model.safetensors.index.json").read_text())
    assert index["metadata"]["training_only_skipped"] == ["d2v_model.decoder.proj.weight"]


def test_emotion_conversion_refuses_tensors_outside_scope(tmp_path, monkeypatch):
    import torch

    tensors = {
        "proj.weight": torch.ones(4, 4),
        "proj.bias": torch.zeros(4),
        "encoder.weight": torch.ones(4, 4),
    }
    declared = {"proj.weight": tensors["proj.weight"], "proj.bias": tensors["proj.bias"]}
    _write_campaign(tmp_path, tensors, declared=declared)
    destination = tmp_path / "native"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare-large",
            "--root",
            str(tmp_path),
            "--family",
            "emotion2vec",
            "--destination",
            str(destination),
        ],
    )
    try:
        main()
    except ValueError as error:
        assert "outside declared inference subset" in str(error)
    else:
        raise AssertionError("conversion must refuse unscoped tensor names")
