"""LLM-ASR transcription drivers for the CLI / API / server layers.

One module, three consumers:
* ``funasr-mlx transcribe`` (CLI) -- single-shot, guarded child.
* ``funasr_mlx.transcribe()`` (Python API) -- resident worker with KV
  reset-reuse across calls (validated bit-stable on M4; the RTF numbers in
  the README come from this pattern).
* ``funasr_mlx serve`` (HTTP) -- wraps the same driver.

The per-family prompt/decode protocols are the acceptance-validated ones
(tools/gptq_acceptance.py); token-level behavior is identical.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

#: family -> (tokenizer source default subdir name, eos token ids)
_QWEN3_EOS = (151643, 151645, 151671)


@dataclass
class TranscribeResult:
    text: str
    token_ids: list[int]
    family: str
    wall_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "token_ids": self.token_ids,
            "family": self.family,
            "wall_ms": round(self.wall_ms, 1),
        }


class LlmAsrTranscriber:
    """Resident transcriber: one model load, many calls.

    Construct inside a guarded child (``memory.run_guarded``); the API layer
    arranges that. KV caches are allocated once and reset (offset=0) between
    utterances -- reuse validated bit-stable in .tmp/opt/LOG.md.
    """

    def __init__(self, checkpoint: str | Path, *, family: str, max_tokens: int = 96):
        import mlx.core as mx

        from .pipeline import load_model

        self.mx = mx
        self.family = family
        self.max_tokens = max_tokens
        self.model, self.config = load_model(checkpoint, family=family)
        self._ckpt_dir = Path(checkpoint).resolve()
        self._ckpt_dir = self._ckpt_dir if self._ckpt_dir.is_dir() else self._ckpt_dir.parent
        self.n_layers = 28
        self._tokenizer = None
        self._build_prompt_protocol()
        from .models.llm import KVCache

        self._caches = [KVCache() for _ in range(self.n_layers)]

    # ---------- tokenizer ----------
    def _tok(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            directory = self._ckpt_dir
            # tokenizer files may live in the checkpoint dir or in a sibling
            # fp16 source dir (quant dirs don't always carry them)
            candidates = [directory, directory.parent / directory.name.split("-w4a16")[0]]
            for cand in candidates:
                if (cand / "vocab.json").is_file() or (cand / "tokenizer.json").is_file():
                    directory = cand
                    break
            self._tokenizer = AutoTokenizer.from_pretrained(str(directory), use_fast=False)
        return self._tokenizer

    def _build_prompt_protocol(self) -> None:
        fam = self.family
        ckpt = Path(self.model_config_directory())
        if fam in ("qwen3-asr", "qwen3-asr-0.6b", "qwen3-asr-1.7b"):
            tok = self._tok()
            self._prefix = tok(
                "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>",
                add_special_tokens=False,
            )["input_ids"]
            self._tail = tok(
                "<|audio_end|><|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False
            )["input_ids"]
            self._pad = 151676
            self._eos = _QWEN3_EOS
            self._kind = "qwen3"
        elif fam == "glm-asr-nano":
            self._prefix = [59253, 10, 59261]
            self._tail = [59262, 10, 59254, 10]
            self._pad = 59260
            self._eos = (59262, 0)
            self._kind = "glm"
        elif fam == "moss-transcribe-diarize":
            tok = self._tok()
            template_path = ckpt / "chat_template.jinja"
            if not template_path.is_file():
                raise ValueError(
                    "moss-transcribe-diarize needs chat_template.jinja next to the checkpoint"
                )
            self._template = template_path.read_text()
            self._prompt_text = (
                "请将音频转写为文本，每一段需以起始时间戳和说话人编号（[S01]、[S02]、[S03]…）开头，"
                "正文为对应的语音内容，并在段末标注结束时间戳，以清晰标明该段语音范围。"
            )
            self._audio_id = tok.convert_tokens_to_ids("<|audio_pad|>")
            self._eos = (151643, 151645)
            self._kind = "moss"
        elif fam == "fun-asr-nano":
            tok = self._tok()
            self._prefix = tok(
                "<|im_start|>system\nYou are a speech recognition model.<|im_end|>\n"
                "<|im_start|>user\n语音转写：<|startofspeech|>",
                add_special_tokens=False,
            )["input_ids"]
            self._tail = tok(
                "<|endofspeech|><|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False
            )["input_ids"]
            self._eos = (151643, 151645)
            self._kind = "nano"
        else:
            raise ValueError(f"Unsupported LLM-ASR family: {fam}")

    def model_config_directory(self) -> Path:
        # load_model does not retain the path; the API layer passes the
        # checkpoint directory and tokenizer files live alongside.
        return self._ckpt_dir

    # ---------- audio front-ends ----------
    def _mel(self, samples: np.ndarray, n_mels: int):
        from .models.whisper import log_mel_spectrogram

        mel = log_mel_spectrogram(self.mx.array(samples[None, :]), n_mels=n_mels)
        return self.mx.transpose(mel[0], (1, 0))

    def _audio_embeds(self, samples: np.ndarray):
        import mlx.core as mx

        fam = self._kind
        if fam in ("qwen3", "glm"):
            n_mels = 128
            mel = self._mel(samples, n_mels).astype(self.model._tower_dtype())
            embeds = self.model.encode_audio(mel)
            mx.eval(embeds)
            return embeds, None
        if fam == "moss":
            tower_dt = self.model.whisper_encoder.conv1.weight.dtype
            parts = []
            for st in range(0, len(samples), 480000):
                chunk = samples[st : st + 480000]
                tl = (len(chunk) - 1) // 1280 + 1
                padbuf = np.zeros(480000, dtype=np.float32)
                padbuf[: len(chunk)] = chunk
                mel_c = self._mel(padbuf, 80).astype(tower_dt)
                h = self.model.whisper_encoder(mel_c)
                parts.append(h[0, : tl * 8, :])
            feat = self.mx.concatenate(parts, axis=0)[None]
            self.mx.eval(feat)
            embeds = self.model.encode_audio_from_hidden(feat)
            self.mx.eval(embeds)
            return embeds, None
        if fam == "nano":
            from .frontend import FbankConfig, WavFrontend

            dtype = self.model.audio_encoder._weight_dtype()
            fcfg = FbankConfig(
                frame_length=25.0, frame_shift=10.0, dither=0.0,
                window="hamming", n_mels=80, lfr_m=7, lfr_n=6,
            )
            fb = np.asarray(WavFrontend(fcfg, cmvn=None)(samples), dtype=np.float32)
            t = fb.shape[0]
            olens = 1 + (t - 3 + 2) // 2
            olens = 1 + (olens - 3 + 2) // 2
            fake_len = (olens - 1) // 2 + 1
            speech = self.mx.array(fb)[None].astype(dtype)
            lengths = self.mx.array([t], dtype=self.mx.int32)
            enc, _ = self.model.audio_encoder(speech, lengths)
            ad = self.model.audio_adaptor(enc)
            self.mx.eval(ad)
            return ad, fake_len
        raise ValueError(self._kind)

    # ---------- main entry ----------
    def transcribe_samples(self, samples: np.ndarray) -> str:
        """Raw-waveform transcription (pseudo-streaming adapter entry)."""
        embeds, fake_len = self._audio_embeds(samples)
        text, _ = self._decode(embeds, fake_len)
        return text

    def transcribe_file(self, audio_path: str | Path) -> TranscribeResult:
        import time

        import mlx.core as mx
        import soundfile as sf

        t0 = time.perf_counter()
        samples, sr = sf.read(str(audio_path), dtype="float32")
        embeds, fake_len = self._audio_embeds(samples)
        text, tokens = self._decode(embeds, fake_len)
        return TranscribeResult(
            text=text, token_ids=tokens, family=self.family,
            wall_ms=(time.perf_counter() - t0) * 1000,
        )

    def _decode(self, embeds, fake_len):
        mx = self.mx
        fam = self._kind
        if fam == "qwen3":
            n = int(embeds.shape[1])
            ids = self._prefix + [self._pad] * n + self._tail
            for c in self._caches:
                c.offset = 0
            lg = self.model.forward_with_audio(
                mx.array([ids], dtype=mx.int32), embeds, n, self._caches
            )
            mx.eval(lg)
            return self._greedy(lg)
        if fam == "glm":
            n = int(embeds.shape[1])
            ids = self._prefix + [self._pad] * n + self._tail
            for c in self._caches:
                c.offset = 0
            lg = self.model.forward_with_audio(
                mx.array([ids], dtype=mx.int32), embeds, n, self._caches
            )
            mx.eval(lg)
            return self._greedy(lg, glm=True)
        if fam == "moss":
            tok = self._tok()
            msgs = [{"role": "user", "content": [
                {"type": "audio", "audio": None},
                {"type": "text", "text": self._prompt_text}]}]
            rendered = tok.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False,
                chat_template=self._template)
            pre, rest = rendered.split("<|audio_pad|>", 1)
            pre_ids, tail_ids = tok.encode(pre), tok.encode(rest)
            n = int(embeds.shape[1])
            ids = pre_ids + [self._audio_id] * n + tail_ids
            for c in self._caches:
                c.offset = 0
            lg = self.model.forward_with_audio(
                mx.array([ids], dtype=mx.int32), embeds, n, self._caches
            )
            mx.eval(lg)
            return self._greedy(lg, moss=True)
        if fam == "nano":
            n = min(int(embeds.shape[1]), int(fake_len))
            beg = len(self._prefix)
            ids = self._prefix + [0] * int(fake_len) + self._tail
            emb = self.model.llm.model.embed_tokens(mx.array([ids], dtype=mx.int32))
            spliced = mx.concatenate(
                [emb[:, :beg, :], embeds[:, :n, :], emb[:, beg + int(fake_len) :, :]], axis=1
            )
            for c in self._caches:
                c.offset = 0
            h = self.model.llm.model(
                input_embeddings=spliced, cache=self._caches, mask="causal"
            )
            lg = self.model.llm.model.embed_tokens.as_linear(h[:, -1:])
            mx.eval(lg)
            return self._greedy_first(lg)
        raise ValueError(fam)

    def _greedy(self, logits, *, glm=False, moss=False):
        mx = self.mx
        tok_id = int(mx.argmax(logits[0, -1]).item())
        out = [tok_id]
        while len(out) < self.max_tokens and tok_id not in self._eos:
            lg = self._step(tok_id)
            mx.eval(lg)
            tok_id = int(mx.argmax(lg[0, -1]).item())
            out.append(tok_id)
        text = self._decode_tokens(out, glm=glm, moss=moss)
        return text, out

    def _greedy_first(self, logits):
        return self._greedy(logits)

    def _step(self, tok_id):
        mx = self.mx
        if self._kind in ("qwen3",):
            return self.model.thinker(inputs=mx.array([[tok_id]], dtype=mx.int32), cache=self._caches)
        if self._kind in ("glm", "moss", "nano"):
            decoder = getattr(self.model, "language_model", None) or self.model.llm
            return decoder(mx.array([[tok_id]], dtype=mx.int32), cache=self._caches)
        raise ValueError(self._kind)

    def _decode_tokens(self, out, *, glm=False, moss=False):
        tok = self._tok()
        text = tok.decode(out)
        for tag in ("<|im_end|>", "<|endofspeech|>"):
            text = text.replace(tag, "")
        return text.strip()


def resolve_family(checkpoint: str | Path, family: str | None) -> str:
    """Map user-provided family (or None) onto a load_model family."""
    if family:
        aliases = {
            "qwen3-asr": "qwen3-asr",
            "qwen3-asr-0.6b": "qwen3-asr",
            "qwen3-asr-1.7b": "qwen3-asr",
            "glm-asr-nano": "glm-asr-nano",
            "moss-transcribe-diarize": "moss-transcribe-diarize",
            "moss": "moss-transcribe-diarize",
            "fun-asr-nano": "fun-asr-nano",
        }
        if family not in aliases:
            raise ValueError(f"Unknown LLM-ASR family: {family}")
        return aliases[family]
    # infer from directory name
    name = Path(checkpoint).name.lower()
    for key, fam in {
        "qwen3-asr": "qwen3-asr", "glm": "glm-asr-nano",
        "moss": "moss-transcribe-diarize", "fun-asr-nano": "fun-asr-nano",
    }.items():
        if key in name:
            return fam
    raise ValueError("Cannot infer family from checkpoint name; pass --family")
