"""Native MLX inference for FunASR's CT-Transformer punctuation model (``ct-punc``).

Full port notes (upstream file map, design rules, known deviations): docs/port-notes/ct_punc.py.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields as dataclass_fields
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import mlx.core as mx
import mlx.nn as nn

from ._strict_load import (
    _as_float,
    _as_int,
    _known_fields,
    _nest,
    _require,
    convert_checkpoint_tensors as _strict_convert,
    flat_parameter_names,
    load_state_dict_strict as _strict_load,
)

__all__ = [
    "ASCII_PUNC_MAP",
    "CtPuncConfig",
    "CharVocab",
    "CtPuncModel",
    "CtPuncPunctuator",
    "ModuleLayerList",
    "PuncResult",
    "PositionwiseFeedForward",
    "SanmAttention",
    "SanmEncoder",
    "SanmEncoderConfig",
    "SanmEncoderLayer",
    "convert_checkpoint_tensors",
    "expand_word_level_punc",
    "flat_parameter_names",
    "is_first_byte_ascii",
    "load_state_dict_strict",
    "make_valid_mask",
    "sinusoidal_position_encoding",
    "split_to_mini_sentence",
    "split_words",
]


# FunASR maps the full-width marks to ASCII when the neighbouring token is ASCII.
ASCII_PUNC_MAP: Mapping[str, str] = {"，": ",", "。": ".", "？": "?", "、": ","}

# ``forward_attention`` fills masked scores with this value before the softmax.
MASK_SCORE_VALUE = float("-inf")

# Depthwise SANM convolution weights are the only tensors needing a layout change.
CONV1D_WEIGHT_SUFFIX = ".self_attn.fsmn_block.weight"


def _depthwise_conv1d(name: str, value):
    # (C, 1, K) -> (C, K, 1); SANM depthwise memory kernels only
    if value.ndim != 3:
        raise ValueError(
            f"{name}: expected a 3-D depthwise conv weight (C, 1, K), got shape {value.shape}"
        )
    return mx.transpose(value, (0, 2, 1))


CONV1D_RULES = {CONV1D_WEIGHT_SUFFIX: _depthwise_conv1d}


def convert_checkpoint_tensors(state):
    return _strict_convert(state, CONV1D_RULES)


def load_state_dict_strict(model, state, *, ignore_prefixes=()):
    return _strict_load(model, state, ignore_prefixes=ignore_prefixes, rules=CONV1D_RULES)

# FunASR's ``LayerNorm`` (funasr/models/transformer/layer_norm.py) overrides the
# PyTorch default of 1e-5.  Getting this wrong silently changes every output.
FUNASR_LAYER_NORM_EPS = 1e-12

# FunASR's jieba word-level expansion only considers CJK Unified Ideographs
# (``"\u4e00" <= token[0] <= "\u9fa5"`` in the reference).
CJK_RANGE = ("\u4e00", "\u9fa5")


# validation helpers


# config
@dataclass(frozen=True)
class SanmEncoderConfig:
    # see docs/port-notes/functions.md :: SanmEncoderConfig

    input_size: int = 256
    output_size: int = 256
    attention_heads: int = 4
    linear_units: int = 2048
    num_blocks: int = 6
    dropout_rate: float = 0.1
    positional_dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.0
    input_layer: str = "pe"
    normalize_before: bool = True
    kernel_size: int = 11
    sanm_shfit: int = 0
    selfattention_layer_type: str = "sanm"
    padding_idx: int = -1

    def __post_init__(self) -> None:
        for name in (
            "input_size",
            "output_size",
            "attention_heads",
            "linear_units",
            "num_blocks",
            "kernel_size",
        ):
            object.__setattr__(self, name, _as_int(getattr(self, name), name, minimum=1))
        object.__setattr__(self, "sanm_shfit", _as_int(self.sanm_shfit, "sanm_shfit", minimum=0))
        object.__setattr__(
            self, "padding_idx", _as_int(self.padding_idx, "padding_idx", minimum=-1)
        )
        for name in ("dropout_rate", "positional_dropout_rate", "attention_dropout_rate"):
            value = _as_float(getattr(self, name), name)
            _require(0.0 <= value <= 1.0, f"{name} must be within [0, 1], got {value}")
            object.__setattr__(self, name, value)
        _require(
            self.output_size % self.attention_heads == 0,
            "output_size must be divisible by attention_heads "
            f"({self.output_size} % {self.attention_heads} != 0)",
        )
        _require(
            self.selfattention_layer_type == "sanm",
            "only selfattention_layer_type='sanm' is implemented, got "
            f"{self.selfattention_layer_type!r}",
        )
        _require(
            self.input_layer == "pe",
            "only input_layer='pe' is implemented for ct-punc (conv2d/embed/linear "
            f"input layers change the time axis or parameter names), got {self.input_layer!r}",
        )
        _require(
            (self.kernel_size - 1) // 2 + self.sanm_shfit <= self.kernel_size - 1,
            f"sanm_shfit={self.sanm_shfit} makes the depthwise convolution padding "
            "negative for kernel_size=" + str(self.kernel_size),
        )

    @property
    def d_k(self) -> int:
        return self.output_size // self.attention_heads

    @property
    def conv_left_pad(self) -> int:
        left = (self.kernel_size - 1) // 2
        if self.sanm_shfit > 0:
            left += self.sanm_shfit
        return left

    @property
    def conv_right_pad(self) -> int:
        return self.kernel_size - 1 - self.conv_left_pad

    @classmethod
    def from_dict(cls, config: Mapping[str, Any]) -> "SanmEncoderConfig":
        # see docs/port-notes/functions.md :: from_dict
        if not isinstance(config, Mapping):
            raise TypeError(f"encoder_conf must be a mapping, got {type(config).__name__}")
        # The official config passes pos_enc_class (a class object the SANM
        # encoder only reads for other input_layers; input_layer="pe" never
        # touches it). It selects no weights here, so accept-and-ignore keeps
        # the config loadable while adding no silent behaviour.
        config = {k: v for k, v in config.items() if k != "pos_enc_class"}
        return cls(**_known_fields(cls, config, "encoder_conf"))


@dataclass(frozen=True)
class CtPuncConfig:
    # see docs/port-notes/functions.md :: CtPuncConfig

    vocab_size: int = 471067
    embed_unit: int = 256
    att_unit: int = 256
    dropout_rate: float = 0.1
    ignore_id: int = 0
    sos: int = 1
    eos: int = 2
    sentence_end_id: int = 3
    punc_list: tuple[str, ...] = ("_", "，", "。", "？", "、")
    punc_weight: tuple[float, ...] = ()
    #: Marks that terminate a sentence for the mini-sentence cache search.
    sentence_end_puncs: tuple[str, ...] = ("。", "？")
    #: Mark considered as the fallback cut point when a chunk grows past the limit.
    comma_punc: str = "，"
    #: ``CTTransformer.inference``'s ``split_size`` kwarg.
    split_size: int = 20
    #: ``CTTransformer.inference``'s ``cache_pop_trigger_limit``.
    cache_pop_trigger_limit: int = 200
    #: Hard bound on tokens fed to the encoder in one forward (attention is O(T^2);
    #: the mini-sentence cache keeps T far below this).
    max_input_tokens: int = 4096
    encoder: SanmEncoderConfig = SanmEncoderConfig()

    def __post_init__(self) -> None:
        for name in ("vocab_size", "embed_unit", "att_unit", "max_input_tokens"):
            object.__setattr__(self, name, _as_int(getattr(self, name), name, minimum=1))
        for name in ("split_size", "cache_pop_trigger_limit"):
            object.__setattr__(self, name, _as_int(getattr(self, name), name, minimum=2))
        for name in ("ignore_id", "sos", "eos"):
            object.__setattr__(self, name, _as_int(getattr(self, name), name))
        object.__setattr__(
            self, "sentence_end_id", _as_int(self.sentence_end_id, "sentence_end_id")
        )
        object.__setattr__(self, "punc_list", tuple(self.punc_list))
        object.__setattr__(self, "punc_weight", tuple(float(w) for w in self.punc_weight))
        object.__setattr__(self, "sentence_end_puncs", tuple(self.sentence_end_puncs))

        _require(len(self.punc_list) > 0, "punc_list must not be empty")
        _require(
            0 <= self.sentence_end_id < len(self.punc_list),
            f"sentence_end_id={self.sentence_end_id} is outside punc_list "
            f"({len(self.punc_list)} entries)",
        )
        if self.punc_weight:
            _require(
                len(self.punc_weight) == len(self.punc_list),
                "punc_weight must have the same length as punc_list "
                f"({len(self.punc_weight)} != {len(self.punc_list)})",
            )
        _require(len(self.sentence_end_puncs) > 0, "sentence_end_puncs must not be empty")
        _require(
            self.embed_unit == self.att_unit,
            "ct-punc checkpoints use embed_unit == att_unit "
            f"({self.embed_unit} != {self.att_unit})",
        )
        _require(
            isinstance(self.encoder, SanmEncoderConfig),
            f"encoder must be a SanmEncoderConfig, got {type(self.encoder).__name__}",
        )
        _require(
            self.encoder.input_size == self.embed_unit,
            "encoder input_size must equal embed_unit "
            f"({self.encoder.input_size} != {self.embed_unit})",
        )
        _require(
            self.encoder.output_size == self.att_unit,
            "encoder output_size must equal att_unit "
            f"({self.encoder.output_size} != {self.att_unit})",
        )

    @property
    def punc_size(self) -> int:
        return len(self.punc_list)

    @classmethod
    def from_dict(cls, config: Mapping[str, Any]) -> "CtPuncConfig":
        # see docs/port-notes/functions.md :: from_dict
        if not isinstance(config, Mapping):
            raise TypeError(f"config must be a mapping, got {type(config).__name__}")
        if "model_conf" in config:
            model_conf = dict(config["model_conf"] or {})
            encoder_conf = dict(config.get("encoder_conf") or {})
        else:
            model_conf = dict(config)
            model_conf.pop("encoder_conf", None)
            encoder_conf = dict(config.get("encoder_conf") or {})

        tokenizer_conf = dict(config.get("tokenizer_conf") or {})
        if "vocab_size" not in model_conf and "token_list" not in tokenizer_conf:
            raise ValueError(
                "vocab_size is required: pass model_conf.vocab_size or a tokenizer "
                "vocabulary (tokenizer_conf.token_list)"
            )

        payload = _known_fields(cls, model_conf, "model_conf")
        payload.pop("encoder", None)
        return cls(encoder=SanmEncoderConfig.from_dict(encoder_conf), **payload)


# vocabulary
class CharVocab:
    # see docs/port-notes/functions.md :: CharVocab

    def __init__(self, token_list: Iterable[str], unk_symbol: str = "<unk>") -> None:
        self.token_list: list[str] = list(token_list)
        self.token2id: dict[str, int] = {}
        for index, token in enumerate(self.token_list):
            if token in self.token2id:
                raise ValueError(f'symbol "{token}" is duplicated in the vocabulary')
            self.token2id[token] = index
        if unk_symbol not in self.token2id:
            raise ValueError(f"unknown symbol {unk_symbol!r} is absent from the vocabulary")
        self.unk_symbol = unk_symbol
        self.unk_id = self.token2id[unk_symbol]

    def __len__(self) -> int:
        return len(self.token_list)

    @classmethod
    def from_file(cls, path: str | Path, unk_symbol: str = "<unk>") -> "CharVocab":
        # see docs/port-notes/functions.md :: from_file
        with Path(path).open("r", encoding="utf-8") as handle:
            tokens = [line.rstrip() for line in handle]
        return cls(tokens, unk_symbol=unk_symbol)

    @classmethod
    def from_config(cls, config: Mapping[str, Any], path: str | Path | None = None) -> "CharVocab":
        # see docs/port-notes/functions.md :: from_config
        source = config or {}
        if "tokenizer_conf" in source:
            tokenizer_conf = dict(source["tokenizer_conf"] or {})
        else:
            tokenizer_conf = dict(source)
        unk_symbol = str(tokenizer_conf.get("unk_symbol", "<unk>"))
        token_list = tokenizer_conf.get("token_list", tokenizer_conf.get("token_path"))
        if path is not None:
            return cls.from_file(path, unk_symbol=unk_symbol)
        if isinstance(token_list, str):
            return cls.from_file(token_list, unk_symbol=unk_symbol)
        if token_list is None:
            raise ValueError(
                "tokenizer_conf.token_list (path or sequence) is required to build a CharVocab"
            )
        return cls(token_list, unk_symbol=unk_symbol)

    def encode(self, tokens: Sequence[str]) -> list[int]:
        unk = self.unk_id
        return [self.token2id.get(token, unk) for token in tokens]


# tensor helpers
def is_first_byte_ascii(token: str) -> bool:
    # see docs/port-notes/functions.md :: is_first_byte_ascii
    if not token:
        return False
    return len(token[0].encode("utf-8")) == 1


def make_valid_mask(
    lengths: Sequence[int] | mx.array, maxlen: int, dtype: Any = mx.float32
) -> mx.array:
    # see docs/port-notes/functions.md :: make_valid_mask
    if isinstance(lengths, mx.array):
        lens = lengths.reshape(-1).astype(mx.int32)
    else:
        lens = mx.array([int(x) for x in lengths], dtype=mx.int32).reshape(-1)
    positions = mx.arange(maxlen, dtype=mx.int32).reshape(1, -1)
    padding = positions >= lens.reshape(-1, 1)
    return mx.logical_not(padding).astype(dtype)[:, None, :]


def sinusoidal_position_encoding(x: mx.array) -> mx.array:
    # see docs/port-notes/functions.md :: sinusoidal_position_encoding
    if x.ndim != 3:
        raise ValueError(f"expected a (B, T, D) array, got shape {x.shape}")
    depth = x.shape[-1]
    if depth % 2 != 0 or depth // 2 - 1 == 0:
        raise ValueError(f"sinusoidal encoding needs an even depth >= 4, got {depth}")
    timesteps = x.shape[1]

    positions = mx.arange(1, timesteps + 1, dtype=mx.float32).reshape(1, -1, 1)
    log_increment = math.log(10000.0) / (depth // 2 - 1)
    inv_timescales = mx.exp(mx.arange(depth // 2, dtype=mx.float32) * (-log_increment))
    scaled_time = positions * inv_timescales.reshape(1, 1, -1)
    encoding = mx.concatenate([mx.sin(scaled_time), mx.cos(scaled_time)], axis=-1)
    return x + encoding.astype(x.dtype)


# modules
class ModuleLayerList(nn.Module):
    # see docs/port-notes/functions.md :: ModuleLayerList

    def __init__(self, modules: Sequence[nn.Module]) -> None:
        super().__init__()
        for index, module in enumerate(modules):
            self[str(index)] = module

    def __len__(self) -> int:
        return dict.__len__(self)

    def __iter__(self):
        for index in range(dict.__len__(self)):
            yield self[str(index)]

    def __getitem__(self, index: int | str) -> nn.Module:
        return self[str(index)] if isinstance(index, int) else dict.__getitem__(self, index)


class PositionwiseFeedForward(nn.Module):
    # see docs/port-notes/functions.md :: PositionwiseFeedForward

    def __init__(self, idim: int, hidden_units: int) -> None:
        super().__init__()
        self.w_1 = nn.Linear(idim, hidden_units)
        self.w_2 = nn.Linear(hidden_units, idim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w_2(nn.relu(self.w_1(x)))


class SanmAttention(nn.Module):
    # see docs/port-notes/functions.md :: SanmAttention

    def __init__(
        self,
        n_head: int,
        in_feat: int,
        n_feat: int,
        kernel_size: int,
        sanm_shfit: int = 0,
        left_pad: int | None = None,
        right_pad: int | None = None,
    ) -> None:
        super().__init__()
        if n_feat % n_head != 0:
            raise ValueError(f"n_feat={n_feat} must be divisible by n_head={n_head}")
        self.h = n_head
        self.d_k = n_feat // n_head
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.linear_q_k_v = nn.Linear(in_feat, n_feat * 3)
        self.fsmn_block = nn.Conv1d(
            n_feat,
            n_feat,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=n_feat,
            bias=False,
        )
        self.left_pad = (kernel_size - 1) // 2 if left_pad is None else int(left_pad)
        self.right_pad = kernel_size - 1 - self.left_pad if right_pad is None else int(right_pad)
        if self.left_pad < 0 or self.right_pad < 0:
            raise ValueError(
                f"depthwise padding must be non-negative, got "
                f"left={self.left_pad}, right={self.right_pad}"
            )

    def forward_qkv(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        b, t, _ = x.shape
        q_k_v = self.linear_q_k_v(x)
        q, k, v = mx.split(q_k_v, 3, axis=-1)
        q_h = q.reshape(b, t, self.h, self.d_k).swapaxes(1, 2)
        k_h = k.reshape(b, t, self.h, self.d_k).swapaxes(1, 2)
        v_h = v.reshape(b, t, self.h, self.d_k).swapaxes(1, 2)
        return q_h, k_h, v_h, v

    def forward_fsmn(self, inputs: mx.array, mask: mx.array | None) -> mx.array:
        """Depthwise memory branch; the mask is applied before *and* after the conv.

        Full notes: docs/port-notes/functions.md :: forward_fsmn
        """
        b = inputs.shape[0]
        mask3 = None
        if mask is not None:
            mask3 = mx.reshape(mask, (b, -1, 1))
            inputs = inputs * mask3
        x = mx.transpose(inputs, (0, 2, 1))  # (B, T, C) -> (B, C, T)
        x = mx.pad(x, [(0, 0), (0, 0), (self.left_pad, self.right_pad)])
        x = mx.transpose(x, (0, 2, 1))  # (B, C, T_padded) -> (B, T_padded, C)
        x = self.fsmn_block(x)  # channels-last conv -> (B, T, C)
        x = x + inputs
        if mask3 is not None:
            x = x * mask3
        return x

    def forward_attention(
        self, value: mx.array, scores: mx.array, mask: mx.array | None
    ) -> mx.array:
        n_batch = value.shape[0]
        if mask is not None:
            attn_mask = mx.expand_dims(mask == 0, 1)  # (B, 1, 1, T)
            scores = mx.where(attn_mask, MASK_SCORE_VALUE, scores)
            attn = mx.softmax(scores, axis=-1)
            attn = mx.where(attn_mask, 0.0, attn)
        else:
            attn = mx.softmax(scores, axis=-1)
        x = attn @ value  # (B, H, T, d_k)
        x = mx.transpose(x, (0, 2, 1, 3)).reshape(n_batch, -1, self.h * self.d_k)
        return self.linear_out(x)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        q_h, k_h, v_h, v = self.forward_qkv(x)
        fsmn_memory = self.forward_fsmn(v, mask)
        q_h = q_h * (self.d_k**-0.5)
        scores = q_h @ mx.transpose(k_h, (0, 1, 3, 2))
        return self.forward_attention(v_h, scores, mask) + fsmn_memory


class SanmEncoderLayer(nn.Module):

    def __init__(self, in_size: int, size: int, config: SanmEncoderConfig) -> None:
        super().__init__()
        self.in_size = int(in_size)
        self.size = int(size)
        self.normalize_before = bool(config.normalize_before)
        self.self_attn = SanmAttention(
            n_head=config.attention_heads,
            in_feat=in_size,
            n_feat=size,
            kernel_size=config.kernel_size,
            sanm_shfit=config.sanm_shfit,
            left_pad=config.conv_left_pad,
            right_pad=config.conv_right_pad,
        )
        self.feed_forward = PositionwiseFeedForward(size, config.linear_units)
        self.norm1 = nn.LayerNorm(in_size, eps=FUNASR_LAYER_NORM_EPS)
        self.norm2 = nn.LayerNorm(size, eps=FUNASR_LAYER_NORM_EPS)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        residual = x
        if self.normalize_before:
            x = self.norm1(x)
        attn = self.self_attn(x, mask)
        # FunASR only adds the residual when in_size == size (encoders0).
        x = residual + attn if self.in_size == self.size else attn
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.feed_forward(x)
        if not self.normalize_before:
            x = self.norm2(x)
        return x


class SanmEncoder(nn.Module):
    # see docs/port-notes/functions.md :: SanmEncoder

    def __init__(self, config: SanmEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self._output_size = int(config.output_size)
        self.encoders0 = ModuleLayerList(
            [SanmEncoderLayer(config.input_size, config.output_size, config)]
        )
        self.encoders = ModuleLayerList(
            [
                SanmEncoderLayer(config.output_size, config.output_size, config)
                for _ in range(config.num_blocks - 1)
            ]
        )
        self.after_norm = (
            nn.LayerNorm(config.output_size, eps=FUNASR_LAYER_NORM_EPS)
            if config.normalize_before
            else None
        )

    def output_size(self) -> int:
        return self._output_size

    def __call__(self, xs_pad: mx.array, ilens: Sequence[int] | mx.array):
        if xs_pad.ndim != 3:
            raise ValueError(f"expected (B, T, D) input, got shape {xs_pad.shape}")
        maxlen = xs_pad.shape[1]
        masks = make_valid_mask(ilens, maxlen, dtype=xs_pad.dtype)
        xs_pad = xs_pad * float(self._output_size) ** 0.5
        xs_pad = sinusoidal_position_encoding(xs_pad)

        for layer in self.encoders0:
            xs_pad = layer(xs_pad, masks)
        for layer in self.encoders:
            xs_pad = layer(xs_pad, masks)

        if self.after_norm is not None:
            xs_pad = self.after_norm(xs_pad)

        olens = [int(v) for v in mx.sum(masks[:, 0, :], axis=-1).astype(mx.int32).tolist()]
        return xs_pad, olens


class CtPuncModel(nn.Module):
    # see docs/port-notes/functions.md :: CtPuncModel

    def __init__(self, config: CtPuncConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.embed_unit)
        self.encoder = SanmEncoder(config.encoder)
        self.decoder = nn.Linear(config.att_unit, config.punc_size)

    def punc_forward(
        self, text: mx.array, text_lengths: Sequence[int] | mx.array | None = None
    ) -> mx.array:
        if text.ndim != 2:
            raise ValueError(f"expected (B, T) token ids, got shape {text.shape}")
        if text_lengths is None:
            text_lengths = [text.shape[1]] * text.shape[0]
        if isinstance(text_lengths, mx.array):
            text_lengths = [int(v) for v in text_lengths.reshape(-1).tolist()]
        x = self.embed(text.astype(mx.int32))
        h, _ = self.encoder(x, text_lengths)
        return self.decoder(h)

    def __call__(
        self, text: mx.array, text_lengths: Sequence[int] | mx.array | None = None
    ) -> mx.array:
        return self.punc_forward(text, text_lengths)

    def argmax_punc_ids(self, logits: mx.array) -> list[list[int]]:
        # see docs/port-notes/functions.md :: argmax_punc_ids
        if logits.ndim != 3:
            raise ValueError(f"expected (B, T, C) logits, got shape {logits.shape}")
        b, t, c = logits.shape
        flat = mx.argmax(mx.reshape(logits, (b * t, c)), axis=-1)
        ids = [int(v) for v in flat.tolist()]
        return [ids[i * t : (i + 1) * t] for i in range(b)]


# strict checkpoint loading


# text post-processing (pure Python, mirrors funasr/models/ct_transformer)
def split_words(text: str) -> list[str]:
    # see docs/port-notes/functions.md :: split_words
    words: list[str] = []
    for segment in text.split():
        current = ""
        for char in segment:
            if len(char.encode("utf-8")) == 1:
                current += char
            else:
                if current:
                    words.append(current)
                    current = ""
                words.append(char)
        if current:
            words.append(current)
    return words


def split_to_mini_sentence(words: Sequence[Any], word_limit: int = 20) -> list[list[Any]]:
    # see docs/port-notes/functions.md :: split_to_mini_sentence
    if word_limit <= 1:
        raise ValueError(f"word_limit must be > 1, got {word_limit}")
    if len(words) <= word_limit:
        return [list(words)]
    sentences: list[list[Any]] = []
    length = len(words)
    sentence_len = length // word_limit
    for i in range(sentence_len):
        sentences.append(list(words[i * word_limit : (i + 1) * word_limit]))
    if length % word_limit > 0:
        sentences.append(list(words[sentence_len * word_limit :]))
    return sentences


def expand_word_level_punc(
    punc_array: Sequence[int],
    tokens: Sequence[str],
    no_punc_id: int = 1,
) -> list[int]:
    # see docs/port-notes/functions.md :: expand_word_level_punc
    values = list(punc_array)
    total = len(tokens)
    for i, token in enumerate(reversed(list(tokens))):
        if not token:
            continue
        if CJK_RANGE[0] <= token[0] <= CJK_RANGE[1] and len(token) > 1:
            insert_at = total - i - 1
            for _ in range(len(token) - 1):
                values.insert(insert_at, no_punc_id)
    return values


@dataclass(frozen=True)
class PuncResult:

    text: str
    punc_array: list[int] | None = None


class CtPuncPunctuator:
    # see docs/port-notes/functions.md :: CtPuncPunctuator

    def __init__(
        self,
        model: CtPuncModel,
        vocab: CharVocab,
        config: CtPuncConfig | None = None,
    ) -> None:
        self.model = model
        self.vocab = vocab
        self.config = config if config is not None else model.config
        if len(vocab) != self.config.vocab_size:
            raise ValueError(
                "vocabulary size does not match the model config "
                f"({len(vocab)} != {self.config.vocab_size})"
            )

    # -- state ---------------------------------------------------------------
    def reset(self) -> None:
        # see docs/port-notes/functions.md :: reset
        return None

    # -- prediction ----------------------------------------------------------
    def predict_punc_ids(self, token_ids: Sequence[int]) -> list[int]:
        ids = [int(i) for i in token_ids]
        if not ids:
            return []
        if len(ids) > self.config.max_input_tokens:
            raise ValueError(
                f"mini-sentence of {len(ids)} tokens exceeds max_input_tokens="
                f"{self.config.max_input_tokens}"
            )
        text = mx.array([ids], dtype=mx.int32)
        logits = self.model.punc_forward(text, [len(ids)])
        return self.model.argmax_punc_ids(logits)[0]

    # -- full pipeline -------------------------------------------------------
    def punctuate(self, text: str) -> PuncResult:
        if not text or not text.strip():
            return PuncResult(text="", punc_array=None)

        tokens = split_words(text)
        token_ids = self.vocab.encode(tokens)
        mini_sentences = split_to_mini_sentence(tokens, self.config.split_size)
        mini_sentence_ids = split_to_mini_sentence(token_ids, self.config.split_size)
        if len(mini_sentences) != len(mini_sentence_ids):
            raise RuntimeError(
                "token and token-id mini-sentences diverged: "
                f"{len(mini_sentences)} != {len(mini_sentence_ids)}"
            )

        cfg = self.config
        punc_list = cfg.punc_list
        cache_sent: list[str] = []
        cache_sent_id: list[int] = []
        accumulated_text = ""
        punc_array: list[int] = []
        out_text = ""
        last_index = len(mini_sentences) - 1

        for chunk_index in range(len(mini_sentences)):
            mini_sentence = cache_sent + list(mini_sentences[chunk_index])
            mini_sentence_id = cache_sent_id + list(mini_sentence_ids[chunk_index])

            punctuations = self.predict_punc_ids(mini_sentence_id)
            if len(punctuations) != len(mini_sentence):
                raise RuntimeError(
                    "punc prediction length mismatch: "
                    f"{len(punctuations)} != {len(mini_sentence)} tokens"
                )

            if chunk_index < last_index:
                sentence_end = -1
                last_comma_index = -1
                # FunASR scans [len - 2 .. 2]; index 0/1 and the final token are
                # never eligible to close a sentence.
                for i in range(len(punctuations) - 2, 1, -1):
                    mark = punc_list[punctuations[i]]
                    if mark in cfg.sentence_end_puncs:
                        sentence_end = i
                        break
                    if last_comma_index < 0 and mark == cfg.comma_punc:
                        last_comma_index = i
                if (
                    sentence_end < 0
                    and len(mini_sentence) > cfg.cache_pop_trigger_limit
                    and last_comma_index >= 0
                ):
                    sentence_end = last_comma_index
                    punctuations[sentence_end] = cfg.sentence_end_id
                cache_sent = mini_sentence[sentence_end + 1 :]
                cache_sent_id = mini_sentence_id[sentence_end + 1 :]
                mini_sentence = mini_sentence[: sentence_end + 1]
                punctuations = punctuations[: sentence_end + 1]

            words_with_punc: list[str] = []
            previous_mark: str | None = None
            previous_ascii = False
            for i, token in enumerate(mini_sentence):
                ascii_first = is_first_byte_ascii(token)
                if ascii_first and (i == 0 or previous_mark in cfg.sentence_end_puncs):
                    token = token.capitalize()
                if i == 0 and ascii_first:
                    token = " " + token
                elif i > 0 and ascii_first and previous_ascii:
                    token = " " + token
                words_with_punc.append(token)

                raw_mark = punc_list[punctuations[i]]
                if raw_mark != "_":
                    mark = ASCII_PUNC_MAP.get(raw_mark, raw_mark) if ascii_first else raw_mark
                    words_with_punc.append(mark)
                previous_mark = raw_mark
                previous_ascii = ascii_first
            accumulated_text += "".join(words_with_punc)

            # FunASR keeps `new_mini_sentence_punc` as the running list and returns
            # it verbatim as `punc_array`; the trailing sentence-end rewrite below
            # is applied to both, so `punc_array` is accumulated directly here.
            out_text = accumulated_text
            rewrite_last_punc = False
            if chunk_index == last_index and accumulated_text:
                last_char = accumulated_text[-1]
                drop_last = False
                replacement: str | None = None
                if last_char in ("，", "、", ","):
                    drop_last = True
                    replacement = "。" if last_char != "," else "."
                elif (
                    last_char not in cfg.sentence_end_puncs and len(last_char.encode("utf-8")) != 1
                ):
                    replacement = "。"
                elif last_char not in (".", "?") and len(last_char.encode("utf-8")) == 1:
                    replacement = "."
                if replacement is not None:
                    out_text = (
                        accumulated_text[:-1] + replacement
                        if drop_last
                        else accumulated_text + replacement
                    )
                    rewrite_last_punc = True

            punc_array += punctuations
            if rewrite_last_punc and punctuations:
                punctuations[-1] = cfg.sentence_end_id
                punc_array[-1] = cfg.sentence_end_id

        return PuncResult(text=out_text, punc_array=punc_array)
