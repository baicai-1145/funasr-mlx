# download.py — port notes

## module

```
Dependency-free model downloader for the CLI.

Resolves and fetches funasr-mlx native checkpoints from Hugging Face or
ModelScope using only the standard library (``urllib``), with no
``huggingface_hub``/``modelscope`` imports. Public repos need no tokens.

Usage:
    funasr-mlx download qwen3-asr-0.6b-mlx --variant w4a16 --dest ./models

Mirrors: HF ``https://huggingface.co/{repo}/resolve/main/{path}``,
MS ``https://www.modelscope.cn/models/{repo}/resolve/master/{path}``.
Both are tried in order (HF first) with per-file fallback, so a transient
failure on one platform does not kill the download.

```

## module

```
Dependency-free model downloader for the CLI.

Resolves and fetches funasr-mlx native checkpoints from Hugging Face or
ModelScope using only the standard library (``urllib``), with no
``huggingface_hub``/``modelscope`` imports. Public repos need no tokens.

Usage:
    funasr-mlx download qwen3-asr-0.6b-mlx --variant w4a16 --dest ./models

Mirrors: HF ``https://huggingface.co/{repo}/resolve/main/{path}``,
MS ``https://www.modelscope.cn/models/{repo}/resolve/master/{path}``.
Both are tried in order (HF first) with per-file fallback, so a transient
failure on one platform does not kill the download.

```

## module

```
Dependency-free model downloader for the CLI.

Resolves and fetches funasr-mlx native checkpoints from Hugging Face or
ModelScope using only the standard library (``urllib``), with no
``huggingface_hub``/``modelscope`` imports. Public repos need no tokens.

Usage:
    funasr-mlx download qwen3-asr-0.6b-mlx --variant w4a16 --dest ./models

Mirrors: HF ``https://huggingface.co/{repo}/resolve/main/{path}``,
MS ``https://www.modelscope.cn/models/{repo}/resolve/master/{path}``.
Both are tried in order (HF first) with per-file fallback, so a transient
failure on one platform does not kill the download.

```
