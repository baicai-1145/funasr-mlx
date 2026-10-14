# api.py — port notes

## module

```
First-class Python API for LLM-ASR transcription.

``transcribe()`` routes through the guarded CLI subprocess (same memory
policy as ``funasr-mlx transcribe``) and parses its JSON output.
``TranscribeWorker`` gives callers a long-lived in-process handle when they
already run inside their own guard (resident model, KV reset-reuse).

Example:
    from funasr_mlx.api import transcribe
    result = transcribe("audio.wav",
                        model="/path/to/qwen3-asr-0.6b-w4a16-awq-w8tower-g32")
    print(result["text"])

```

## module

```
First-class Python API for LLM-ASR transcription.

``transcribe()`` routes through the guarded CLI subprocess (same memory
policy as ``funasr-mlx transcribe``) and parses its JSON output.
``TranscribeWorker`` gives callers a long-lived in-process handle when they
already run inside their own guard (resident model, KV reset-reuse).

Example:
    from funasr_mlx.api import transcribe
    result = transcribe("audio.wav",
                        model="/path/to/qwen3-asr-0.6b-w4a16-awq-w8tower-g32")
    print(result["text"])

```

## module

```
First-class Python API for LLM-ASR transcription.

``transcribe()`` routes through the guarded CLI subprocess (same memory
policy as ``funasr-mlx transcribe``) and parses its JSON output.
``TranscribeWorker`` gives callers a long-lived in-process handle when they
already run inside their own guard (resident model, KV reset-reuse).

Example:
    from funasr_mlx.api import transcribe
    result = transcribe("audio.wav",
                        model="/path/to/qwen3-asr-0.6b-w4a16-awq-w8tower-g32")
    print(result["text"])

```
