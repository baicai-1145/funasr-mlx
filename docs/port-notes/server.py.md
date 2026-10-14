# server.py — port notes

## module

```
OpenAI-compatible local transcription server (stdlib-only HTTP).

POST /v1/audio/transcriptions  (multipart file= or raw body)
GET  /v1/models

The transcriber is a resident ``LlmAsrTranscriber`` (one model load, KV
reset-reuse between requests). Designed to run INSIDE the guarded child via
``funasr-mlx serve``; the parent process holds the memory policy.

```

## module

```
OpenAI-compatible local transcription server (stdlib-only HTTP).

POST /v1/audio/transcriptions  (multipart file= or raw body)
GET  /v1/models

The transcriber is a resident ``LlmAsrTranscriber`` (one model load, KV
reset-reuse between requests). Designed to run INSIDE the guarded child via
``funasr-mlx serve``; the parent process holds the memory policy.

```

## module

```
OpenAI-compatible local transcription server (stdlib-only HTTP).

POST /v1/audio/transcriptions  (multipart file= or raw body)
GET  /v1/models

The transcriber is a resident ``LlmAsrTranscriber`` (one model load, KV
reset-reuse between requests). Designed to run INSIDE the guarded child via
``funasr-mlx serve``; the parent process holds the memory policy.

```
