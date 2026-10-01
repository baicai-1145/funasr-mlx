# FunASR MLX

Native MLX inference ports for Apple Silicon, developed against the FunASR reference repository.

## Development Status

This is a new implementation, not yet a validated replacement for FunASR. Each model needs checkpoint conversion, numerical comparison, and real-audio acceptance before being marked supported.

| Model | State |
| --- | --- |
| SenseVoiceSmall | Implementation in progress |
| Paraformer | Implementation in progress |
| ct-punc | Implementation in progress |
| fsmn-vad | Implementation in progress |
| CAM++ | Planned |
| emotion2vec+large | Planned |
| Whisper | Planned (native MLX backend) |

## Memory Policy

The hard ceiling is 10 GB (decimal), including all processes launched for this project, not 10 GiB. Large jobs are serialized. Preflight checks must reject unsafe jobs; MLX allocation/cache limits and a process watchdog provide additional protection. A process cannot reserve a global system-memory ceiling against other applications, so jobs must also consider available host memory and refuse to start under pressure. No automatic downloads or simultaneous model loading.

Performance claims require measured steady-state latency, real-time factor, peak RSS and MLX allocation, input duration, precision, chip, and reference accuracy. Quantization is opt-in and must be assessed per model.
