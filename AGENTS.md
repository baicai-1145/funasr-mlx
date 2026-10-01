# Development Rules

Reference repository: `/Users/baicai1145/repos/funasr` (read-only).

- Port inference semantics from the exact FunASR source, including masking, layout, padding, epsilon, and checkpoint key names. No PyTorch inference dependency.
- Absolute memory ceiling is 10,000,000,000 bytes. Leave headroom: use small synthetic dimensions only. Never load/download full weights or run MLX/PyTorch numerical tests without lead approval. All expensive verification is serialized by the lead through the memory guard.
- Agents implement and use syntax/static checks only unless explicitly authorized. Maximum two concurrent implementation agents. Do not launch other agents.
- Runtime uses `mlx.core` / `mlx.nn`; keep MLX imports lazy at the package root and CLI. Float32 reductions where needed, float16 optional weights, bounded input lengths. No global model cache.
- Preserve checkpoint-compatible parameter names and provide explicit tensor-layout conversion rules. Loading must be strict; never silently skip mismatched keys.
- Tests use tiny configurations and bounded inputs, never network access. Report what has actually been verified; architecture tests are not real-checkpoint acceptance.
- Temporary artifacts belong in `.tmp/`. Do not write to `/tmp`.
- Scope edits to assigned files, commit changes, and report commit SHA and remaining gaps.
- Use `read`/`grep`/`find` for retrieval and `edit`/`write` for manual changes.
