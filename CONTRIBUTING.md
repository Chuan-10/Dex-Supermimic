# Contributing to Dex SuperMimic

This repository trains and plays Wuji Hand 2 cube reorientation policies with mjlab and MuJoCo Warp.

## Setup

```bash
uv sync
```

Source code lives in `src/dex_supermimic`; the pinned Wuji resource package lives in
`packages/wuji-description`. Training requires an NVIDIA CUDA GPU.

## Guidelines

- Keep changes focused and preserve the task semantics unless intentionally changing them.
- Prefer mjlab-native implementations and update documentation with public interface changes.
- Keep the `dex-supermimic-train` and `dex-supermimic-play` entry points usable.
- Preserve upstream licenses, notices and attribution.

## Validation

Include relevant validation commands and results in pull requests. Check imports and
model compilation, then run short GPU train/play checks when changing the runtime.
Explain any validation that the current environment cannot support.

## License

Contributions follow [LICENSE](LICENSE) and
[CONTRIBUTOR_AGREEMENT.md](CONTRIBUTOR_AGREEMENT.md).
