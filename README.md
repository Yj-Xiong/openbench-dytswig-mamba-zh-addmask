# DyTSwiG-Mamba: OpenBench SE Skill

This directory is the runnable OpenBench Skill for the Chinese THCHS+DNS
DyTSwiG-Mamba AddMask checkpoint. It is separate from the VoiceBank, CCFAATC and URGENT
model versions.

| Item | Value |
|---|---|
| Task | `SE` (single-channel speech enhancement) |
| Input | Audio file path; any input sample rate is resampled to 16 kHz and channels are averaged to mono |
| Output | 16 kHz mono float WAV path in `audio_path` |
| Model | `models/AddMask_checkpoint.py:DBD_LKFCA_Net` |
| Checkpoint | `ckpt/AddMask-g_best_zh1` |
| Runtime | PyTorch FP32, CUDA, batch size 1 |
| Interface | stdio MCP `enhance_speech` tool |

The checkpoint-compatible model source is kept in
`models/AddMask_checkpoint.py` with `models/GPFCA_checkpoint.py`. The archived
`models/MulMask_checkpoint.py` supplies the matching SRU implementation.
The Skill loads all 774 checkpoint keys with `strict=True` and uses the
AddMask phase fusion formula.

## Skill Files

- `model.spec.yaml`: OpenBench registry, runtime, IO and MCP metadata.
- `model.py`: audio loading, normalization, STFT, model inference and WAV output.
- `server.py`: line-delimited JSON-RPC 2.0 MCP server with `initialize`,
  `tools/list`, `tools/call` and `ping`.
- `validate.py`: real-checkpoint MCP smoke test with a 1-second WAV.
- `pyproject.toml`: inference dependencies.

## Install and Validate

Use Python 3.10+ with a compatible CUDA PyTorch build. Install PyTorch and
CUDA extensions for the target GPU before the remaining dependencies.

```bash
cd DyTSwiG-Mamba/DyTSwiG-SE
bash setup.sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python validate.py
```

The local validation used PyTorch 2.2.0+cu121, `mamba-ssm` 2.1.0 and
`causal-conv1d` 1.4.0 on an RTX 3090. The server command in
`model.spec.yaml` is `.venv/bin/python server.py` and is run with this
directory as the working directory.

## MCP Contract

Call `enhance_speech` with `{"audio_path": "/path/noisy.wav"}`. An optional
`output_path` selects the destination. The tool returns
`{"audio_path": "/path/enhanced.wav", "sample_rate": 16000}` in
`structuredContent` and as JSON in text content. Without `output_path`, files
are written under `.runtime/outputs/`.

Silence produces a silent output. Non-finite input or model output raises an
MCP error. The checkpoint filename is an artifact identifier; no OpenBench
score is claimed here.

## Provenance

The [project README](https://github.com/Yj-Xiong/DyTSwiG-SE) describes the DyTSwiG-Mamba paper,
authors, datasets and architecture. This snapshot does not contain a
package-level license file; confirm redistribution terms for code and weights
before public upload.
