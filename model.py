"""OpenBench SE wrapper for the DyTSwiG-Mamba checkpoint."""

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "ckpt" / "AddMask-g_best_zh1"


@dataclass(frozen=True)
class AudioResult:
    audio_path: str
    sample_rate: int

    @property
    def text(self):
        return self.audio_path


class DyTSwiGMamba:
    def __init__(self, model_path=None, device="auto"):
        import torch

        from env import AttrDict
        from models.AddMask_checkpoint import DBD_LKFCA_Net

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")

        with (ROOT / "config.json").open(encoding="utf-8") as handle:
            self.config = AttrDict(json.load(handle))
        self.device = torch.device(device)
        self.model = DBD_LKFCA_Net(self.config).to(self.device).eval()
        checkpoint = Path(model_path or DEFAULT_CHECKPOINT).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError("Model checkpoint not found: {}".format(checkpoint))
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        self.model.load_state_dict(state["generator"], strict=True)

    def run_tool(self, audio_path, output_path=None):
        import librosa
        import numpy as np
        import soundfile as sf
        import torch

        source = Path(audio_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError("Input audio not found: {}".format(source))
        samples, rate = sf.read(str(source), dtype="float32", always_2d=True)
        if samples.shape[0] == 0:
            raise ValueError("Input audio is empty")
        mono = samples.mean(axis=1)
        target_rate = int(self.config.sampling_rate)
        if rate != target_rate:
            mono = librosa.resample(mono, orig_sr=rate, target_sr=target_rate)
        if not np.isfinite(mono).all():
            raise ValueError("Input audio contains non-finite samples")

        signal = torch.from_numpy(mono.copy()).to(self.device).unsqueeze(0)
        if signal.square().sum().item() <= 1e-12:
            enhanced = np.zeros(len(mono), dtype=np.float32)
        else:
            with torch.inference_mode():
                scale = torch.sqrt(signal.numel() / signal.square().sum())
                normalized = signal * scale
                window = torch.hann_window(self.config.win_size, device=self.device)
                spectrum = torch.stft(
                    normalized,
                    n_fft=self.config.n_fft,
                    hop_length=self.config.hop_size,
                    win_length=self.config.win_size,
                    window=window,
                    return_complex=True,
                )
                magnitude = (spectrum.real.square() + spectrum.imag.square() + 1e-10).sqrt()
                phase = torch.atan2(spectrum.imag + 1e-10, spectrum.real + 1e-10)
                magnitude = magnitude.pow(self.config.compress_factor)
                estimate_mag, estimate_phase, _ = self.model(magnitude, phase)
                estimate = torch.polar(
                    estimate_mag.clamp_min(0).pow(1 / self.config.compress_factor),
                    estimate_phase,
                )
                waveform = torch.istft(
                    estimate,
                    n_fft=self.config.n_fft,
                    hop_length=self.config.hop_size,
                    win_length=self.config.win_size,
                    window=window,
                    length=signal.shape[-1],
                ) / scale
                enhanced = waveform.squeeze(0).cpu().numpy()
            if not np.isfinite(enhanced).all():
                raise RuntimeError("Model produced non-finite audio")

        if output_path is None:
            output_dir = ROOT / ".runtime" / "outputs"
            output_dir.mkdir(parents=True, exist_ok=True)
            descriptor, destination = tempfile.mkstemp(
                prefix=source.stem + "_", suffix=".wav", dir=str(output_dir)
            )
            os.close(descriptor)
        else:
            destination = str(Path(output_path).expanduser().resolve())
            Path(destination).parent.mkdir(parents=True, exist_ok=True)
        sf.write(destination, enhanced, target_rate, subtype="FLOAT")
        return AudioResult(audio_path=destination, sample_rate=target_rate)


Model = DyTSwiGMamba
