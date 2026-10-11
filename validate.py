"""Exercise the OpenBench MCP Skill with a real model invocation."""

import json
import os
import select
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml


ROOT = Path(__file__).resolve().parent


def exchange(process, request, timeout=300):
    process.stdin.write(json.dumps(request) + "\n")
    process.stdin.flush()
    ready, _, _ = select.select([process.stdout], [], [], timeout)
    if not ready:
        raise TimeoutError("MCP server did not respond in {} seconds".format(timeout))
    line = process.stdout.readline()
    if not line:
        raise RuntimeError("MCP server exited before responding")
    response = json.loads(line)
    if response.get("id") != request["id"]:
        raise AssertionError("MCP response ID mismatch")
    if "error" in response:
        raise RuntimeError("MCP error: {}".format(response["error"]))
    return response["result"]


def main():
    with tempfile.TemporaryDirectory(prefix="openbench-se-") as temp:
        temp_dir = Path(temp)
        sample_rate = 16000
        time = np.arange(sample_rate, dtype=np.float32) / sample_rate
        noisy = 0.15 * np.sin(2 * np.pi * 220 * time)
        noisy += 0.02 * np.sin(2 * np.pi * 1040 * time)
        source = temp_dir / "noisy.wav"
        destination = temp_dir / "enhanced.wav"
        sf.write(str(source), noisy, sample_rate, subtype="FLOAT")

        env = os.environ.copy()
        with (ROOT / "model.spec.yaml").open(encoding="utf-8") as handle:
            spec = yaml.safe_load(handle)
        env.update(spec["server"].get("env", {}))
        env.setdefault("MODEL_DEVICE", "cuda")
        process = subprocess.Popen(
            [str(ROOT / spec["server"]["command"][0]),
             *spec["server"]["command"][1:]],
            cwd=str(ROOT),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        try:
            initialized = exchange(process, {
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                           "clientInfo": {"name": "openbench-validation", "version": "1.0"}},
            })
            assert initialized["capabilities"]["tools"] == {"listChanged": False}
            tools = exchange(process, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
            assert any(tool["name"] == "enhance_speech" for tool in tools["tools"])
            result = exchange(process, {
                "jsonrpc": "2.0", "id": 3, "method": "tools/call",
                "params": {"name": "enhance_speech", "arguments": {
                    "audio_path": str(source), "output_path": str(destination)}},
            })
            output = result["structuredContent"]
            assert output["audio_path"] == str(destination)
            assert output["sample_rate"] == sample_rate
            signal, rate = sf.read(str(destination), dtype="float32")
            assert rate == sample_rate and len(signal) == sample_rate
            assert np.isfinite(signal).all() and np.max(np.abs(signal)) > 0
            print(json.dumps({"ok": True, "audio_path": str(destination),
                              "sample_rate": rate, "samples": len(signal)}))
        finally:
            process.terminate()
            try:
                process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()


if __name__ == "__main__":
    main()
