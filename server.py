"""Minimal stdio MCP server for OpenBench speech enhancement."""

import contextlib
import json
import os
import sys
import traceback


TOOL = {
    "name": "enhance_speech",
    "description": "Enhance a noisy speech file and return the enhanced WAV path.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "audio_path": {"type": "string", "description": "Path to the input audio file"},
            "output_path": {"type": "string", "description": "Optional output WAV path"},
        },
        "required": ["audio_path"],
    },
    "outputSchema": {
        "type": "object",
        "properties": {
            "audio_path": {"type": "string"},
            "sample_rate": {"type": "integer"},
        },
        "required": ["audio_path", "sample_rate"],
    },
}


class Server:
    def __init__(self):
        self.model = None
        self.initialized = False

    def dispatch(self, request):
        if not isinstance(request, dict) or request.get("jsonrpc") != "2.0":
            return self.error(request.get("id") if isinstance(request, dict) else None,
                              -32600, "Invalid Request")
        method = request.get("method")
        request_id = request.get("id")
        if method == "initialize":
            params = request.get("params")
            if params is None:
                params = {}
            if not isinstance(params, dict):
                return self.error(request_id, -32602, "params must be an object")
            self.initialized = True
            requested_version = params.get("protocolVersion", "2024-11-05")
            return self.result(request_id, {
                "protocolVersion": requested_version,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "openbench-se", "version": "1.0.0"},
            })
        if method == "notifications/initialized" or method == "notifications/cancelled":
            return None
        if method == "ping":
            return self.result(request_id, {})
        if not self.initialized:
            return self.error(request_id, -32001, "Server not initialized")
        if method == "tools/list":
            return self.result(request_id, {"tools": [TOOL]})
        if method == "tools/call":
            params = request.get("params")
            if params is None:
                params = {}
            if not isinstance(params, dict):
                return self.error(request_id, -32602, "params must be an object")
            if params.get("name") != TOOL["name"]:
                return self.error(request_id, -32601, "Unknown tool")
            arguments = params.get("arguments") or {}
            if not isinstance(arguments, dict):
                return self.error(request_id, -32602, "arguments must be an object")
            if not isinstance(arguments.get("audio_path"), str):
                return self.error(request_id, -32602, "audio_path must be a string")
            if "output_path" in arguments and not isinstance(arguments["output_path"], str):
                return self.error(request_id, -32602, "output_path must be a string")
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    if self.model is None:
                        from model import Model

                        self.model = Model(
                            model_path=os.environ.get("MODEL_PATH"),
                            device=os.environ.get("MODEL_DEVICE", "auto"),
                        )
                    output = self.model.run_tool(
                        arguments["audio_path"], arguments.get("output_path")
                    )
                data = {"audio_path": output.audio_path, "sample_rate": output.sample_rate}
                return self.result(request_id, {
                    "content": [{"type": "text", "text": json.dumps(data)}],
                    "structuredContent": data,
                    "isError": False,
                })
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                return self.error(request_id, -32603, str(exc))
        return self.error(request_id, -32601, "Method not found")

    @staticmethod
    def result(request_id, value):
        return {"jsonrpc": "2.0", "id": request_id, "result": value}

    @staticmethod
    def error(request_id, code, message):
        return {"jsonrpc": "2.0", "id": request_id,
                "error": {"code": code, "message": message}}


def main():
    server = Server()
    for line in sys.stdin:
        try:
            response = server.dispatch(json.loads(line))
        except json.JSONDecodeError:
            response = server.error(None, -32700, "Parse error")
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
