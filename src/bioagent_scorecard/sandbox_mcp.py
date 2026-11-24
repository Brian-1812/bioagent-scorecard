"""A one-tool MCP server: `run_python`, backed by sandbox.run_python.

Row B starts this server for each task through `claude -p --mcp-config`.
Transport: MCP stdio (one JSON-RPC 2.0 message per line). Standard library only.

Usage:
    python3 -m bioagent_scorecard.sandbox_mcp --log calls.jsonl [--max-calls 10]

Each tool call appends one line to the log file: the code, its output, the
exit code and the time. The log stays in the local run folder (git ignores
runs/). It can hold benchmark text, because the model often pastes the
question's sequence into its code, so do not publish it as is.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time

from .sandbox import Limits, run_python

SERVER_NAME = "pysandbox"
TOOL_NAME = "run_python"
#: The name Claude Code gives the tool: mcp__<server>__<tool>.
CLAUDE_TOOL_NAME = f"mcp__{SERVER_NAME}__{TOOL_NAME}"

TOOL_DESCRIPTION = (
    "Run a Python 3 program and return its stdout and stderr. "
    "Only the Python standard library is available (no Biopython, no numpy). "
    "There is no network and no access to files outside the program. "
    "Each call starts a new process with no state from earlier calls. "
    "Limits per call: {wall:.0f} s wall clock, {cpu} s CPU, {mem} MB memory. "
    "Print the values you need."
)

DEFAULT_MAX_CALLS = 10


def tool_spec(limits: Limits) -> dict:
    return {
        "name": TOOL_NAME,
        "description": TOOL_DESCRIPTION.format(wall=limits.wall_seconds,
                                               cpu=limits.cpu_seconds, mem=limits.memory_mb),
        "inputSchema": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "The Python program."}},
            "required": ["code"],
        },
    }


def format_result(r) -> str:
    parts = [f"exit_code: {r.exit_code}"]
    if r.error:
        parts.append(f"error: {r.error}")
    parts.append("stdout:\n" + (r.stdout or ""))
    if r.stderr:
        parts.append("stderr:\n" + r.stderr)
    return "\n".join(parts)


class Server:
    def __init__(self, log_path: str | None, max_calls: int, limits: Limits,
                 runner=run_python):
        self.log_path = log_path
        self.max_calls = max_calls
        self.limits = limits
        self.runner = runner
        self.calls = 0

    def _log(self, rec: dict) -> None:
        if not self.log_path:
            return
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError as e:
            # A log problem must not kill the tool: the model would see a crash.
            print(f"sandbox_mcp: cannot write the log {self.log_path}: {e}", file=sys.stderr)

    def handle(self, msg: dict) -> dict | None:
        """Return the reply for one JSON-RPC message, or None for a notification."""
        method = msg.get("method")
        mid = msg.get("id")
        if mid is None:
            return None                     # Notifications need no reply.
        if method == "initialize":
            version = (msg.get("params") or {}).get("protocolVersion") or "2025-06-18"
            result = {"protocolVersion": version,
                      "capabilities": {"tools": {"listChanged": False}},
                      "serverInfo": {"name": SERVER_NAME, "version": "1"}}
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": [tool_spec(self.limits)]}
        elif method == "tools/call":
            result = self._call((msg.get("params") or {}))
        else:
            return {"jsonrpc": "2.0", "id": mid,
                    "error": {"code": -32601, "message": f"method not found: {method}"}}
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def _call(self, params: dict) -> dict:
        if params.get("name") != TOOL_NAME:
            return {"content": [{"type": "text", "text": f"unknown tool {params.get('name')!r}"}],
                    "isError": True}
        code = (params.get("arguments") or {}).get("code")
        self.calls += 1
        if not isinstance(code, str) or not code.strip():
            text, is_error, r = "error: the 'code' argument must be a non-empty string", True, None
        elif self.calls > self.max_calls:
            text, is_error, r = (f"error: the limit of {self.max_calls} tool calls for this "
                                 "question is reached; give your final answer now"), True, None
        else:
            r = self.runner(code, self.limits)
            text, is_error = format_result(r), bool(r.error) or r.exit_code != 0
        self._log({
            "n": self.calls, "time": time.time(),
            "code_sha256": hashlib.sha256((code or "").encode()).hexdigest() if isinstance(code, str) else None,
            "code": code if isinstance(code, str) else None,
            "result": r.as_dict() if r else None,
            "is_error": is_error, "reply_chars": len(text),
        })
        return {"content": [{"type": "text", "text": text}], "isError": is_error}


def serve(server: Server, stdin=None, stdout=None) -> None:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            reply = {"jsonrpc": "2.0", "id": None,
                     "error": {"code": -32700, "message": "parse error"}}
        else:
            reply = server.handle(msg) if isinstance(msg, dict) else None
        if reply is not None:
            stdout.write(json.dumps(reply) + "\n")
            stdout.flush()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sandbox_mcp")
    p.add_argument("--log")
    p.add_argument("--max-calls", type=int, default=DEFAULT_MAX_CALLS)
    a = p.parse_args(argv)
    serve(Server(a.log, a.max_calls, Limits()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
