"""Rows A and B through headless Claude Code (`claude -p`) on a Claude subscription.

Agent names:
- `claude-cli:<model>`      row A, plain model, no tools (the baseline).
- `claude-sandbox:<model>`  row B, model + one sandboxed Python tool
                            (`run_python`, see sandbox.py and sandbox_mcp.py).
The CLI accepts them only with the `--allow-model-calls` flag.

For each task the adapter runs one `claude -p` process in a new empty folder.
The MCQ prompt goes in on stdin. Both rows use the same flags. Row B adds only
the MCP server and the permission for its one tool:

    claude -p --model <model> --output-format stream-json --verbose
           --tools "" --restricted --setting-sources "" --strict-mcp-config
           --permission-mode dontAsk --no-session-persistence
           --disable-slash-commands --system-prompt <system prompt>
    row B: + --mcp-config <file> --allowedTools mcp__pysandbox__run_python

Why each flag:
- `--tools ""`: no built-in tools (no Bash, no file tools, no web).
- `--restricted` and `--setting-sources ""`: no user, project or local
  settings, so no CLAUDE.md, hooks, skills or plugins from settings.
  Claude Code's own built-in plugins still load (the init event lists them;
  the trace records the names), and Claude Code adds a short environment
  block and an account block to the context. Both rows get the same.
  (`--safe-mode`, used in an earlier version, also turns off `--mcp-config` servers, so
  row B cannot use it. Row A uses the same flags as row B, so the two rows
  differ only in the tool and the system prompt.)
- `--strict-mcp-config`: only the MCP server we pass (none for row A).
- `--permission-mode dontAsk` + `--allowedTools`: only `run_python` can run.
- `--output-format stream-json --verbose`: the stream has every tool call and
  a `rate_limit_event` with the plan's usage (five_hour and seven_day
  utilization). The runner uses it to stop a chunk in time.
- `--system-prompt`: replaces the Claude Code system prompt with a short fixed
  text, which we publish. Claude Code still adds a small environment block.

Billing: the child process gets no ANTHROPIC_* or CLAUDE* variables except the
subscription login, so Claude Code uses the logged-in subscription (for
example a Max plan). A run costs usage-limit quota, not dollars. The harness still
computes a $ figure from prices.json. It is the API list-price equivalent, not
money spent, and the agent config says so (`billing`). Any public result must
say so too.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

from ..mcq import MCQ
from .base import AgentResult, ToolCall

SYSTEM_PROMPT_NO_TOOLS = (
    "You answer multiple choice questions about biology. "
    "You have no tools: you cannot run code, search the web or open files. "
    "Work the answer out by reasoning. You can count, translate, compare and "
    "calculate step by step in your own text. "
    "Then give exactly one option letter in the answer format of the question. "
    "Choose the option that says the information is insufficient only when the "
    "question itself does not give enough information. Do not choose it because "
    "you have no tools."
)

SYSTEM_PROMPT_SANDBOX = (
    "You answer multiple choice questions about biology. "
    "You have one tool, run_python. It runs a Python 3 program with only the "
    "standard library, no network and no files, and returns its output. "
    "Use it for sequence work and calculations. "
    "Then give exactly one option letter in the answer format of the question. "
    "Choose the option that says the information is insufficient only when the "
    "question itself does not give enough information."
)

# Kept for older imports.
SYSTEM_PROMPT = SYSTEM_PROMPT_NO_TOOLS

#: Environment variables that would switch Claude Code from the subscription
#: to API billing. The adapter removes them from the child environment.
API_KEY_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")

#: The child environment drops every variable that can change billing, routing
#: or answers: all ANTHROPIC_* (keys, base URL, model), all CLAUDE* (provider
#: switches such as CLAUDE_CODE_USE_BEDROCK, CLAUDE_EFFORT, the parent
#: session's variables), MAX_THINKING_TOKENS and AWS_BEARER_TOKEN_BEDROCK.
#: Kept: the subscription login (CLAUDE_CODE_OAUTH_TOKEN, CLAUDE_CONFIG_DIR).
ENV_DROP_PREFIXES = ("ANTHROPIC_", "CLAUDE", "MAX_THINKING_TOKENS", "AWS_BEARER_TOKEN_BEDROCK")
ENV_KEEP = ("CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CONFIG_DIR")
ENV_RULE = ("drop ANTHROPIC_*, CLAUDE* (keep CLAUDE_CODE_OAUTH_TOKEN, CLAUDE_CONFIG_DIR), "
            "MAX_THINKING_TOKENS, AWS_BEARER_TOKEN_BEDROCK")

BILLING = ("claude-max-subscription: costs usage-limit quota, not dollars; "
           "cost_usd is the API list-price equivalent")

#: Text in a failed result that means the plan's usage limit is reached.
#: Not seen live yet: the exact wording is a guess from Claude Code messages.
USAGE_LIMIT_RE = re.compile(r"usage limit|hit your (usage )?limit|limit reached|"
                            r"rate[_ ]limit|out of (extra )?usage", re.IGNORECASE)

# (argv, stdin text, env, cwd, timeout seconds) -> (returncode, stdout, stderr)
Runner = Callable[[list[str], str, dict, str, float], tuple[int, str, str]]


def subprocess_runner(argv: list[str], stdin: str, env: dict, cwd: str,
                      timeout: float) -> tuple[int, str, str]:
    p = subprocess.run(argv, input=stdin, capture_output=True, text=True,
                       env=env, cwd=cwd, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def cli_version(binary: str = "claude") -> str:
    try:
        p = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=30)
        return p.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


class ClaudeCLIAgent:
    """Row A: plain model, no tools."""

    makes_api_calls = True
    row = "A"
    kind = "plain_model_no_tools"
    prefix = "claude-cli"
    system_prompt = SYSTEM_PROMPT_NO_TOOLS

    def __init__(self, model: str, binary: str = "claude", effort: str | None = None,
                 timeout: float = 600.0, runner: Runner | None = None,
                 version: str | None = None):
        self.model = model
        self.binary = binary
        self.effort = effort
        self.timeout = timeout
        self.runner = runner or subprocess_runner
        self._version = version
        self.name = f"{self.prefix}-{model}"
        #: Highest num_turns of a normal answer. Row A has one turn.
        self.max_turns = 1
        #: The runner sets this to <run folder>/tool_logs for agents with tools.
        self.tool_log_dir: str | None = None

    @property
    def version(self) -> str:
        if self._version is None:
            self._version = cli_version(self.binary)
        return self._version

    def argv(self, mcp_config: str | None = None) -> list[str]:
        a = [self.binary, "-p", "--model", self.model,
             "--output-format", "stream-json", "--verbose",
             "--tools", "", "--restricted", "--setting-sources", "",
             "--strict-mcp-config", "--permission-mode", "dontAsk",
             "--no-session-persistence", "--disable-slash-commands",
             "--system-prompt", self.system_prompt]
        if self.effort:
            a += ["--effort", self.effort]
        return a

    def expected_tools(self) -> list[str]:
        return []

    def config(self) -> dict:
        return {
            "agent": self.name,
            "kind": self.kind,
            "row": self.row,
            "transport": "claude-code-headless",
            "model": self.model,
            "effort": self.effort,
            "claude_code_version": self.version,
            "system_prompt": self.system_prompt,
            "argv": self.argv("<mcp.json>")[1:],
            "tools": self.expected_tools(),
            "billing": BILLING,
            "child_env_rule": ENV_RULE,
            "version": 3,
        }

    def child_env(self) -> dict:
        return {k: v for k, v in os.environ.items()
                if k in ENV_KEEP or not k.startswith(ENV_DROP_PREFIXES)}

    def _prepare(self, cwd: str, mcq: MCQ) -> str | None:
        """Write the files the call needs into cwd. Return the --mcp-config path."""
        return None

    def answer(self, mcq: MCQ, seed: int) -> AgentResult:
        # Sampling is not seeded; the seed only fixes the option order.
        with tempfile.TemporaryDirectory(prefix="bas-claude-cli-") as cwd:
            mcp = self._prepare(cwd, mcq)
            try:
                rc, out, err = self.runner(self.argv(mcp), mcq.prompt, self.child_env(),
                                           cwd, self.timeout)
            except subprocess.TimeoutExpired:
                return AgentResult(text=None, model=self.model,
                                   error=f"timeout after {self.timeout:.0f}s")
        return parse_cli_output(out, rc, err, self.model, max_turns=self.max_turns,
                                expected_tools=self.expected_tools())


class ClaudeSandboxAgent(ClaudeCLIAgent):
    """Row B: model + one sandboxed Python tool (`run_python`) through MCP."""

    row = "B"
    kind = "model_plus_python_sandbox"
    prefix = "claude-sandbox"
    system_prompt = SYSTEM_PROMPT_SANDBOX

    def __init__(self, model: str, max_tool_calls: int | None = None, **kw):
        super().__init__(model, **kw)
        from ..sandbox_mcp import DEFAULT_MAX_CALLS
        self.max_tool_calls = max_tool_calls or DEFAULT_MAX_CALLS
        # One turn per tool call, one for the answer, one spare.
        self.max_turns = self.max_tool_calls + 2

    def expected_tools(self) -> list[str]:
        from ..sandbox_mcp import CLAUDE_TOOL_NAME
        return [CLAUDE_TOOL_NAME]

    def preflight(self) -> None:
        """Refuse to start a row-B run when the sandbox does not work here.
        Without it every tool call fails and the run would not be row B."""
        from ..sandbox import isolation_available, run_python
        ok, why = isolation_available()
        if not ok:
            raise SystemExit(f"row B: sandbox unavailable on this host: {why}")
        r = run_python("print(6 * 7)")
        if r.stdout.strip() != "42" or r.error:
            raise SystemExit(f"row B: sandbox self-test failed: {r.error or r.stderr[-200:]}")

    def argv(self, mcp_config: str | None = None) -> list[str]:
        from ..sandbox_mcp import CLAUDE_TOOL_NAME
        a = super().argv(mcp_config)
        return a + ["--mcp-config", mcp_config or "mcp.json", "--allowedTools", CLAUDE_TOOL_NAME]

    def config(self) -> dict:
        from ..sandbox import Limits
        from ..sandbox_mcp import tool_spec
        cfg = super().config()
        cfg["tool"] = {"spec": tool_spec(Limits()), "limits": Limits().as_dict(),
                       "max_calls_per_task": self.max_tool_calls,
                       "isolation": ("systemd scope per call (MemoryMax, MemorySwapMax=0, "
                                     "TasksMax, CPUQuota); unshare net+mount+pid+ipc+uts; "
                                     "allowlisted root (read-only /usr, /tmp, /proc, 4 /dev "
                                     "nodes, 2-file /etc, read-only /sbx) via pivot_root; "
                                     "uid nobody, no capabilities, no_new_privs, new session "
                                     "keyring; rlimits AS/CPU/FSIZE/NPROC; output cap; "
                                     "dies with its parent; no seccomp"),
                       "python": sys.version.split()[0], "packages": "standard library only"}
        return cfg

    def _prepare(self, cwd: str, mcq: MCQ) -> str:
        # Absolute: the MCP server runs in the task's temp folder (cwd), so a
        # relative run folder such as runs/pilot would point to the wrong place.
        log_dir = Path(self.tool_log_dir or cwd).resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        log = log_dir / f"{mcq.task_id}.jsonl"
        if log.exists():
            log.unlink()                    # A retried task starts a new log.
        src = str(Path(__file__).resolve().parents[2])      # The folder with the package.
        cfg = {"mcpServers": {"pysandbox": {
            "type": "stdio", "command": sys.executable,
            "args": ["-m", "bioagent_scorecard.sandbox_mcp", "--log", str(log),
                     "--max-calls", str(self.max_tool_calls)],
            "env": {"PYTHONPATH": src}}}}
        path = Path(cwd) / "mcp.json"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        return str(path)


def _rate_limit(info: dict) -> dict:
    """Keep the fields of a rate_limit_event that the runner needs."""
    out = {"status": info.get("status"), "type": info.get("rateLimitType"),
           "is_using_overage": bool(info.get("isUsingOverage"))}
    for name, w in (info.get("unifiedWindows") or {}).items():
        if isinstance(w, dict):
            out[name] = {"utilization": w.get("utilization"), "resets_at": w.get("resetsAt"),
                         "status": w.get("status")}
    return out


def read_stream(stdout: str) -> tuple[dict | None, list[dict]]:
    """Split `claude -p` output into (result object, other events).

    It reads both `--output-format json` (one object) and `stream-json`
    (one event per line; the result event has type "result")."""
    text = (stdout or "").strip()
    if not text:
        return None, []
    try:
        one = json.loads(text)
        if isinstance(one, dict):
            return one, []
    except json.JSONDecodeError:
        pass
    result, events = None, []
    for line in text.splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(d, dict):
            continue
        if d.get("type") == "result":
            result = d
        else:
            events.append(d)
    return result, events


def _is_usage_limit(rate: dict, text: str) -> bool:
    if rate.get("status") == "rejected":
        return True
    if rate.get("is_using_overage"):
        return True     # Overage costs money: a run may use quota, not spend.
    if any(isinstance(w, dict) and w.get("status") == "rejected" for w in rate.values()):
        return True
    return bool(text and USAGE_LIMIT_RE.search(text))


def parse_cli_output(stdout: str, returncode: int, stderr: str, model: str,
                     max_turns: int = 1, expected_tools: list[str] | None = None) -> AgentResult:
    """Turn the `claude -p` output (json or stream-json) into an AgentResult."""
    d, events = read_stream(stdout)
    rate: dict = {}
    init: dict = {}
    calls: dict[str, ToolCall] = {}
    for e in events:
        kind = e.get("type")
        if kind == "rate_limit_event":
            rate = _rate_limit(e.get("rate_limit_info") or {})
        elif kind == "system" and e.get("subtype") == "init":
            init = e
        elif kind == "assistant":
            for c in (e.get("message") or {}).get("content") or []:
                if isinstance(c, dict) and c.get("type") == "tool_use":
                    code = str((c.get("input") or {}).get("code", ""))
                    calls[str(c.get("id", len(calls)))] = ToolCall(
                        name=str(c.get("name")),
                        args_summary=(f"code {len(code)} chars, sha256 "
                                      f"{hashlib.sha256(code.encode()).hexdigest()[:12]}"))
        elif kind == "user":
            content = (e.get("message") or {}).get("content")
            for c in content if isinstance(content, list) else []:
                if isinstance(c, dict) and c.get("type") == "tool_result":
                    tc = calls.get(str(c.get("tool_use_id")))
                    if tc and c.get("is_error"):
                        tc.ok = False
                        tc.error = str(c.get("content"))[:200]
    if d is None:
        tail = (stderr or stdout or "").strip()[-300:]
        res = AgentResult(text=None, model=model, rate_limit=rate,
                          error=f"claude exit {returncode}, no JSON result: {tail}")
        res.usage_limit_hit = _is_usage_limit(rate, tail)
        return res
    res = parse_result(d, returncode, stderr, model, max_turns)
    res.rate_limit = rate
    res.tool_calls = list(calls.values())
    t = res.trace[0]
    t["tool_calls"] = len(calls)
    t["tool_errors"] = sum(not c.ok for c in calls.values())
    if expected_tools is not None and not init and (events or expected_tools):
        # stream-json without an init event (or row B with no stream at all):
        # the tool set is unknown, so this is not a valid row-A or row-B answer.
        res.error = res.error or "no init event in the stream; the tool set is unknown"
    if init:
        tools = [str(x) for x in init.get("tools") or []]
        t["init_tools"] = tools
        t["init_mcp_servers"] = [f"{s.get('name')}:{s.get('status')}"
                                 for s in init.get("mcp_servers") or [] if isinstance(s, dict)]
        t["init_plugins"] = sorted(str(p.get("name")) for p in init.get("plugins") or []
                                   if isinstance(p, dict))
        if expected_tools is not None and sorted(tools) != sorted(expected_tools):
            # The tool set is what the row measures. Another set is not this row.
            res.error = res.error or f"tools {tools}, expected {expected_tools}"
    # A rejected window means the plan refused or will refuse calls: stop the run,
    # even if this one call returned an answer (the next chunk retries the task).
    if _is_usage_limit(rate, "") or (res.error and _is_usage_limit(rate, res.error)):
        res.usage_limit_hit = True
    return res


def parse_result(d: dict, returncode: int, stderr: str, model: str,
                 max_turns: int = 1) -> AgentResult:
    """Turn the final `result` object of `claude -p` into an AgentResult."""
    usage = d.get("usage") or {}
    fresh = int(usage.get("input_tokens") or 0)
    cache_write = int(usage.get("cache_creation_input_tokens") or 0)
    cache_read = int(usage.get("cache_read_input_tokens") or 0)
    trace = [{
        "type": "claude_cli_result",
        "subtype": d.get("subtype"),
        "stop_reason": d.get("stop_reason"),
        "num_turns": d.get("num_turns"),
        "duration_ms": d.get("duration_ms"),
        "duration_api_ms": d.get("duration_api_ms"),
        "cli_cost_usd": d.get("total_cost_usd"),
        "cache_creation_input_tokens": cache_write,
        "cache_read_input_tokens": cache_read,
        "thinking_tokens": (usage.get("output_tokens_details") or {}).get("thinking_tokens"),
        # Only the count: a denial holds the tool input, which can hold task text.
        "permission_denials": len(d.get("permission_denials") or []),
        "safety_stops": d.get("safety_stops"),
        "models_used": sorted((d.get("modelUsage") or {}).keys()),
    }]
    res = AgentResult(
        text=d.get("result"),
        # Token contract (base.py): input_tokens includes cache reads and writes.
        # Each task is a new process, so most input is a cache write, which
        # costs 2x input (1-hour cache). pricing.py uses the cache_write rate.
        input_tokens=fresh + cache_write + cache_read,
        output_tokens=int(usage.get("output_tokens") or 0),
        cached_input_tokens=cache_read,
        cache_write_tokens=cache_write,
        model=model,
        trace=trace,
    )
    if d.get("is_error") or d.get("subtype") != "success" or returncode != 0:
        status = d.get("api_error_status")
        res.error = (f"claude exit {returncode}, subtype {d.get('subtype')!r}"
                     + (f", api status {status}" if status else "")
                     + f": {str(d.get('result') or stderr or '')[:300]}")
        return res
    used = d.get("modelUsage") or {}
    if used and set(used) != {model}:
        # Another model answered all or part (a fallback or a helper call).
        # Do not credit the answer, a refusal or the tokens to --model.
        res.error = f"answer used models {sorted(used)}, not only {model}"
        return res
    if d.get("stop_reason") == "refusal":
        # Not seen live yet: the field names for the category are a guess.
        res.safety_refusal = str(d.get("stop_details") or d.get("refusal_category") or "refusal")
        return res          # A safety refusal is its own outcome, not an agent error.
    if (d.get("num_turns") or 1) > max_turns:
        # Row A has one turn. More turns means a tool loop ran.
        res.error = (f"expected at most {max_turns} turn(s), "
                     f"got {d.get('num_turns')} (tools active?)")
    return res


def parse_cli_json(stdout: str, returncode: int, stderr: str, model: str) -> AgentResult:
    """Row-A parser for one `--output-format json` object (kept for old callers)."""
    return parse_cli_output(stdout, returncode, stderr, model, max_turns=1)
