"""Offline tests for row B (sandboxed Python tool through
MCP), the stream-json parser, chunk and usage-limit stops, resume, stratified
selection, the SHA-256 pin, answer-text stripping and the anchor scale.

No model calls. The real-sandbox tests run only as root (they build Linux
namespaces); CI runs them with sudo in a separate step.

Run: PYTHONPATH=src python3 -m unittest discover -s tests -v
"""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bioagent_scorecard import anchors, sandbox, sandbox_mcp
from bioagent_scorecard.agents import get_agent
from bioagent_scorecard.agents.claude_cli import (ClaudeCLIAgent, ClaudeSandboxAgent,
                                                  SYSTEM_PROMPT_NO_TOOLS, parse_cli_output)
from bioagent_scorecard.cli import EXIT_STOPPED, main as cli_main
from bioagent_scorecard.mcq import build_mcq
from bioagent_scorecard.runner import ChunkLimits, UsageMeter, run, strip_task_text
from bioagent_scorecard.selection import stratified
from bioagent_scorecard.tasks import Task, load_jsonl, load_meta, pin_status

FIX = Path(__file__).parent / "fixtures"
SYNTH = FIX / "synthetic.jsonl"
STREAM = (FIX / "claude_sandbox_stream.jsonl").read_text()


def task(tid, subtask, question="Q?"):
    return Task(id=tid, benchmark="synthetic", subset="S", subtask=subtask,
                question=question, ideal="A1", distractors=("B1", "C1"))


class StreamRunner:
    """A fake `claude -p` that returns a recorded stream (optionally edited)."""

    def __init__(self, stream=STREAM, rc=0):
        self.stream, self.rc, self.calls = stream, rc, []

    def __call__(self, argv, stdin, env, cwd, timeout):
        mcp = None
        if "--mcp-config" in argv:
            mcp = json.loads(Path(argv[argv.index("--mcp-config") + 1]).read_text())
        self.calls.append({"argv": argv, "stdin": stdin, "cwd": cwd, "mcp": mcp})
        return self.rc, self.stream, ""


def edit_stream(fn):
    lines = [json.loads(x) for x in STREAM.splitlines() if x.strip()]
    for d in lines:
        fn(d)
    return "\n".join(json.dumps(d) for d in lines)


def set_five_hour(util, resets=1795268400, status="allowed"):
    def fn(d):
        if d["type"] == "rate_limit_event":
            d["rate_limit_info"]["unifiedWindows"]["five_hour"] = {
                "utilization": util, "resetsAt": resets, "status": status}
    return fn


class TestStreamParser(unittest.TestCase):
    def test_sandbox_fixture(self):
        r = parse_cli_output(STREAM, 0, "", "claude-haiku-5-5", max_turns=12,
                             expected_tools=["mcp__pysandbox__run_python"])
        self.assertEqual(r.error, "")
        self.assertEqual(r.text, "6*7 printed 42.\n\n[ANSWER]B[/ANSWER]")
        self.assertEqual(len(r.tool_calls), 1)
        self.assertEqual(r.tool_calls[0].name, "mcp__pysandbox__run_python")
        self.assertTrue(r.tool_calls[0].ok)
        self.assertEqual(r.rate_limit["five_hour"]["utilization"], 0.05)
        self.assertEqual(r.rate_limit["seven_day"]["utilization"], 0.87)
        self.assertEqual(r.input_tokens, 4 + 1881 + 1235)
        self.assertFalse(r.usage_limit_hit)

    def test_row_a_rejects_a_stream_with_tools(self):
        r = parse_cli_output(STREAM, 0, "", "claude-haiku-5-5", max_turns=1, expected_tools=[])
        self.assertTrue(r.error)
        r = parse_cli_output(STREAM, 0, "", "claude-haiku-5-5", max_turns=5, expected_tools=[])
        self.assertIn("expected []", r.error)

    def test_rejected_window_is_a_usage_limit(self):
        s = edit_stream(set_five_hour(1.0, status="rejected"))
        r = parse_cli_output(s, 0, "", "claude-haiku-5-5", max_turns=12)
        self.assertTrue(r.usage_limit_hit)

    def test_limit_error_text_is_a_usage_limit(self):
        def fn(d):
            if d["type"] == "result":
                d.update(is_error=True, subtype="success",
                         result="You've hit your limit · resets 7pm")
        r = parse_cli_output(edit_stream(fn), 1, "", "claude-haiku-5-5", max_turns=12)
        self.assertTrue(r.error)
        self.assertTrue(r.usage_limit_hit)

    def test_overage_stops(self):
        def fn(d):
            if d["type"] == "rate_limit_event":
                d["rate_limit_info"]["isUsingOverage"] = True
        r = parse_cli_output(edit_stream(fn), 0, "", "claude-haiku-5-5", max_turns=12)
        self.assertTrue(r.usage_limit_hit)

    def test_stream_without_init_is_an_error(self):
        s = "\n".join(x for x in STREAM.splitlines() if '"subtype": "init"' not in x)
        r = parse_cli_output(s, 0, "", "claude-haiku-5-5", max_turns=12, expected_tools=[])
        self.assertIn("no init event", r.error)

    def test_no_result_with_limit_text(self):
        r = parse_cli_output("", 1, "Claude AI usage limit reached", "claude-haiku-5-5")
        self.assertTrue(r.usage_limit_hit)

    def test_plain_error_is_not_a_usage_limit(self):
        r = parse_cli_output("", 1, "segfault", "claude-haiku-5-5")
        self.assertFalse(r.usage_limit_hit)


class TestRowB(unittest.TestCase):
    def setUp(self):
        self.mcq = build_mcq(load_jsonl(SYNTH)[0], 0)

    def test_argv_and_mcp_config(self):
        fake = StreamRunner()
        a = ClaudeSandboxAgent("claude-haiku-5-5", runner=fake, version="test")
        a.tool_log_dir = tempfile.mkdtemp()
        r = a.answer(self.mcq, 0)
        self.assertEqual(r.error, "")
        argv = fake.calls[0]["argv"]
        self.assertEqual(argv[argv.index("--tools") + 1], "")       # No built-in tools.
        self.assertEqual(argv[argv.index("--allowedTools") + 1], "mcp__pysandbox__run_python")
        self.assertIn("--strict-mcp-config", argv)
        srv = fake.calls[0]["mcp"]["mcpServers"]["pysandbox"]
        self.assertEqual(srv["args"][:2], ["-m", "bioagent_scorecard.sandbox_mcp"])
        self.assertTrue(srv["args"][srv["args"].index("--log") + 1].startswith(a.tool_log_dir))

    def test_relative_log_dir_becomes_absolute(self):        # A smoke run found this: the server crashed.
        fake = StreamRunner()
        a = ClaudeSandboxAgent("claude-haiku-5-5", runner=fake, version="test")
        a.tool_log_dir = "runs/x/tool_logs"
        with tempfile.TemporaryDirectory() as d, contextlib.chdir(d) if hasattr(contextlib, "chdir") else contextlib.nullcontext():
            a.answer(self.mcq, 0)
        args = fake.calls[0]["mcp"]["mcpServers"]["pysandbox"]["args"]
        self.assertTrue(os.path.isabs(args[args.index("--log") + 1]))

    def test_rows_differ_only_in_tool_and_prompt(self):
        a = ClaudeCLIAgent("m", version="t").argv()
        b = ClaudeSandboxAgent("m", version="t").argv("x.json")
        ia, ib = a.index("--system-prompt"), b.index("--system-prompt")
        self.assertEqual(a[:ia], b[:ib])
        self.assertEqual(b[ib + 2:], ["--mcp-config", "x.json", "--allowedTools",
                                      "mcp__pysandbox__run_python"])

    def test_config_records_the_sandbox(self):
        cfg = get_agent("claude-sandbox:claude-haiku-5-5", version="t").config()
        self.assertEqual(cfg["row"], "B")
        self.assertIn("nobody", cfg["tool"]["isolation"])
        self.assertIn("standard library", cfg["tool"]["packages"])
        self.assertEqual(cfg["tools"], ["mcp__pysandbox__run_python"])

    def test_no_tools_prompt(self):
        self.assertIn("You have no tools", SYSTEM_PROMPT_NO_TOOLS)
        self.assertIn("exactly one option letter", SYSTEM_PROMPT_NO_TOOLS)
        self.assertIn("Do not choose it because you have no tools", SYSTEM_PROMPT_NO_TOOLS)
        # The refusal option itself stays in every question.
        self.assertIn("Insufficient information", self.mcq.prompt)


class FakeSandbox:
    def __init__(self):
        self.codes = []

    def __call__(self, code, limits):
        self.codes.append(code)
        return sandbox.SandboxResult("42\n", "", 0, False, 0.01, "namespaces")


class TestMCPServer(unittest.TestCase):
    def rpc(self, msgs, max_calls=2):
        log = Path(tempfile.mkdtemp()) / "calls.jsonl"
        fake = FakeSandbox()
        srv = sandbox_mcp.Server(str(log), max_calls, sandbox.Limits(), runner=fake)
        out = io.StringIO()
        sandbox_mcp.serve(srv, io.StringIO("\n".join(json.dumps(m) for m in msgs) + "\n"), out)
        return [json.loads(x) for x in out.getvalue().splitlines()], log, fake

    def call(self, i, code="print(6*7)"):
        return {"jsonrpc": "2.0", "id": i, "method": "tools/call",
                "params": {"name": "run_python", "arguments": {"code": code}}}

    def test_handshake_and_list(self):
        replies, _, _ = self.rpc([
            {"jsonrpc": "2.0", "id": 0, "method": "initialize",
             "params": {"protocolVersion": "2025-06-18", "capabilities": {}}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 2, "method": "nope"}])
        self.assertEqual(len(replies), 3)                    # No reply to a notification.
        self.assertEqual(replies[0]["result"]["protocolVersion"], "2025-06-18")
        tools = replies[1]["result"]["tools"]
        self.assertEqual([t["name"] for t in tools], ["run_python"])
        self.assertEqual(tools[0]["inputSchema"]["required"], ["code"])
        self.assertEqual(replies[2]["error"]["code"], -32601)

    def test_call_limit_and_log(self):
        replies, log, fake = self.rpc([self.call(1), self.call(2), self.call(3)], max_calls=2)
        self.assertIn("42", replies[0]["result"]["content"][0]["text"])
        self.assertFalse(replies[0]["result"]["isError"])
        self.assertTrue(replies[2]["result"]["isError"])
        self.assertIn("limit of 2 tool calls", replies[2]["result"]["content"][0]["text"])
        self.assertEqual(len(fake.codes), 2)                 # The third call did not run.
        lines = [json.loads(x) for x in log.read_text().splitlines()]
        self.assertEqual([x["n"] for x in lines], [1, 2, 3])
        self.assertEqual(lines[0]["code"], "print(6*7)")

    def test_log_error_does_not_kill_the_tool(self):
        srv = sandbox_mcp.Server("/nonexistent/dir/calls.jsonl", 2, sandbox.Limits(),
                                 runner=FakeSandbox())
        with contextlib.redirect_stderr(io.StringIO()):
            reply = srv.handle(self.call(1))
        self.assertFalse(reply["result"]["isError"])

    def test_bad_arguments(self):
        replies, _, fake = self.rpc([self.call(1, code="")])
        self.assertTrue(replies[0]["result"]["isError"])
        self.assertEqual(fake.codes, [])


class TestSandboxFailsClosed(unittest.TestCase):
    def test_not_root_does_not_run(self):
        with mock.patch.object(sandbox.os, "geteuid", return_value=1000), \
             mock.patch.object(sandbox.subprocess, "Popen") as sp:
            r = sandbox.run_python("print(1)")
        sp.assert_not_called()                  # Popen, not run.
        self.assertEqual(r.isolation, "unavailable")
        self.assertIn("did not run", r.error)

    def test_argv_builds_all_namespaces(self):
        argv = sandbox.build_argv("/tmp/w", "/usr/bin/python3", sandbox.Limits())
        for flag in ("--net", "--mount", "--pid", "--ipc", "--uts", "--fork", "--kill-child"):
            self.assertIn(flag, argv)                         # --ipc, --uts.
        self.assertEqual(argv[argv.index("--propagation") + 1], "private")
        # One cgroup per call caps all its processes together.
        self.assertEqual(argv[:3], ["systemd-run", "--scope", "--quiet"])
        for prop in ("MemoryMax=1024M", "MemorySwapMax=0", "TasksMax=64", "CPUQuota=100%"):
            self.assertIn(prop, argv)
        script = argv[argv.index("-c") + 1]
        self.assertIn("--reuid=65534", script)
        self.assertIn("--no-new-privs", script)
        self.assertIn("pivot_root . .old", script)
        self.assertIn("umount -l /.old", script)
        self.assertIn("remount,bind,ro,nosuid,nodev /mnt/sbx", script)

    def test_preflight_refuses_without_sandbox(self):
        a = ClaudeSandboxAgent("claude-haiku-5-5", runner=StreamRunner(), version="t")
        with mock.patch.object(sandbox.os, "geteuid", return_value=1000), \
             self.assertRaises(SystemExit):
            run(a, load_jsonl(SYNTH), load_meta(SYNTH), tempfile.mkdtemp())


@unittest.skipUnless(sandbox.isolation_available()[0], "needs root, unshare and setpriv")
class TestRealSandbox(unittest.TestCase):
    L = sandbox.Limits(wall_seconds=15, cpu_seconds=2, memory_mb=512)

    def run_code(self, code):
        return sandbox.run_python(code, self.L)

    def test_hello(self):
        r = self.run_code("print(2 + 2)")
        self.assertEqual((r.stdout, r.exit_code, r.isolation), ("4\n", 0, "namespaces"))

    def test_runs_as_nobody_with_hidden_host(self):
        r = self.run_code("import os; print(os.getuid(), os.path.exists('/root'), os.path.exists('/srv'))")
        self.assertEqual(r.stdout.strip(), "65534 False False")

    def test_no_network(self):
        r = self.run_code("import socket\n"
                          "try:\n socket.create_connection(('1.1.1.1', 53), timeout=3); print('OPEN')\n"
                          "except OSError as e: print('BLOCKED')")
        self.assertEqual(r.stdout.strip(), "BLOCKED")

    def test_work_folder_is_read_only(self):
        r = self.run_code("open('x', 'w').write('a')")
        self.assertNotEqual(r.exit_code, 0)
        self.assertIn("Read-only file system", r.stderr)

    def test_cpu_limit(self):
        r = self.run_code("while True: pass")
        self.assertIn("CPU time limit", r.error)

    def test_memory_limit(self):
        r = self.run_code("x = bytearray(2 * 1024 ** 3)")
        self.assertIn("MemoryError", r.stderr)

    def test_only_the_allowlist_is_visible(self):
        r = self.run_code("import os; print(sorted(os.listdir('/etc')), os.path.exists('/sys/class'), "
                          "os.path.exists('/root'), os.listdir('/usr/local'))")
        self.assertEqual(r.stdout.strip(), "['ld.so.cache', 'passwd'] False False []")

    def test_own_ipc_and_uts_namespaces(self):
        r = self.run_code("import os; print(os.readlink('/proc/self/ns/ipc'), "
                          "os.readlink('/proc/self/ns/uts'))")
        ipc, uts = r.stdout.split()
        self.assertNotEqual(ipc, os.readlink("/proc/self/ns/ipc"))
        self.assertNotEqual(uts, os.readlink("/proc/self/ns/uts"))

    def test_total_memory_of_all_processes_is_capped(self):
        code = ("import os, time\nkids = []\nfor i in range(4):\n    pid = os.fork()\n"
                "    if pid == 0:\n        b = bytearray(400 * 1024 * 1024)\n"
                "        for j in range(0, len(b), 4096): b[j] = 1\n        time.sleep(3)\n"
                "        os._exit(0)\n    kids.append(pid)\n"
                "print([os.waitpid(k, 0)[1] for k in kids])")
        r = self.run_code(code)            # 4 x 400 MB > the 512 MB cap of the call.
        self.assertTrue(r.error or r.exit_code != 0 or "[0, 0, 0, 0]" not in r.stdout)

    def test_fork_bomb_stops(self):
        r = self.run_code("import os\nn = 0\ntry:\n    for i in range(500):\n"
                          "        if os.fork() == 0:\n            import time; time.sleep(2); os._exit(0)\n"
                          "        n += 1\nexcept OSError:\n    pass\nprint(n)")
        self.assertLess(int(r.stdout.strip() or 0), 64)

    def test_output_is_capped(self):
        r = self.run_code("import sys\ns = 'x' * 1000000\nwhile True: sys.stdout.write(s)")
        self.assertLessEqual(len(r.stdout), sandbox.OUTPUT_LIMIT + 100)
        self.assertIn("output cut", r.stdout)

    def test_sandbox_dies_with_its_parent(self):
        import signal, subprocess, sys, time
        def nobody():
            out = subprocess.run(["ps", "-u", "65534", "-o", "pid="], capture_output=True, text=True)
            return set(out.stdout.split())
        before = nobody()
        child = subprocess.Popen([sys.executable, "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); from bioagent_scorecard import sandbox; "
            "sandbox.run_python('import time; time.sleep(60)', sandbox.Limits(wall_seconds=60))",
            str(Path(sandbox.__file__).parents[1])])
        new = set()
        for _ in range(50):                     # Wait until the sandbox runs.
            time.sleep(0.2)
            new = nobody() - before
            if new:
                break
        self.assertTrue(new, "the sandbox did not start")
        child.send_signal(signal.SIGKILL)
        child.wait()
        time.sleep(2)
        self.assertEqual(nobody() & new, set())

    def test_own_session_keyring(self):
        code = ("import ctypes\nlibc = ctypes.CDLL(None)\n"
                "print(libc.syscall(250, 0, -3, 0))")   # KEYCTL_GET_KEYRING_ID(SESSION, no create)
        r = self.run_code(code)
        mine = ctypes_session_keyring()
        if mine is not None:
            self.assertNotEqual(r.stdout.strip(), str(mine))

    def test_host_mounts_unchanged(self):
        before = Path("/proc/self/mounts").read_text()
        self.run_code("print(1)")
        self.assertEqual(before, Path("/proc/self/mounts").read_text())


def ctypes_session_keyring():
    import ctypes
    if os.uname().machine != "x86_64":
        return None
    v = ctypes.CDLL(None).syscall(250, 0, -3, 0)
    return v if v > 0 else None


class TestUsageMeter(unittest.TestCase):
    def rl(self, u, resets=100, sd=0.5):
        return {"five_hour": {"utilization": u, "resets_at": resets},
                "seven_day": {"utilization": sd}}

    def test_stops_after_the_share(self):
        m = UsageMeter(ChunkLimits(max_five_hour=0.10))
        self.assertIsNone(m.update(self.rl(0.30)))     # Start value.
        self.assertIsNone(m.update(self.rl(0.35)))
        self.assertEqual(m.update(self.rl(0.40)), "chunk_limit_five_hour")
        self.assertAlmostEqual(m.used, 0.10)

    def test_new_window_counts_from_zero(self):
        m = UsageMeter(ChunkLimits(max_five_hour=0.10))
        m.update(self.rl(0.95, resets=100))
        self.assertIsNone(m.update(self.rl(0.02, resets=200)))
        self.assertAlmostEqual(m.used, 0.02)

    def test_seven_day_and_task_caps(self):
        self.assertEqual(UsageMeter(ChunkLimits(stop_seven_day=0.9)).update(self.rl(0.1, sd=0.91)),
                         "seven_day_limit")
        m = UsageMeter(ChunkLimits(max_tasks=2))
        self.assertIsNone(m.update({}))
        self.assertEqual(m.update({}), "chunk_limit_tasks")


class TestRunnerStops(unittest.TestCase):
    def agent(self, streams):
        """A row-B agent whose fake CLI returns one stream per call."""
        it = iter(streams)

        def runner(argv, stdin, env, cwd, timeout):
            return 0, next(it), ""
        a = ClaudeSandboxAgent("claude-haiku-5-5", runner=runner, version="test")
        a.preflight = lambda: None          # The real check needs root (TestRealSandbox).
        return a

    def test_usage_limit_stops_and_keeps_the_task(self):
        ok = STREAM
        limit = edit_stream(set_five_hour(1.0, status="rejected"))
        out = run(self.agent([ok, ok, limit]), load_jsonl(SYNTH), load_meta(SYNTH), tempfile.mkdtemp())
        m = json.loads((out / "manifest.json").read_text())
        self.assertEqual((m["status"], m["done"], m["total"]), ("stopped", 2, 6))
        self.assertEqual(m["stop_reason"], "usage_limit_error")
        self.assertEqual(len((out / "results.jsonl").read_text().splitlines()), 2)
        # Resume runs the other 4, including the one the limit stopped.
        out2 = run(self.agent([ok] * 4), load_jsonl(SYNTH), load_meta(SYNTH), "unused",
                   resume_dir=out)
        m = json.loads((out2 / "manifest.json").read_text())
        self.assertEqual((m["status"], m["done"], len(m["chunks"])), ("complete", 6, 2))
        ids = [json.loads(x)["task_id"] for x in (out / "results.jsonl").read_text().splitlines()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_chunk_share_stops(self):
        streams = [edit_stream(set_five_hour(u)) for u in (0.10, 0.15, 0.21, 0.30)]
        out = run(self.agent(streams), load_jsonl(SYNTH), load_meta(SYNTH), tempfile.mkdtemp(),
                  chunk=ChunkLimits(max_five_hour=0.10))
        m = json.loads((out / "manifest.json").read_text())
        self.assertEqual((m["done"], m["stop_reason"]), (3, "chunk_limit_five_hour"))

    def test_resume_refuses_a_changed_task_file(self):
        out = run(self.agent([STREAM] * 6), load_jsonl(SYNTH), load_meta(SYNTH), tempfile.mkdtemp(),
                  chunk=ChunkLimits(max_tasks=1))
        meta = {**load_meta(SYNTH), "sha256_now": "0" * 64}
        with self.assertRaises(SystemExit):
            run(self.agent([STREAM] * 6), load_jsonl(SYNTH), meta, "unused", resume_dir=out)

    def test_resume_refuses_other_config(self):
        out = run(self.agent([STREAM] * 6), load_jsonl(SYNTH), load_meta(SYNTH), tempfile.mkdtemp(),
                  chunk=ChunkLimits(max_tasks=1))
        other = ClaudeCLIAgent("claude-haiku-5-5", runner=lambda *a: (0, STREAM, ""), version="t")
        with self.assertRaises(SystemExit):
            run(other, load_jsonl(SYNTH), load_meta(SYNTH), "unused", resume_dir=out)

    def test_cli_exit_code_when_stopped(self):
        limit = edit_stream(set_five_hour(1.0, status="rejected"))
        fake = mock.patch("bioagent_scorecard.agents.claude_cli.subprocess_runner",
                          side_effect=lambda *a: (0, limit, ""))
        # Hermetic: no real `claude --version`, no real sandbox check.
        ver = mock.patch("bioagent_scorecard.agents.claude_cli.cli_version", return_value="test")
        pre = mock.patch.object(ClaudeSandboxAgent, "preflight", lambda self: None)
        buf = io.StringIO()
        with fake, ver, pre, contextlib.redirect_stdout(buf):
            rc = cli_main(["run", "--agent", "claude-sandbox:claude-haiku-5-5", "--allow-model-calls",
                           "--tasks", str(SYNTH), "--out", tempfile.mkdtemp()])
        self.assertEqual(rc, EXIT_STOPPED)
        self.assertIn("STOPPED (usage_limit_error) at 0/6", buf.getvalue())

    def test_errors_in_a_row_stop_and_resume_retries_them(self):
        bad = "not json at all"
        out = run(self.agent([STREAM, bad, bad, bad]), load_jsonl(SYNTH), load_meta(SYNTH),
                  tempfile.mkdtemp(), chunk=ChunkLimits(max_errors_in_a_row=3))
        m = json.loads((out / "manifest.json").read_text())
        self.assertEqual((m["status"], m["stop_reason"], m["done"]), ("stopped", "errors_in_a_row", 4))
        out = run(self.agent([STREAM] * 5), load_jsonl(SYNTH), load_meta(SYNTH), "unused",
                  resume_dir=out)
        recs = [json.loads(x) for x in (out / "results.jsonl").read_text().splitlines()]
        self.assertEqual(len(recs), 6)
        self.assertEqual(sum(r["outcome"] == "agent_error" for r in recs), 0)

    def test_no_usage_data_stops(self):
        def fn(d):
            if d["type"] == "rate_limit_event":
                d["type"] = "something_else"
        out = run(self.agent([edit_stream(fn)] * 6), load_jsonl(SYNTH), load_meta(SYNTH),
                  tempfile.mkdtemp(), chunk=ChunkLimits(max_calls_without_usage=2))
        m = json.loads((out / "manifest.json").read_text())
        self.assertEqual((m["done"], m["stop_reason"]), (2, "no_usage_data"))

    def test_resume_after_claude_code_update(self):
        out = run(self.agent([STREAM] * 6), load_jsonl(SYNTH), load_meta(SYNTH), tempfile.mkdtemp(),
                  chunk=ChunkLimits(max_tasks=2))
        a = self.agent([STREAM] * 4)
        a._version = "9.9.9 (Claude Code)"
        out = run(a, load_jsonl(SYNTH), load_meta(SYNTH), "unused", resume_dir=out)
        m = json.loads((out / "manifest.json").read_text())
        self.assertEqual(m["status"], "complete")
        self.assertEqual([c["claude_code_version"] for c in m["chunks"]], ["test", "9.9.9 (Claude Code)"])

    def test_crash_leaves_a_manifest_and_resume_works(self):
        def runner(argv, stdin, env, cwd, timeout):
            runner.n += 1
            if runner.n == 3:
                raise KeyboardInterrupt
            return 0, STREAM, ""
        runner.n = 0
        a = ClaudeSandboxAgent("claude-haiku-5-5", runner=runner, version="test")
        a.preflight = lambda: None
        root = tempfile.mkdtemp()
        with self.assertRaises(KeyboardInterrupt):
            run(a, load_jsonl(SYNTH), load_meta(SYNTH), root)
        out = next(Path(root).iterdir())
        m = json.loads((out / "manifest.json").read_text())
        self.assertEqual((m["status"], m["done"], m["stop_reason"]), ("stopped", 2, "crashed"))
        with open(out / "results.jsonl", "a") as f:
            f.write('{"task_id": "half')                 # A broken last line.
        out = run(self.agent([STREAM] * 4), load_jsonl(SYNTH), load_meta(SYNTH), "unused",
                  resume_dir=out)
        self.assertEqual(json.loads((out / "manifest.json").read_text())["done"], 6)

    def test_shared_meter_stops_the_next_run_at_once(self):
        meter = UsageMeter(ChunkLimits(max_tasks=2))
        root = tempfile.mkdtemp()
        run(self.agent([STREAM] * 6), load_jsonl(SYNTH), load_meta(SYNTH), root, meter=meter)
        out = run(self.agent([STREAM] * 6), load_jsonl(SYNTH), load_meta(SYNTH), root, meter=meter)
        m = json.loads((out / "manifest.json").read_text())
        self.assertEqual((m["done"], m["stop_reason"]), (0, "chunk_limit_tasks"))

    def test_tool_logs_go_to_the_run_folder(self):
        out = run(self.agent([STREAM] * 6), load_jsonl(SYNTH), load_meta(SYNTH), tempfile.mkdtemp())
        rec = json.loads((out / "results.jsonl").read_text().splitlines()[0])
        self.assertEqual(rec["tool_calls"], 1)
        self.assertEqual(rec["rate_limit"]["five_hour"]["utilization"], 0.05)
        s = json.loads((out / "summary.json").read_text())
        self.assertEqual(s["tool_calls"]["total"], 6)
        self.assertIn("refusal_rate", s)


class TestSelection(unittest.TestCase):
    def tasks(self):
        return [task(f"{s}-{i}", s) for s in ("a", "b", "c") for i in range(10)]

    def test_even_and_deterministic(self):
        ts = self.tasks()
        p = stratified(ts, 7, seed=0)
        counts = sorted(sum(t.subtask == s for t in p) for s in "abc")
        self.assertEqual(counts, [2, 2, 3])
        self.assertEqual([t.id for t in p], [t.id for t in stratified(list(reversed(ts)), 7, 0)])
        self.assertNotEqual([t.id for t in p], [t.id for t in stratified(ts, 7, 1)])

    def test_small_subtask_gives_its_share_away(self):
        ts = self.tasks()[:21] + [task("d-0", "d")]       # a 10, b 10, c 1, d 1
        p = stratified(ts, 12, 0)
        self.assertEqual(len(p), 12)
        self.assertEqual(sum(t.subtask == "d" for t in p), 1)

    def test_limit_records_the_selection(self):
        out = run(get_agent("first"), load_jsonl(SYNTH), load_meta(SYNTH), tempfile.mkdtemp(),
                  dry_run=True, limit=3)
        m = json.loads((out / "manifest.json").read_text())
        self.assertEqual(m["selection"]["method"], "stratified_by_subtask")
        self.assertEqual(m["selection"]["n_selected"], 3)
        self.assertEqual(m["n_tasks"], 3)


class TestPin(unittest.TestCase):
    def test_status(self):
        pinned = "7fa41a863832e2eaed8e419ca0df7bda31ed24e82cc54de2f22520d434868054"  # JSONL made from the fresh 5c77cec6 parquet
        meta = {"source": "hf://datasets/futurehouse/lab-bench", "config": "SeqQA"}
        self.assertEqual(pin_status({**meta, "sha256_now": pinned}), "match")
        self.assertEqual(pin_status({**meta, "sha256_now": "0" * 64}), "mismatch")
        self.assertEqual(pin_status({"source": "local file, no meta", "sha256_now": "x"}), "unpinned")

    def test_run_refuses_a_changed_file(self):
        meta = {"source": "hf://datasets/futurehouse/lab-bench", "config": "SeqQA",
                "sha256_now": "0" * 64}
        with self.assertRaises(SystemExit):
            run(get_agent("first"), load_jsonl(SYNTH), meta, tempfile.mkdtemp(), dry_run=True)
        out = run(get_agent("first"), load_jsonl(SYNTH), meta, tempfile.mkdtemp(), dry_run=True,
                  allow_unpinned=True)
        self.assertEqual(json.loads((out / "manifest.json").read_text())["tasks"]["pin"], "mismatch")


class TestStripTaskText(unittest.TestCase):
    def test_strips_copied_sequence_and_lines(self):
        seq = "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG"
        t = Task(id="x", benchmark="b", subset="S", subtask="s",
                 question=f"What is the ORF in\n{seq}\n?", ideal="MAIVMGR", distractors=("MAIV",))
        ans = f"The sequence {seq}, starts with ATG here.\n{seq}\nMail me@example.org\n[ANSWER]B[/ANSWER]"
        out = strip_task_text(ans, t)
        self.assertNotIn(seq, out)
        self.assertIn(f"[sequence: {len(seq)} nt]", out)
        self.assertIn("starts with ATG", out)                 # Short text stays.
        self.assertIn("[email]", out)
        self.assertIn("[ANSWER]B[/ANSWER]", out)

    def test_strips_prose_lower_case_and_codons(self):
        seq = "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG"
        prose = "Which primer pair amplifies the full coding region of the target gene"
        t = Task(id="x", benchmark="b", subset="S", subtask="s",
                 question=f"{prose}?\n{seq}", ideal="A", distractors=("B",))
        cases = [f"You asked: {prose.lower()}, so I compared them.",   # Prose in a longer line.
                 f"It is {seq.lower()} in lower case.",                  # Lower case.
                 "Codons: " + " ".join(seq[i:i + 3] for i in range(0, len(seq), 3))]
        for c in cases:
            out = strip_task_text(c, t)
            self.assertNotIn(prose.lower()[:40], out.lower())
            self.assertNotIn(seq[:20].lower(), out.lower().replace(" ", ""))

    def test_strips_split_sequence_formats(self):
        seq = "ATGGCCTTAGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAGATGGCCTTAGGCCATTGTAA"
        t = Task(id="x", benchmark="b", subset="S", subtask="s", question=seq, ideal="A",
                 distractors=("B",))
        forms = [" ".join(seq[i:i + 3] for i in range(0, len(seq), 3)).lower(),
                 ", ".join(seq[i:i + 3] for i in range(0, len(seq), 3)),
                 "\n".join(seq[i:i + 10] for i in range(0, len(seq), 10)),
                 " | ".join(seq[i:i + 15] for i in range(0, len(seq), 15)),
                 " ".join(f"{i + 1}:{c}" for i, c in enumerate(seq))]
        for f in forms:
            out = strip_task_text(f"Here: {f} done", t)
            letters = "".join(c for c in out if c in "ACGTacgt")
            self.assertLess(len(letters), 16, out[:120])
        self.assertEqual(strip_task_text("A cat sat. Use EcoRI.", t), "A cat sat. Use EcoRI.")

    def test_error_and_tool_errors_are_scrubbed(self):
        seq = "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG"
        t = Task(id="x", benchmark="b", subset="S", subtask="s", question=seq, ideal="A",
                 distractors=("B",))
        from bioagent_scorecard.agents.base import AgentResult, ToolCall
        from bioagent_scorecard.runner import run_one

        class A:
            makes_api_calls = True
            name = "a"

            def answer(self, mcq, seed):
                return AgentResult(text=None, model="m", error=f"claude exit 1: seq = '{seq}'",
                                   tool_calls=[ToolCall("t", ok=False, error=f'seq = "{seq}"')])
        rec, _ = run_one(A(), t, 0, {"models": {}}, 900, 2)
        self.assertNotIn(seq, json.dumps(rec))

    def test_keeps_answer_without_copies(self):
        t = task("x", "s", question="How many fragments?")
        self.assertEqual(strip_task_text("Three fragments. [ANSWER]C[/ANSWER]", t),
                         "Three fragments. [ANSWER]C[/ANSWER]")


class TestAnchors(unittest.TestCase):
    def test_base_subtask(self):
        self.assertEqual(anchors.base_subtask("ORF-seq-AAid-v1-public"), "ORF-seq-AAid")

    def test_ceiling_over_the_same_tasks(self):
        ts = [task("1", "ORF-transeff-v1-public"), task("2", "Prop-seq-gcpercent-v1-public")]
        a = anchors.anchors_for(ts)
        self.assertAlmostEqual(a["ceiling"], (0.38 + 1.00) / 2)
        self.assertEqual(a["ceiling_status"], "reported")
        self.assertIsNone(anchors.anchors_for([task("3", "unknown")])["ceiling"])

    def test_score(self):
        self.assertEqual(anchors.anchor_score(0.269, 0.269, 0.789), 0.0)
        self.assertEqual(anchors.anchor_score(0.789, 0.269, 0.789), 1.0)
        self.assertLess(anchors.anchor_score(0.1, 0.269, 0.789), 0)   # Not clipped.
        self.assertIsNone(anchors.anchor_score(0.5, 0.269, None))

    def test_all_seqqa_subtasks_have_a_ceiling(self):
        data = Path(__file__).parents[1] / "data" / "lab-bench" / "SeqQA.jsonl"
        if not data.exists():
            self.skipTest("SeqQA not fetched")
        a = anchors.anchors_for(load_jsonl(data))
        self.assertAlmostEqual(a["ceiling"], 0.7893, places=3)
        self.assertAlmostEqual(a["floor_longest"], 0.2612, places=3)   # 5c77cec6 data (0.269 on the stale copy)


if __name__ == "__main__":
    unittest.main()
