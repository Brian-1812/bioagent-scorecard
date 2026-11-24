"""A small sandbox for model-written Python code (row B).

`run_python(code)` runs the code in a new process with these limits:

- New network, mount, PID, IPC and UTS namespaces (`unshare`). No network
  interfaces, no host SysV shared memory or message queues.
- A new root file system (allowlist). It holds only a read-only bind of
  /usr (with /usr/local hidden), the links /bin, /lib, /lib64 and /sbin into
  it, /etc/ld.so.cache, /etc/passwd with one `nobody` line, /dev/null,
  /dev/zero, /dev/random, /dev/urandom, a new /proc, a 64 MB /tmp and the
  work folder at /sbx (read-only, it holds only main.py). `pivot_root` then
  detaches the old root, so no host path (/root, /srv, /etc, /sys, ...) is
  reachable, even from a nested user namespace.
- The code runs as `nobody` (uid 65534) with no capabilities and
  `no_new_privs`.
- Its own cgroup (a transient systemd scope): MemoryMax, MemorySwapMax=0,
  TasksMax and CPUQuota cap ALL processes of the call together.
- rlimits per process: CPU seconds, address space, file size, processes.
- stdout and stderr go to files under the file-size limit, so a print loop
  cannot fill the memory of the calling process. The model sees at most
  OUTPUT_LIMIT characters of each.
- A wall-clock limit. The namespace init gets SIGKILL when its parent dies
  (PR_SET_PDEATHSIG), and the PID namespace then kills every process in it.
- Python runs with `-I -B` (isolated mode: no user site, no PYTHON* env).

The sandbox fails closed. When it cannot build all of this (no root, no
`unshare`, `setpriv`, `pivot_root` or `systemd-run`), it does not run the
code. It returns `isolation: "unavailable"` and an error text.

Limits of this design (say them in any public result):
- It is not a VM. The code shares the host kernel, and there is no seccomp
  filter, so the kernel attack surface (for example user namespaces and
  nf_tables) is the normal one for an unprivileged process.
- Only the Python standard library is available (no Biopython, no numpy).
- It needs root and systemd on the host.
"""

from __future__ import annotations

import ctypes
import dataclasses
import os
import resource
import shutil
import signal
import subprocess
import tempfile
import time

NOBODY_UID = 65534
NOBODY_GID = 65534

#: Kept for the agent config text: the old host-folder hiding is replaced by
#: an allowlisted root, so no host folder is visible.
HIDE_DIRS = ("(new root: /usr read-only only; old root detached)",)

OUTPUT_LIMIT = 20_000      # Characters of stdout and of stderr that the model sees.

#: Optional systemd slice for the per-call scopes (for example jobs.slice,
#: so the calls also count against that slice's limits).
SLICE_ENV = "BAS_SANDBOX_SLICE"


@dataclasses.dataclass(frozen=True)
class Limits:
    wall_seconds: float = 30.0
    cpu_seconds: int = 20
    memory_mb: int = 1024        # Per process (RLIMIT_AS) and for the whole call (MemoryMax).
    file_size_mb: int = 16       # Also caps stdout and stderr.
    max_processes: int = 64      # TasksMax for the call; RLIMIT_NPROC is a backstop.
    cpu_quota_percent: int = 100
    scratch_mb: int = 64

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool
    seconds: float
    isolation: str          # "namespaces" or "unavailable"
    error: str = ""

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def isolation_available() -> tuple[bool, str]:
    """Return (True, "") when this host can build the sandbox, else (False, why)."""
    if os.name != "posix":
        return False, "not a POSIX host"
    if os.geteuid() != 0:
        return False, "the sandbox needs root to build namespaces and drop to nobody"
    for tool in ("unshare", "setpriv", "mount", "pivot_root", "umount", "systemd-run"):
        if shutil.which(tool) is None:
            return False, f"{tool} not found"
    if not os.path.exists("/run/systemd/system"):
        return False, "systemd is not running (needed for the per-call cgroup)"
    return True, ""


def _inner_script(scratch_mb: int) -> str:
    """Shell text that runs inside the new namespaces, as root, before the drop.

    $1 is the work folder on the host, $2 the python binary (under /usr)."""
    r = "/mnt"
    lines = [
        "set -e",
        # Step 1: the new root is a tmpfs on /mnt (private mount namespace).
        f"mount -t tmpfs -o size=4m,mode=0755 tmpfs {r}",
        f"mkdir {r}/usr {r}/etc {r}/dev {r}/proc {r}/tmp {r}/sbx {r}/.old",
        f"mount --bind /usr {r}/usr",
        f"mount -o remount,bind,ro,nosuid,nodev {r}/usr",
        f"if [ -d /usr/local ]; then mount -t tmpfs -o size=1m,ro tmpfs {r}/usr/local; fi",
    ]
    for d in ("bin", "lib", "lib64", "sbin"):
        # Merged-usr hosts have links; other hosts get a read-only bind.
        lines.append(f"if [ -L /{d} ]; then ln -s \"$(readlink /{d})\" {r}/{d}; "
                     f"elif [ -d /{d} ]; then mkdir {r}/{d}; mount --bind /{d} {r}/{d}; "
                     f"mount -o remount,bind,ro,nosuid,nodev {r}/{d}; fi")
    lines += [
        f"touch {r}/etc/ld.so.cache",
        f"if [ -f /etc/ld.so.cache ]; then mount --bind /etc/ld.so.cache {r}/etc/ld.so.cache; "
        f"mount -o remount,bind,ro {r}/etc/ld.so.cache; fi",
        f"echo 'nobody:x:{NOBODY_UID}:{NOBODY_GID}:nobody:/tmp:/usr/sbin/nologin' > {r}/etc/passwd",
    ]
    for n in ("null", "zero", "random", "urandom"):
        lines.append(f"touch {r}/dev/{n} && mount --bind /dev/{n} {r}/dev/{n}")
    lines += [
        f"mount -t proc -o nosuid,nodev,noexec proc {r}/proc",
        f"mount -t tmpfs -o size={scratch_mb}m,mode=1777,nosuid,nodev tmpfs {r}/tmp",
        f"mount --bind \"$1\" {r}/sbx",
        f"mount -o remount,bind,ro,nosuid,nodev {r}/sbx",
        # Step 2: switch to the new root and detach the old one.
        f"cd {r}",
        "pivot_root . .old",
        "cd /",
        "umount -l /.old",
        "rmdir /.old",
        "mount -o remount,ro /",
        # Step 3: drop to nobody with no capabilities and run the code. No
        # `exec`: the shell waits, so a signal death (for example SIGXCPU)
        # comes back as exit code 128 + signal.
        "cd /sbx",
        f"setpriv --reuid={NOBODY_UID} --regid={NOBODY_GID} --clear-groups "
        "--inh-caps=-all --bounding-set=-all --no-new-privs "
        "env -i PATH=/usr/bin:/bin HOME=/tmp LANG=C.UTF-8 "
        "\"$2\" -I -B /sbx/main.py && rc=0 || rc=$?",
        "exit $rc",
    ]
    return "\n".join(lines)


def build_argv(work_dir: str, python: str, limits: Limits) -> list[str]:
    """The command: a transient cgroup scope, then the namespaces, then main.py."""
    scope = ["systemd-run", "--scope", "--quiet", "--collect",
             "-p", f"MemoryMax={limits.memory_mb}M", "-p", "MemorySwapMax=0",
             "-p", f"TasksMax={limits.max_processes}",
             "-p", f"CPUQuota={limits.cpu_quota_percent}%"]
    slice_ = os.environ.get(SLICE_ENV) or _own_slice()
    if slice_:
        # Default: the caller's slice, so the calls count against its limits
        # (for example the slice of a job runner).
        scope.append(f"--slice={slice_}")
    return scope + ["--", "unshare", "--net", "--mount", "--pid", "--ipc", "--uts",
                    "--fork", "--kill-child", "--propagation", "private", "--",
                    "/bin/sh", "-c", _inner_script(limits.scratch_mb), "sandbox", work_dir, python]


def _own_slice() -> str | None:
    """The systemd slice of this process, from /proc/self/cgroup, or None."""
    try:
        path = open("/proc/self/cgroup", encoding="utf-8").read().strip().split("::", 1)[-1]
    except OSError:
        return None
    slices = [p for p in path.split("/") if p.endswith(".slice")]
    return slices[-1] if slices else None


PR_SET_PDEATHSIG = 1
KEYCTL_JOIN_SESSION_KEYRING = 1
SYS_KEYCTL = {"x86_64": 250, "aarch64": 219}


def _set_limits(limits: Limits):
    parent = os.getpid()

    def apply():
        libc = ctypes.CDLL(None, use_errno=True)
        # The sandbox dies with its parent (the MCP server). This survives the
        # exec of systemd-run and unshare (no set-uid binaries in the chain).
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
        if os.getppid() != parent:
            os._exit(1)                     # The parent died before prctl.
        # A new, empty session keyring: the code must not reach the caller's
        # (root's) session keyring.
        nr = SYS_KEYCTL.get(os.uname().machine)
        if nr is None or libc.syscall(nr, KEYCTL_JOIN_SESSION_KEYRING, None) < 0:
            os._exit(1)
        mem = limits.memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        # Soft limit sends SIGXCPU (a clear error); the hard limit 1 s later kills.
        resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds + 1))
        fsz = limits.file_size_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_FSIZE, (fsz, fsz))
        # RLIMIT_NPROC counts all processes of uid nobody on the host, so it is
        # only a backstop; TasksMax on the scope is the real cap.
        nproc = limits.max_processes * 4
        resource.setrlimit(resource.RLIMIT_NPROC, (nproc, nproc))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return apply


def _read_capped(f) -> str:
    f.seek(0)
    data = f.read(OUTPUT_LIMIT * 4 + 1).decode("utf-8", errors="replace")
    size = os.fstat(f.fileno()).st_size
    if len(data) > OUTPUT_LIMIT or size > OUTPUT_LIMIT * 4:
        return data[:OUTPUT_LIMIT] + f"\n[output cut: {size} bytes in total]"
    return data


def run_python(code: str, limits: Limits | None = None, python: str = "/usr/bin/python3") -> SandboxResult:
    """Run `code` in the sandbox and return its output. Never raises for bad code."""
    limits = limits or Limits()
    ok, why = isolation_available()
    if ok and not python.startswith("/usr/"):
        ok, why = False, f"python {python} is not under /usr, so it is not in the sandbox root"
    if not ok:
        return SandboxResult("", "", None, False, 0.0, "unavailable",
                             error=f"sandbox unavailable: {why}; the code did not run")
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="bas-sbx-") as work, \
            tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        os.chmod(work, 0o755)
        path = os.path.join(work, "main.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        os.chmod(path, 0o444)
        argv = build_argv(work, python, limits)
        timed_out = False
        try:
            p = subprocess.Popen(argv, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                                 preexec_fn=_set_limits(limits), start_new_session=True)
            try:
                rc = p.wait(timeout=limits.wall_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                # Kill the whole session: systemd-run, unshare and, through the PID
                # namespace, every process of the code.
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                rc = p.wait()
        except OSError as e:
            return SandboxResult("", "", None, False, round(time.monotonic() - t0, 3),
                                 "unavailable", error=f"sandbox failed to start: {e}")
        res = SandboxResult(_read_capped(out), _read_capped(err), None if timed_out else rc,
                            timed_out, round(time.monotonic() - t0, 3), "namespaces")
    if timed_out:
        res.error = f"wall-clock limit of {limits.wall_seconds:.0f}s reached"
        return res
    # The shell reports a signal death as 128 + signal. Only the signals that
    # the limits send are named; any other exit code is just the exit code.
    names = {24: "CPU time limit (SIGXCPU)", 25: "file size or output limit (SIGXFSZ)",
             9: "SIGKILL (CPU hard limit or out of memory)",
             15: "SIGTERM (the call's cgroup was stopped, most often at its memory cap)"}
    sig = -rc if rc < 0 else rc - 128
    if sig in names:
        res.error = f"exit code {rc}: killed by signal {sig}, {names[sig]}"
    return res
