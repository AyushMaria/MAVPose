"""
mavpose/safe_executor.py

Restricted execution of LLM-generated Python plotting scripts.

This is defence in depth, not a security boundary.  CPython cannot be fully
sandboxed from inside itself; for untrusted input, run MAVPose inside a
container or VM as well.  What this module does:

  1. Syntax check       — compile() before spawning anything.
  2. Clean subprocess   — runs ``python -E -B`` in a throwaway working
                          directory with a minimal environment, so secrets
                          such as OPENROUTER_API_KEY never reach the script.
  3. Resource limits    — on POSIX: CPU seconds, address space, max file
                          size, no core dumps.
  4. Audit hook         — a PEP 578 hook (which cannot be removed once
                          installed) denies process creation, sockets,
                          ctypes (outside library imports), and any file
                          write / delete / rename outside the allowed
                          output directories.  Writable directories are
                          removed from sys.path and cannot be imported from.  Unlike
                          an import denylist, this also catches dangerous
                          calls reached through allowed libraries
                          (e.g. ``pandas.io.common.os.system``).
  5. Import guard       — direct imports of obviously dangerous modules
                          from the script fail fast with a clear message,
                          which gives the self-healing loop a useful error.
  6. Timeout            — the child is killed after ``timeout_seconds``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from typing import Iterable, Optional, Tuple

# Modules the generated script must not import directly.
BLOCKED_MODULES = {
    "os",
    "subprocess",
    "sys",
    "shutil",
    "socket",
    "importlib",
    "ctypes",
    "signal",
    "pty",
    "tty",
    "termios",
    "fcntl",
    "pwd",
    "grp",
    "resource",
    "multiprocessing",
    "threading",
    "concurrent",
    "asyncio",
    "urllib",
    "http",
    "ftplib",
    "smtplib",
    "pickle",
    "shelve",
    "dbm",
    "sqlite3",
}

# Audit events that are always denied (exact match or prefix match on "x.").
_DENIED_EVENTS = (
    "subprocess.Popen",
    "os.system",
    "os.exec",
    "os.posix_spawn",
    "os.spawn",
    "os.fork",
    "os.forkpty",
    "os.kill",
    "os.killpg",
    "os.startfile",
    "pty.spawn",
    "socket.",
    "urllib.Request",
    "webbrowser.open",
    "winreg.",
)

# ctypes events are denied except while a library import is in progress:
# ``import ctypes`` itself calls dlopen(None), and some numeric libraries
# import ctypes as they load.  Once imports finish, any ctypes use from
# the script (including via an allowed library's attributes) is denied.
_IMPORT_TIME_ONLY_EVENTS = ("ctypes.",)

# File-system mutation events whose first argument is a path.
_PATH_WRITE_EVENTS = (
    "os.remove",
    "os.rmdir",
    "os.mkdir",
    "os.chmod",
    "os.chown",
    "os.truncate",
    "os.utime",
    "shutil.rmtree",
)
# Events whose first two arguments are paths (src, dst).
_PATH_PAIR_EVENTS = ("os.rename", "os.link", "os.symlink", "shutil.copyfile", "shutil.move")

# Environment variables passed through to the child; everything else is dropped.
_ENV_PASSTHROUGH = ("PATH", "LANG", "LC_ALL", "SYSTEMROOT", "TZ")

DEFAULT_MEMORY_LIMIT_MB = 4096
DEFAULT_FILE_SIZE_LIMIT_MB = 200

# Runner executed as __main__.  It installs the guards, then executes the
# generated script from its own file so tracebacks keep the script's real
# line numbers (the self-healing fix prompt relies on them).  Everything is
# defined inside a function so no helper (e.g. the real __import__) leaks
# into the script's globals.  The audit hook is installed last.
_RUNNER = textwrap.dedent("""\
    def __mavpose_install_guards():
        import builtins, os, sys

        script = {script!r}
        blocked = {blocked!r}
        denied = {denied!r}
        import_only = {import_only!r}
        path_write = {path_write!r}
        path_pair = {path_pair!r}
        allowed = tuple(os.path.realpath(p) for p in {allowed!r})
        write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
        real_import = builtins.__import__
        getframe = sys._getframe
        realpath = os.path.realpath
        fsdecode = os.fsdecode
        isfile = os.path.isfile
        import_depth = [0]

        def safe_import(name, *args, **kwargs):
            if name.split(".")[0] in blocked:
                caller = getframe(1).f_globals.get("__file__", "") or ""
                if caller == script:
                    raise ImportError(
                        "Import '" + name + "' is not allowed in this sandbox."
                    )
            import_depth[0] += 1
            try:
                return real_import(name, *args, **kwargs)
            finally:
                import_depth[0] -= 1

        def is_allowed(path):
            if isinstance(path, int):  # already-open file descriptor
                return True
            try:
                p = realpath(fsdecode(path))
            except Exception:
                return False
            return any(p == a or p.startswith(a + os.sep) for a in allowed)

        def deny(event):
            raise PermissionError("Blocked by MAVPose sandbox: " + event)

        def hook(event, args):
            for d in denied:
                if event == d or (d.endswith(".") and event.startswith(d)):
                    deny(event)
            if import_depth[0] == 0:
                for d in import_only:
                    if event.startswith(d):
                        deny(event)
            if event == "compile":
                # Block importing modules the script wrote itself into a
                # writable directory (which would run inside an import).
                filename = args[1] if len(args) > 1 else None
                if (isinstance(filename, (str, bytes)) and filename != script
                        and isfile(filename) and is_allowed(filename)):
                    deny("compile " + fsdecode(filename))
            elif event == "open":
                path, mode, flags = (tuple(args) + (None, None, None))[:3]
                writing = (
                    (isinstance(mode, str) and any(c in mode for c in "wax+"))
                    or (isinstance(flags, int) and flags & write_flags)
                )
                if writing and not is_allowed(path):
                    deny("write to " + str(path))
            elif event in path_write:
                if args and not is_allowed(args[0]):
                    deny(event + " " + str(args[0]))
            elif event in path_pair:
                for p in args[:2]:
                    if not is_allowed(p):
                        deny(event + " " + str(p))

        # Writable directories must never be importable.
        sys.path[:] = [
            p for p in sys.path
            if p and not is_allowed(p)
        ]
        builtins.__import__ = safe_import
        sys.addaudithook(hook)

    __mavpose_install_guards()
    del __mavpose_install_guards

    with open({script!r}, encoding="utf-8") as __f:
        __src = __f.read()
    __code = compile(__src, {script!r}, "exec")
    del __f, __src
    exec(__code, {{"__name__": "__main__", "__file__": {script!r}, "__builtins__": __builtins__}})
""")


def _make_preexec(timeout_seconds: int, memory_mb: int, file_size_mb: int):
    """Return a preexec_fn that applies POSIX resource limits in the child."""

    def _apply_limits() -> None:  # pragma: no cover - runs in the child
        import resource

        def _set(limit, value):
            try:
                _soft, hard = resource.getrlimit(limit)
                if hard != resource.RLIM_INFINITY:
                    value = min(value, hard)
                resource.setrlimit(limit, (value, value))
            except (ValueError, OSError):
                pass  # not supported on this platform / already lower

        _set(resource.RLIMIT_CPU, timeout_seconds + 1)
        _set(resource.RLIMIT_AS, memory_mb * 1024 * 1024)
        _set(resource.RLIMIT_FSIZE, file_size_mb * 1024 * 1024)
        _set(resource.RLIMIT_CORE, 0)

    return _apply_limits


def build_runner(script_path: str, allowed_write_dirs: Iterable[str]) -> str:
    """Return the runner source that guards and then executes *script_path*."""
    return _RUNNER.format(
        script=script_path,
        blocked=sorted(BLOCKED_MODULES),
        denied=_DENIED_EVENTS,
        import_only=_IMPORT_TIME_ONLY_EVENTS,
        path_write=_PATH_WRITE_EVENTS,
        path_pair=_PATH_PAIR_EVENTS,
        allowed=[os.path.abspath(d) for d in allowed_write_dirs],
    )


def execute_script(
    code: str,
    timeout_seconds: int = 30,
    allowed_write_dirs: Optional[Iterable[str]] = None,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
    file_size_limit_mb: int = DEFAULT_FILE_SIZE_LIMIT_MB,
) -> Tuple[bool, str]:
    """
    Execute *code* in a restricted subprocess.

    Parameters
    ----------
    code:
        Python source code to execute.
    timeout_seconds:
        Wall-clock timeout; the process is killed if exceeded.
    allowed_write_dirs:
        Directories the script may write to (e.g. where the plot is saved).
        A private temporary working directory is always allowed.
    memory_limit_mb, file_size_limit_mb:
        POSIX resource limits for the child process.

    Returns
    -------
    (success, output) — success is True if the exit code is 0; output is the
    combined stdout and stderr.
    """
    # 1. Syntax check before spawning a process
    try:
        compile(code, "<generated>", "exec")
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"

    workdir = tempfile.mkdtemp(prefix="mavpose_exec_")
    try:
        allowed = [workdir, *(allowed_write_dirs or [])]
        script_path = os.path.join(workdir, "plot_script.py")
        runner_path = os.path.join(workdir, "_mavpose_runner.py")
        with open(script_path, "w", encoding="utf-8") as fh:
            fh.write(code)
        with open(runner_path, "w", encoding="utf-8") as fh:
            fh.write(build_runner(script_path, allowed))

        env = {k: os.environ[k] for k in _ENV_PASSTHROUGH if k in os.environ}
        env.update({
            "HOME": workdir,
            "TMPDIR": workdir,
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": os.path.join(workdir, ".matplotlib"),
            "PYTHONIOENCODING": "utf-8",
        })

        kwargs = {}
        if os.name == "posix":
            kwargs["preexec_fn"] = _make_preexec(
                timeout_seconds, memory_limit_mb, file_size_limit_mb
            )

        # -E: ignore PYTHON* env vars; -B: don't write .pyc files.
        result = subprocess.run(
            [sys.executable, "-E", "-B", runner_path],
            cwd=workdir,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            **kwargs,
        )
        output = (result.stdout + result.stderr).strip()
        return result.returncode == 0, output

    except subprocess.TimeoutExpired:
        return False, f"Script timed out after {timeout_seconds}s."
    except Exception as exc:
        return False, str(exc)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
