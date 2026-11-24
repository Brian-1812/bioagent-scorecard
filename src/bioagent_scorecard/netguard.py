"""Block network access from Python code inside a `with` block.

The dry-run mode uses this guard. Inside the block, these raise
NetworkBlocked: socket connect / connect_ex / sendto / sendmsg,
create_connection, getaddrinfo, gethostbyname and gethostbyname_ex.

Limits: this is a Python-level guard, not a
sandbox. Code that uses the C module `_socket` directly, a C extension, or a
subprocess (curl, claude) is not blocked. The dry run is safe because only
agents with makes_api_calls = False may run in it, and the dummy agents do no
I/O. For a hard guarantee, run the dry run in a container with no network
(the CI job does this with `unshare -n` where it is allowed).
"""

from __future__ import annotations

import contextlib
import socket


class NetworkBlocked(RuntimeError):
    pass


_SOCKET_METHODS = ("connect", "connect_ex", "sendto", "sendmsg")
_MODULE_FUNCS = ("create_connection", "getaddrinfo", "gethostbyname", "gethostbyname_ex")


@contextlib.contextmanager
def no_network():
    saved_methods = {m: getattr(socket.socket, m) for m in _SOCKET_METHODS if hasattr(socket.socket, m)}
    saved_funcs = {f: getattr(socket, f) for f in _MODULE_FUNCS}

    def deny(*_a, **_k):
        raise NetworkBlocked("network access is blocked in dry-run mode")

    for m in saved_methods:
        setattr(socket.socket, m, deny)
    for f in saved_funcs:
        setattr(socket, f, deny)
    try:
        yield
    finally:
        for m, orig in saved_methods.items():
            setattr(socket.socket, m, orig)
        for f, orig in saved_funcs.items():
            setattr(socket, f, orig)
