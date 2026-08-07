from __future__ import annotations

import sys


def launch_server(run_shell: bool = False) -> None:
    from .api_server import run_api_server
    from .args import parse_args
    from .supervisor import create_backend_supervisor

    server_args, run_shell = parse_args(sys.argv[1:], run_shell)
    supervisor = create_backend_supervisor(server_args)

    run_api_server(
        server_args,
        supervisor.start,
        stop_backend=supervisor.shutdown,
        run_shell=run_shell,
    )


if __name__ == "__main__":
    launch_server()
