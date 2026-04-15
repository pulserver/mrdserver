#!/usr/bin/env python3
"""CLI entry point for the MRD streaming server."""

import argparse
import logging
import os
import sys

from .server import Server

_DEFAULTS = {
    "host": "0.0.0.0",
    "port": 9002,
    "default_handler": "savedataonly",
    "output_dir": os.environ.get(
        "PULSERVER_OUTPUT",
        os.path.join(os.environ.get("PULSERVER_BASE", "/tmp"), "mrdserver"),
    ),
    "log_level": "INFO",
}


def main(argv: list[str] | None = None) -> None:
    """Run the MRD streaming server.

    Parameters
    ----------
    argv : list[str] or None
        Command-line arguments.  Defaults to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(
        description="MRD streaming reconstruction server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-H", "--host", type=str, help="Bind address")
    parser.add_argument("-p", "--port", type=int, help="Bind port")
    parser.add_argument(
        "-d",
        "--default-handler",
        type=str,
        help="Fallback handler module when config is empty or unknown",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        help="Directory for saved MRD data and logs",
    )
    parser.add_argument(
        "-s",
        "--save-data",
        action="store_true",
        help="Save incoming MRD data to HDF5",
    )
    parser.add_argument(
        "-r",
        "--handler-dir",
        type=str,
        action="append",
        default=[],
        help="Additional directory to search for handler .py files (repeatable)",
    )
    parser.add_argument(
        "-l",
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    parser.add_argument(
        "--logfile",
        type=str,
        default=None,
        help="Path to log file (in addition to stdout)",
    )

    parser.set_defaults(**_DEFAULTS)
    args = parser.parse_args(argv)

    # Logging setup
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if args.logfile:
        log_dir = os.path.dirname(os.path.abspath(args.logfile))
        os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(args.logfile))

    logging.basicConfig(
        format=fmt,
        level=getattr(logging, args.log_level),
        handlers=handlers,
        force=True,
    )

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    server = Server(
        host=args.host,
        port=args.port,
        default_handler=args.default_handler,
        output_dir=args.output_dir,
        save_data=args.save_data,
        handler_dirs=args.handler_dir,
    )
    server.serve()


if __name__ == "__main__":
    main()
