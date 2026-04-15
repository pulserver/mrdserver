"""MRD streaming server with dynamic handler loading."""

__all__ = ["Server"]

import importlib
import importlib.util
import logging
import os
import signal
import socket
import sys
import threading
from types import ModuleType

import ismrmrd
import ismrmrd.xsd

from . import constants
from .connection import Connection


class Server:
    """TCP server that accepts ISMRMRD/MRD streaming connections.

    Each incoming connection is handled in a separate thread.  The server
    reads the config message to determine which handler module to load,
    then delegates processing to ``module.process(connection, config, metadata)``.

    Parameters
    ----------
    host : str
        Bind address (default ``"0.0.0.0"``).
    port : int
        Bind port (default ``9002``).
    default_handler : str
        Fallback handler module name when config is empty or unknown.
    output_dir : str
        Directory for saved MRD data and per-connection logs.
    save_data : bool
        Whether to persist incoming MRD data to HDF5.
    handler_dirs : list[str] or None
        Additional directories to search for handler ``.py`` files.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 9002,
        default_handler: str = "savedataonly",
        output_dir: str = "",
        save_data: bool = False,
        handler_dirs: list[str] | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.default_handler = default_handler
        self.output_dir = output_dir
        self.save_data = save_data
        self.handler_dirs = handler_dirs or []

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((self.host, self.port))

        logging.info(
            "MRD server listening on %s:%d  (default handler: %s)",
            self.host,
            self.port,
            self.default_handler,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def serve(self) -> None:
        """Block and accept connections until interrupted."""
        self._socket.listen(5)

        # Graceful shutdown on SIGTERM / SIGINT
        def _shutdown(signum, frame):
            logging.info("Received signal %d — shutting down", signum)
            self._socket.close()
            sys.exit(0)

        signal.signal(signal.SIGTERM, _shutdown)
        signal.signal(signal.SIGINT, _shutdown)

        while True:
            try:
                sock, (remote_addr, remote_port) = self._socket.accept()
            except OSError:
                break

            logging.info(
                "Accepted connection from %s:%d", remote_addr, remote_port
            )
            t = threading.Thread(
                target=self._handle_connection,
                args=(sock,),
                daemon=True,
            )
            t.start()

    # ------------------------------------------------------------------
    # Connection handler (runs in its own thread)
    # ------------------------------------------------------------------

    def _handle_connection(self, sock: socket.socket) -> None:
        connection = Connection(
            sock,
            savedata=self.save_data,
            savedataFolder=self.output_dir,
            savedataGroup="dataset",
        )

        try:
            # 1) Config message
            config = next(connection)
            if config is None and connection.is_exhausted:
                logging.info("Connection closed without data")
                return

            # 2) XML header
            metadata_xml = next(connection)
            if metadata_xml is None and connection.is_exhausted:
                logging.info("Connection closed without MRD header")
                return

            # Parse header
            try:
                metadata = ismrmrd.xsd.CreateFromDocument(metadata_xml)
                if metadata.acquisitionSystemInformation.systemFieldStrength_T is not None:
                    logging.info(
                        "Data from %s %s at %1.1fT",
                        metadata.acquisitionSystemInformation.systemVendor,
                        metadata.acquisitionSystemInformation.systemModel,
                        metadata.acquisitionSystemInformation.systemFieldStrength_T,
                    )
            except Exception:
                logging.warning("Metadata is not valid MRD XML — passing as text")
                metadata = metadata_xml

            # 3) Resolve handler module
            module = self._resolve_handler(config)
            logging.info("Starting handler '%s'", module.__name__)
            module.process(connection, config, metadata)

        except Exception:
            logging.exception("Error handling connection")

        finally:
            connection.shutdown_close()
            # Close HDF5 dataset if still open
            if hasattr(connection.saver, "dset") and connection.saver.dset is not None:
                try:
                    connection.saver.dset.close()
                except Exception:
                    pass
            if hasattr(connection.saver, "mrdFilePath") and connection.saver.mrdFilePath:
                logging.info(
                    "Incoming data saved at %s", connection.saver.mrdFilePath
                )

    # ------------------------------------------------------------------
    # Handler resolution
    # ------------------------------------------------------------------

    def _resolve_handler(self, config: str) -> ModuleType:
        """Resolve a handler module from a *config* string.

        Resolution order:

        1. ``importlib.import_module(config)`` (installed packages / sys.path).
        2. Search ``handler_dirs`` for ``<config>.py`` with a ``process`` callable.
        3. Fall back to ``self.default_handler``.

        Parameters
        ----------
        config : str
            Handler name sent by the client in the CONFIG message.

        Returns
        -------
        ModuleType
            A module exposing a ``process(connection, config, metadata)`` callable.
        """
        # Fast path: try standard import
        if config and config != "null":
            mod = self._try_import(config)
            if mod is not None:
                return mod

            # Search handler directories
            for d in self.handler_dirs:
                path = os.path.join(d, config + ".py")
                if os.path.isfile(path):
                    mod = self._load_from_file(config, path)
                    if mod is not None:
                        return mod

            logging.warning(
                "Handler '%s' not found — falling back to '%s'",
                config,
                self.default_handler,
            )

        # Fallback
        mod = self._try_import(self.default_handler)
        if mod is not None:
            return mod

        for d in self.handler_dirs:
            path = os.path.join(d, self.default_handler + ".py")
            if os.path.isfile(path):
                mod = self._load_from_file(self.default_handler, path)
                if mod is not None:
                    return mod

        # Last resort: inline drain
        return _NullHandler

    @staticmethod
    def _try_import(name: str) -> ModuleType | None:
        """Import *name* and return it if it has a ``process`` callable."""
        try:
            mod = importlib.import_module(name)
            if hasattr(mod, "process") and callable(mod.process):
                return mod
        except ImportError:
            pass
        return None

    @staticmethod
    def _load_from_file(name: str, path: str) -> ModuleType | None:
        """Load a module from *path* and return it if it has ``process``."""
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "process") and callable(mod.process):
                return mod
        except Exception as exc:
            logging.error("Failed to load '%s' from %s: %s", name, path, exc)
        return None


class _NullHandler:
    """Drain all messages without processing."""

    __name__ = "null"

    @staticmethod
    def process(connection, config, metadata):
        logging.info("Null handler — draining connection")
        try:
            for _msg in connection:
                pass
        finally:
            end = constants.GadgetMessageIdentifier.pack(constants.GADGET_MESSAGE_CLOSE)
            connection.socket.write(end)
