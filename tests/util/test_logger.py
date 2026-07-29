import logging
import logging.handlers
import os
import tempfile
import unittest
from unittest import mock

from simsopt.util import logger as logger_module
from simsopt.util.logger import initialize_logging


def _reset_root_logger():
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass
    root.setLevel(logging.WARNING)


class InitializeLoggingNonMPITests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._cwd = os.getcwd()
        os.chdir(self._tmp.name)

    def tearDown(self):
        os.chdir(self._cwd)
        _reset_root_logger()
        self._tmp.cleanup()

    def test_defaults_install_console_and_file_handlers(self):
        initialize_logging()
        handlers = logging.getLogger().handlers
        # Two console handlers and a file handler should remain (no MPI handler).
        self.assertTrue(any(isinstance(h, logging.StreamHandler)
                            and not isinstance(h, logging.FileHandler)
                            for h in handlers))
        self.assertTrue(any(isinstance(h, logging.handlers.RotatingFileHandler)
                            for h in handlers))

    def test_filename_is_applied_to_file_handler(self):
        target = os.path.join(self._tmp.name, "custom.log")
        initialize_logging(filename=target)
        file_handlers = [
            h for h in logging.getLogger().handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        self.assertEqual(len(file_handlers), 1)
        self.assertEqual(os.path.abspath(file_handlers[0].baseFilename),
                         os.path.abspath(target))

    def test_level_is_applied_to_file_handler(self):
        initialize_logging(level="WARNING")
        file_handlers = [
            h for h in logging.getLogger().handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        self.assertEqual(len(file_handlers), 1)
        self.assertEqual(file_handlers[0].level, logging.WARNING)

    def test_mpi_true_without_mpi4py_logs_warning(self):
        # initialize_logging() reconfigures the root logger via dictConfig, which
        # removes any handler assertLogs would install, so patch logging.warning
        # to observe the emitted message instead.
        with mock.patch.object(logger_module, "MPI", None), \
                mock.patch.object(logging, "warning") as mock_warning:
            initialize_logging(mpi=True)
        mock_warning.assert_called_once()
        self.assertIn("mpi4py not installed", mock_warning.call_args.args[0])
        # Even with mpi=True, MPI absent falls through the non-MPI branch.
        handler_classes = {type(h).__name__ for h in logging.getLogger().handlers}
        self.assertFalse(any("MPILogHandler" in c for c in handler_classes))


if __name__ == "__main__":
    unittest.main()
