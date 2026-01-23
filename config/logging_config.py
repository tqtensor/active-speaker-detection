import logging
import os
import sys

# Optional colored output
try:
    import colorlog

    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False


class TqdmLoggingHandler(logging.Handler):
    """A logging handler that redirects logs to work with tqdm progress bars.

    Writes log messages using tqdm.write() to prevent conflicts with progress bars.
    Falls back to standard stream writing if tqdm is not available.
    """

    def emit(self, record):
        """Emits a log record using tqdm.write() for progress bar compatibility.

        Args:
            record: LogRecord instance to be emitted.
        """
        try:
            import tqdm

            msg = self.format(record)
            tqdm.tqdm.write(msg, file=sys.stderr)
        except ImportError:
            # Fall back to standard stderr if tqdm not available
            try:
                msg = self.format(record)
                sys.stderr.write(msg + "\n")
                sys.stderr.flush()
            except Exception:
                self.handleError(record)
        except Exception:
            self.handleError(record)


def setup_logging(args) -> None:
    """Initializes logging configuration based on arguments.

    Sets up console and optional file logging with appropriate formatters and
    handlers. Supports colored console output, configurable log levels, and
    tqdm compatibility.

    Args:
        args: Namespace with attributes:
            - logLevel: Log level string (DEBUG, INFO, WARNING, ERROR).
            - logFile: Optional path to log file (None for console only).
            - noColor: Boolean to disable colored output.
    """
    # Get log level from args or environment variable
    log_level_str = getattr(args, "logLevel", None) or os.environ.get(
        "ASD_LOG_LEVEL", "INFO"
    )
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)

    # Get log file from args or environment variable
    log_file = getattr(args, "logFile", None) or os.environ.get("ASD_LOG_FILE", None)

    # Check if colors should be disabled
    no_color = getattr(args, "noColor", False) or os.environ.get("NO_COLOR", False)

    # Determine if terminal supports color
    use_color = (
        HAS_COLOR
        and not no_color
        and hasattr(sys.stderr, "isatty")
        and sys.stderr.isatty()
    )

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove any existing handlers
    root_logger.handlers.clear()

    # Console handler with tqdm support
    console_handler = TqdmLoggingHandler()
    console_handler.setLevel(log_level)

    # Create formatter (colored or plain)
    if use_color:
        console_formatter = colorlog.ColoredFormatter(
            "%(log_color)s[%(asctime)s] [%(levelname)s] [%(name)s]%(reset)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
    else:
        console_formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Optional file handler
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] [%(filename)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Retrieves a module-specific logger.

    Creates or retrieves a logger with the given name, typically the module's
    __name__ attribute. This creates a logger hierarchy that can be filtered
    by module.

    Args:
        name: Logger name (typically __name__ from calling module).

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)


def log_section(logger: logging.Logger, title: str, width: int = 60) -> None:
    """Logs a section header with visual separators.

    Creates a visually distinct section in the logs with separator lines above
    and below the title.

    Args:
        logger: Logger instance to use.
        title: Section title text.
        width: Width of separator line (default: 60).
    """
    logger.info("=" * width)
    logger.info(title)
    logger.info("=" * width)
