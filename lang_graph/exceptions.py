"""Custom exceptions for the sandbox tools.

Tools no longer return a {content, is_error} dict. On success they return a
string; on failure they raise one of the exceptions below. LangGraph's ToolNode
catches exceptions raised by a tool and wraps the message into a ToolMessage
with status="error" that is fed back to the model -- so each exception message
is the "what went wrong + what to do next" text the model will read.

Grouped into a few categories:
    path / security    PathTraversalError, SecretFileAccessError
    file existence     FileNotFoundInSandboxError, NotAFileError, NotADirectoryInSandboxError
    file content       FileDecodeError, EmptyContentError
    replacement        StringNotFoundError, AmbiguousReplacementError
    shell              CommandNotAllowedError, CommandTimeoutError
    argument / calc    InvalidArgumentError, InvalidTimezoneError, CalculationError
    search             NoMatchesError, SearchFailedError
"""


class SandboxError(Exception):
    """Base class for all sandbox tool exceptions, so callers can catch broadly."""


# --- path / security -----------------------------------------------------
class PathTraversalError(SandboxError):
    """The target path escapes the sandbox directory."""


class SecretFileAccessError(SandboxError):
    """Attempted to read a forbidden secret/credential file."""


# --- file existence ------------------------------------------------------
class FileNotFoundInSandboxError(SandboxError):
    """The target file does not exist."""


class NotAFileError(SandboxError):
    """The path exists but is not a file (likely a directory)."""


class NotADirectoryInSandboxError(SandboxError):
    """The path exists but is not a directory."""


# --- file content --------------------------------------------------------
class FileDecodeError(SandboxError):
    """The file is not valid UTF-8 text (likely binary)."""


class EmptyContentError(SandboxError):
    """The content to write is empty."""


# --- string replacement --------------------------------------------------
class StringNotFoundError(SandboxError):
    """The old_str to replace was not found in the file."""


class AmbiguousReplacementError(SandboxError):
    """old_str matches multiple places but replace_all is off; refused for safety."""


# --- shell ---------------------------------------------------------------
class CommandNotAllowedError(SandboxError):
    """The command is not in the whitelist."""


class CommandTimeoutError(SandboxError):
    """The command timed out."""


# --- argument / calculation ----------------------------------------------
class InvalidArgumentError(SandboxError):
    """A tool argument is invalid (missing, malformed, etc.)."""


class InvalidTimezoneError(SandboxError):
    """Unknown timezone."""


class CalculationError(SandboxError):
    """The arithmetic expression is invalid or failed to evaluate."""


# --- search --------------------------------------------------------------
class NoMatchesError(SandboxError):
    """The search returned no matches."""


class SearchFailedError(SandboxError):
    """The search itself failed (ripgrep error, timeout, etc.)."""
