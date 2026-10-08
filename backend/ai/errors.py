from enum import Enum, auto

class AIVisionErrorStatus(Enum):
    INVALID_INPUT = auto()
    ENGINE_UNAVAILABLE = auto()
    ENGINE_FAILURE = auto()
    INVALID_ENGINE_OUTPUT = auto()
    TIMEOUT = auto()
    UNSUPPORTED_INPUT = auto()

class AIVisionError(Exception):
    def __init__(self, status: AIVisionErrorStatus, message: str):
        super().__init__(message)
        self.status = status
