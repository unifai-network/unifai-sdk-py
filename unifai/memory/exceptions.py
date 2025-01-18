class MemoryError(Exception):
    """Base exception for memory operations"""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class EmptyContentError(MemoryError):
    """Raised when memory content is empty"""
    def __init__(self):
        super().__init__("Cannot generate embedding: Memory content is empty")

class EmbeddingDimensionError(MemoryError):
    """Raised when embedding dimensions don't match"""
    def __init__(self, expected: int, actual: int):
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Invalid embedding dimension: expected {expected}, got {actual}"
        )

class CollectionError(MemoryError):
    """Raised when there's an error with collection operations"""
    def __init__(self, operation: str, details: str):
        self.operation = operation
        self.details = details
        super().__init__(f"Collection operation '{operation}' failed: {details}")

class ConnectionError(MemoryError):
    """Raised when there's an error connecting to the database"""
    def __init__(self, host: str, port: int, details: str):
        self.host = host
        self.port = port
        super().__init__(f"Failed to connect to database at {host}:{port} - {details}")