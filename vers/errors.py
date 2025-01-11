class VersError(Exception):
    def __init__(self, *, error_type: str, message: str) -> None:
        self.error_type = error_type
        self.message = message
        super().__init__(message)
