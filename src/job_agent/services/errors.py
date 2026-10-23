"""Domain errors mapped to HTTP status codes by the API layer."""


class DomainError(Exception):
    """Base class. Messages are safe to show to the requesting user."""


class NotFound(DomainError):
    pass


class Conflict(DomainError):
    pass


class InvalidInput(DomainError):
    pass
