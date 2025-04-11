"""Errors module for Goose.

This module defines the custom exception classes used in the Goose framework.
"""


class Honk(Exception):
    """Base exception class for Goose framework errors.

    This exception is raised when an error occurs in the Goose framework,
    such as invalid task configuration, missing requirements, or runtime errors.

    The name "Honk" follows the goose theme of the framework.
    """

    pass
