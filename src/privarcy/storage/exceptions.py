"""Custom exception types for the data_store package."""


class DataStoreError(Exception):
    """Raised for any storage failure: bad input, DB write/read failure,
    or a lookup for a record that doesn't exist where one was required."""
