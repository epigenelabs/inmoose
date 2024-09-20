def round_scientific_notation(num: float) -> str:
    """Round a number in scientific notation to 2 decimals.

    Args:
        num (float): number to round

    Returns:
        str: rounded number in scientific notation
    """
    return "{:0.2e}".format(num)


def truncate_name(name, max_length=10):
    """
    Truncates a given string to a maximum length.

    Parameters
    ----------
    name: str
        The input string to be truncated.
    max_length: int, optional
        The maximum allowed length of the truncated string. Defaults to 10.

    Returns:
        str: The truncated string, or the original string if it is already shorter than or equal to the maximum length.
    """
    return name if len(name) <= max_length else name[: (max_length - 3)] + "..."
