# -----------------------------------------------------------------------------
# Copyright (C) 2024 L. Meunier

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------


def round_scientific_notation(num: float) -> str:
    """Round a number in scientific notation to 2 decimals.

    Parameters
    ----------
    num: float
        number to round

    Returns
    -------
    str
        rounded number in scientific notation
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

    Returns
    -------
    str
        The truncated string, or the original string if it is already shorter than or equal to the maximum length.
    """
    return name if len(name) <= max_length else name[: (max_length - 3)] + "..."
