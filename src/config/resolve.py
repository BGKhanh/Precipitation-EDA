"""
Centralized column-name resolution helpers.

Many files independently do ``target_col = target_col or Config.COLUMN_MAPPING.get(...)``.
These helpers eliminate the duplication.
"""

from ..config.constants import Config


def resolve_target_col(target_col: str = None) -> str:
    """Return the canonical target column name."""
    return target_col or Config.COLUMN_MAPPING.get('PRECTOTCORR', 'Lượng mưa')


def resolve_date_col(date_col: str = None) -> str:
    """Return the canonical date column name."""
    return date_col or Config.COLUMN_MAPPING.get('DATE', 'Ngày')
