"""Shared constants for the thesis pipeline."""

NORWAY_ZONES: tuple[str, ...] = ("NO_1", "NO_2", "NO_3", "NO_4", "NO_5")
SEASONAL_FEATURES: tuple[str, ...] = ("hour", "day_of_week", "month", "is_weekend")
RENEWABLE_KEYWORDS: tuple[str, ...] = ("Wind", "Solar", "Hydro", "Biomass", "Geothermal")

CROSSBORDER_PAIRS: tuple[tuple[str, str], ...] = (
    ("NO_1", "NO_2"),
    ("NO_1", "NO_3"),
    ("NO_1", "NO_5"),
    ("NO_2", "NO_5"),
    ("NO_3", "NO_4"),
    ("NO_3", "NO_5"),
    ("NO_1", "SE_3"),
    ("NO_2", "SE_4"),
    ("NO_3", "SE_2"),
    ("NO_4", "SE_1"),
    ("NO_4", "SE_2"),
    ("NO_4", "FI"),
    ("NO_2", "DK_1"),
    ("NO_2", "NL"),
    ("NO_2", "DE"),
    ("NO_2", "GB"),
)

