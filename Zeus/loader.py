"""
loader.py
---------
DataLoader handles all CSV ingestion and normalisation.

Responsibilities
----------------
1. Parse the uploaded price CSV into a DataFrame.
2. Validate that all required columns are present and typed correctly.
3. Normalise all timestamps to ``America/New_York`` (localised, not UTC-offset).
4. Sort by timestamp ascending.
5. Return a clean, index-sorted DataFrame ready for indicator computation.

Timezone policy
---------------
If the CSV timestamps are timezone-naive, they are *assumed* to be in
``America/New_York`` and localised accordingly.  If they carry a UTC offset
or a tz name, they are converted to ET.  This avoids silent bugs when a user
pastes data from a broker that reports in UTC.
"""

from __future__ import annotations

from io import StringIO
from typing import Union

import pandas as pd
import pytz

REQUIRED_PRICE_COLUMNS = {"Timestamp", "Open", "High", "Low", "Close", "Volume"}
ET = pytz.timezone("America/New_York")


class DataLoader:
    """Stateless CSV loader + validator."""

    # ---------------------------------------------------------------------------
    # Price data
    # ---------------------------------------------------------------------------

    @staticmethod
    def load_price_data(raw: Union[str, StringIO]) -> pd.DataFrame:
        """Read and validate a price CSV.

        Parameters
        ----------
        raw : str or file-like
            The CSV content (from ``UploadedFile.getvalue().decode()`` or a path).

        Returns
        -------
        DataFrame
            Indexed by a tz-aware ``Timestamp`` column (ET), sorted ascending.

        Raises
        ------
        ValueError
            If required columns are missing or types cannot be coerced.
        """
        df = pd.read_csv(raw if isinstance(raw, StringIO) else StringIO(raw))

        # ------------------------------------------------------------------
        # Case-insensitive column normalisation
        # Build a map from the lowercase version of each required name to its
        # canonical Title_Case form.  Any column in the file whose lowered,
        # stripped name matches a required column gets renamed.  Columns that
        # are not required (e.g. "vwap", "transactions") are left untouched.
        # ------------------------------------------------------------------
        canonical = {name.lower(): name for name in REQUIRED_PRICE_COLUMNS}
        rename_map = {
            col: canonical[col.strip().lower()]
            for col in df.columns
            if col.strip().lower() in canonical
        }
        df = df.rename(columns=rename_map)

        # ------------------------------------------------------------------
        # Column validation
        # ------------------------------------------------------------------
        present = {c.strip() for c in df.columns}
        missing = REQUIRED_PRICE_COLUMNS - present
        if missing:
            raise ValueError(
                f"Price CSV is missing required columns: {sorted(missing)}. "
                f"Found: {sorted(present)}"
            )

        # Normalise column names (strip whitespace)
        df.columns = [c.strip() for c in df.columns]

        # ------------------------------------------------------------------
        # Type coercion
        # ------------------------------------------------------------------
        numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # ------------------------------------------------------------------
        # Timestamp parsing + TZ normalisation
        # ------------------------------------------------------------------
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

        if df["Timestamp"].isna().all():
            raise ValueError("Could not parse any Timestamp values. Check format.")

        df = DataLoader._normalise_tz(df)

        # ------------------------------------------------------------------
        # Sort & index
        # ------------------------------------------------------------------
        df = df.sort_values("Timestamp").reset_index(drop=True)
        df = df.set_index("Timestamp")

        return df

    # ---------------------------------------------------------------------------
    # Blackout dates
    # ---------------------------------------------------------------------------

    @staticmethod
    def load_blackout_dates(raw: Union[str, StringIO]) -> pd.DataFrame:
        """Read a blackout file (CSV or whitespace-delimited .txt).

        Expected columns: Date, Reason  (case-insensitive).
        If the file is whitespace- or tab-delimited instead of comma-separated,
        the loader detects this automatically and re-parses.

        Returns
        -------
        DataFrame with columns [Date (datetime.date), Reason (str)]
        """
        text = raw if isinstance(raw, str) else raw.read()

        # ------------------------------------------------------------------
        # Parse: try CSV first; if that yields only one column, fall back to
        # whitespace-delimited (handles .txt files with space/tab separation).
        # ------------------------------------------------------------------
        df = pd.read_csv(StringIO(text))
        if len(df.columns) == 1:
            df = pd.read_csv(StringIO(text), sep=r"\s+", engine="python")

        # ------------------------------------------------------------------
        # Case-insensitive column normalisation for Date / Reason
        # ------------------------------------------------------------------
        canonical_blackout = {"date": "Date", "reason": "Reason"}
        rename_map = {
            col: canonical_blackout[col.strip().lower()]
            for col in df.columns
            if col.strip().lower() in canonical_blackout
        }
        df = df.rename(columns=rename_map)
        df.columns = [c.strip() for c in df.columns]

        if "Date" not in df.columns:
            # Check if the first column looks like a date — this means the file
            # is missing a header row and the first date was parsed as the column name
            first_col = df.columns[0] if len(df.columns) > 0 else ""
            if any(sep in first_col for sep in ["-", "/"]):
                raise ValueError(
                    "Blackout file is missing a header row. "
                    f"The first line '{first_col}' looks like a date, not a column name. "
                    "Add 'Date' as the first line, or 'Date,Reason' if you have reasons."
                )
            raise ValueError(
                "Blackout file must contain a 'Date' column. "
                f"Found: {sorted(df.columns)}"
            )

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
        if "Reason" not in df.columns:
            df["Reason"] = "Unspecified"

        df = df.dropna(subset=["Date"])
        return df[["Date", "Reason"]]

    # ---------------------------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------------------------

    @staticmethod
    def _normalise_tz(df: pd.DataFrame) -> pd.DataFrame:
        """Convert all timestamps to America/New_York.
        
        If timestamps are naive (no timezone), assume they are already in ET and localize.
        If timestamps are tz-aware (e.g., UTC from market data APIs), convert to ET.
        This ensures all trading logic (session timing, Friday expiry) operates in ET.
        """
        ts = df["Timestamp"]

        if ts.dt.tz is None:
            # Naive → assume ET, localise
            df["Timestamp"] = ts.dt.tz_localize(ET, ambiguous="infer", nonexistent="shift_forward")
        else:
            # Already tz-aware (e.g., UTC) → convert to ET
            df["Timestamp"] = ts.dt.tz_convert(ET)

        return df
