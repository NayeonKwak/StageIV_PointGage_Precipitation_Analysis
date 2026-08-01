from pathlib import Path
import pandas as pd

# ----------------------------------------------------------
# USER SETTINGS -- EDIT THESE ONLY
# ----------------------------------------------------------
INPUT_FOLDER = Path("/Users/nayeonkwak/Downloads/LCRA_manual_chunks")   # folder with your downloaded LCRA csv files
COMBINED_OUT = Path("/Users/nayeonkwak/Downloads/LCRA_combined_clean.csv")
HOURLY_OUT = Path("/Users/nayeonkwak/Downloads/LCRA_hourly_end_of_hour.csv")

LOCAL_TIMEZONE = "US/Central"

# ----------------------------------------------------------
# LOAD ALL CSV FILES
# ----------------------------------------------------------
csv_files = sorted(INPUT_FOLDER.glob("*.csv"))

if len(csv_files) == 0:
    raise ValueError(f"No CSV files found in folder: {INPUT_FOLDER}")

print(f"Found {len(csv_files)} CSV files")

dfs = []

for f in csv_files:
    print(f"Loading {f.name}")
    df = pd.read_csv(f)

    # Try to identify likely datetime and rainfall columns
    possible_time_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    possible_val_cols = [c for c in df.columns if "rain" in c.lower() or "precip" in c.lower() or "pc" in c.lower()]

    if len(possible_time_cols) == 0:
        raise ValueError(f"Could not find a datetime column in {f.name}")
    if len(possible_val_cols) == 0:
        raise ValueError(f"Could not find a precipitation column in {f.name}")

    time_col = possible_time_cols[0]
    val_col = possible_val_cols[0]

    df = df.rename(columns={
        time_col: "DateTimeLocalRaw",
        val_col: "RainRaw"
    })

    df["DateTimeLocalRaw"] = pd.to_datetime(df["DateTimeLocalRaw"], errors="coerce")
    df["RainRaw"] = pd.to_numeric(df["RainRaw"], errors="coerce")

    df = df.dropna(subset=["DateTimeLocalRaw"]).copy()

    dfs.append(df[["DateTimeLocalRaw", "RainRaw"]])

# ----------------------------------------------------------
# COMBINE ALL FILES
# ----------------------------------------------------------
combined = pd.concat(dfs, ignore_index=True)

print("\nRows before de-duplication:", len(combined))

# Sort by local time first
combined = combined.sort_values("DateTimeLocalRaw")

# Remove exact duplicate local timestamps from chunk overlap
combined = combined.drop_duplicates(subset=["DateTimeLocalRaw"], keep="first")

print("Rows after local duplicate removal:", len(combined))

# ----------------------------------------------------------
# LOCALIZE TO CENTRAL TIME, HANDLE DST, CONVERT TO UTC
# ----------------------------------------------------------
# ambiguous="NaT" safely handles repeated fall DST hour
# nonexistent="shift_forward" safely handles spring DST skipped hour
combined["DateTimeLocal"] = combined["DateTimeLocalRaw"].dt.tz_localize(
    LOCAL_TIMEZONE,
    ambiguous="NaT",
    nonexistent="shift_forward"
)

n_ambiguous = combined["DateTimeLocal"].isna().sum()
print("Ambiguous DST rows dropped:", n_ambiguous)

combined = combined.dropna(subset=["DateTimeLocal"]).copy()

combined["DateTimeUTC"] = combined["DateTimeLocal"].dt.tz_convert("UTC")

# Final sort + UTC dedup
combined = combined.sort_values("DateTimeUTC")
combined = combined.drop_duplicates(subset=["DateTimeUTC"], keep="first")
combined = combined.reset_index(drop=True)

# Save combined clean file
combined_to_save = combined[["DateTimeLocal", "DateTimeUTC", "RainRaw"]].copy()
combined_to_save = combined_to_save.rename(columns={"RainRaw": "Point Gage Precip (in)"})
combined_to_save.to_csv(COMBINED_OUT, index=False)

print("\nSaved combined clean file:")
print(COMBINED_OUT)

# ----------------------------------------------------------
# ROLL UP TO HOURLY END-OF-HOUR TOTALS
# ----------------------------------------------------------
# This matches Stage IV convention:
# timestamp 01:00 UTC = precipitation accumulated from 00:00 to 01:00 UTC
hourly = (
    combined.set_index("DateTimeUTC")["RainRaw"]
            .resample("1h", label="right", closed="right")
            .sum(min_count=1)
            .reset_index()
)

hourly = hourly.rename(columns={"RainRaw": "Point Gage Precip (in)"})
hourly = hourly.sort_values("DateTimeUTC").reset_index(drop=True)

hourly.to_csv(HOURLY_OUT, index=False)

print("\nSaved hourly end-of-hour file:")
print(HOURLY_OUT)

# ----------------------------------------------------------
# QUICK CHECKS
# ----------------------------------------------------------
print("\n====================")
print("QUICK CHECKS")
print("====================")

print("\nCombined preview:")
print(combined_to_save.head())
print(combined_to_save.tail())

print("\nHourly preview:")
print(hourly.head())
print(hourly.tail())

print("\nHourly timestep check:")
print(hourly["DateTimeUTC"].diff().value_counts().head())

print("\nHourly precipitation summary:")
print(hourly["Point Gage Precip (in)"].describe())

print("\nDone.")