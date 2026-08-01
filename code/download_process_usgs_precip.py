import requests
import pandas as pd

# ----------------------------------------------------------
# USER SETTINGS
# ----------------------------------------------------------
SITE_NO = "02336526"
PARAM_CD = "00045"   # precipitation
START = "2017-01-01"
END = "2026-12-31"

RAW_OUT = f"USGS_{SITE_NO}_precip_raw_{START}_to_{END}.csv"
HOURLY_OUT = f"USGS_{SITE_NO}_precip_hourly_{START}_to_{END}.csv"

# ----------------------------------------------------------
# DOWNLOAD FROM USGS IV SERVICE
# ----------------------------------------------------------
url = "https://waterservices.usgs.gov/nwis/iv/"
params = {
    "format": "json",
    "sites": SITE_NO,
    "parameterCd": PARAM_CD,
    "startDT": START,
    "endDT": END,
    "siteStatus": "all",
}

r = requests.get(url, params=params, timeout=120)
r.raise_for_status()
j = r.json()

series = j["value"]["timeSeries"]
if not series:
    raise ValueError(
        "No time series returned. Check site number, parameter code, and date range."
    )

# keep the precipitation series
target = None
for ts in series:
    variable = ts.get("variable", {})
    codes = variable.get("variableCode", [])
    if codes and codes[0].get("value") == PARAM_CD:
        target = ts
        break

if target is None:
    target = series[0]

vals = target["values"][0]["value"]

df = pd.DataFrame(vals)
if df.empty:
    raise ValueError("USGS returned an empty precipitation table.")

# ----------------------------------------------------------
# CLEAN RAW DATA
# ----------------------------------------------------------
df["dateTime"] = pd.to_datetime(df["dateTime"], utc=True, errors="coerce")
df["value"] = pd.to_numeric(df["value"], errors="coerce")

raw = (
    df.rename(columns={"value": "Precip_in"})
      .loc[:, ["dateTime", "Precip_in", "qualifiers"]]
      .dropna(subset=["dateTime"])
      .sort_values("dateTime")
      .drop_duplicates(subset=["dateTime"])
      .reset_index(drop=True)
)

# trim exactly to requested range
raw = raw[
    (raw["dateTime"] >= pd.Timestamp(START, tz="UTC")) &
    (raw["dateTime"] < pd.Timestamp("2027-01-01", tz="UTC"))
].reset_index(drop=True)

raw.to_csv(RAW_OUT, index=False)

# ----------------------------------------------------------
# DIAGNOSTIC: CHECK RAW TIME STEP
# ----------------------------------------------------------
dt_counts = raw["dateTime"].diff().value_counts(dropna=True)
print("Most common raw timestep(s):")
print(dt_counts.head())

# ----------------------------------------------------------
# HOURLY AGGREGATION
# ----------------------------------------------------------
# This sums all sub-hourly precipitation values inside each UTC hour.
# It assumes the raw series is incremental precipitation depth per interval.
hourly = (
    raw.set_index("dateTime")["Precip_in"]
       .resample("1h", label="right", closed="right")
       .sum(min_count=1)
       .reset_index()
)

hourly = hourly.rename(columns={"Precip_in": "Point Gage Precip (in)"})
hourly = hourly.sort_values("dateTime").reset_index(drop=True)

# optional exact requested range again
hourly = hourly[
    (hourly["dateTime"] >= pd.Timestamp(START, tz="UTC")) &
    (hourly["dateTime"] < pd.Timestamp("2027-01-01", tz="UTC"))
].reset_index(drop=True)

hourly.to_csv(HOURLY_OUT, index=False)

# ----------------------------------------------------------
# OPTIONAL MERGE-READY VERSION
# ----------------------------------------------------------
merge_ready = hourly.rename(columns={"dateTime": "DateTimeUTC"})
MERGE_OUT = f"USGS_{SITE_NO}_precip_hourly_merge_ready_{START}_to_{END}.csv"
merge_ready.to_csv(MERGE_OUT, index=False)

print("\nSaved files:")
print(f"Raw interval data: {RAW_OUT}")
print(f"Hourly data:       {HOURLY_OUT}")
print(f"Merge-ready data:  {MERGE_OUT}")

print("\nRaw preview:")
print(raw.head())
print(raw.tail())

print("\nHourly preview:")
print(hourly.head())
print(hourly.tail())
print(f"\nRaw rows: {len(raw)}")
print(f"Hourly rows: {len(hourly)}")