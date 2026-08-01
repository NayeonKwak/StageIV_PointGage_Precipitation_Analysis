from pathlib import Path
import pandas as pd
import numpy as np
import subprocess
import gzip
import re
import os
import time
from tqdm import tqdm

# ----------------------------------------------------------
# OPTIONAL IMPORT
# ----------------------------------------------------------
# This script prefers pygrib when available, then falls back
# to wgrib2 text extraction for legacy files.
try:
    import pygrib
    HAS_PYGRIB = True
except Exception:
    HAS_PYGRIB = False

# ----------------------------------------------------------
# USER SETTINGS
# ----------------------------------------------------------
ROOT_DIR = Path("/Volumes/StageIV/StageIV2015-2025")
MASK_DIR = ROOT_DIR / "StageIV_masks"
OUTPUT_DIR = ROOT_DIR / "outputs_auto"
TEMP_DIR = OUTPUT_DIR / "temp_auto"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# ---- WHAT TO PROCESS ----
# Start with a small test first, then expand.
YEARS = ["2017", "2018", "2019", "2020"]
MONTHS = [f"{m:02d}" for m in range (1,13)]

MASK_FILES = {
    "Philadelphia": MASK_DIR / "Philadelphia_StageIV_exact_cells.csv",
    "Atlanta": MASK_DIR / "Atlanta_StageIV_exact_cells.csv",
    "Austin": MASK_DIR / "Austin_StageIV_exact_cells.csv",
}

SUBTRACT_ONE_FROM_MASK_INDICES = False
FLIP_Y_AXIS = False
FORCE_RERUN = False
WGRIB2_THREADS = "1"

# Fixed Stage IV HRAP grid
STAGEIV_NX = 1121
STAGEIV_NY = 881

MM_TO_IN = 1.0 / 25.4

print("RUNNING AUTO-DETECT STAGE IV EXTRACTOR")

# ----------------------------------------------------------
# HELPERS
# ----------------------------------------------------------
def parse_timestamp_from_filename(filename: str):
    m = re.search(r"(?<!\d)(\d{10})(?!\d)", filename)
    if not m:
        return pd.NaT
    try:
        return pd.to_datetime(m.group(1), format="%Y%m%d%H", utc=True)
    except Exception:
        return pd.NaT


def classify_hourly_file(fp: Path):
    """
    Priority:
      1 = st4_conus raw grb2
      2 = st4_pr raw grb2
      3 = legacy ST4 raw
      4 = legacy ST4 gz
    """
    name = fp.name.lower()

    if not fp.is_file():
        return None
    if ".01h" not in name:
        return None
    if name.endswith(".idx"):
        return None

    ts = parse_timestamp_from_filename(fp.name)
    if pd.isna(ts):
        return None

    if "st4_conus" in name:
        source_type = "conus"
    elif "st4_pr" in name:
        source_type = "st4_pr"
    elif name.startswith("st4.") or "stage4" in name or "stageiv" in name:
        source_type = "legacy"
    else:
        source_type = "unknown"

    compression = "gz" if name.endswith(".gz") else "raw"

    if source_type == "conus" and compression == "raw":
        priority = 1
    elif source_type == "st4_pr" and compression == "raw":
        priority = 2
    elif source_type == "legacy" and compression == "raw":
        priority = 3
    elif source_type == "legacy" and compression == "gz":
        priority = 4
    else:
        priority = 99

    return {
        "path": str(fp),
        "timestamp": ts,
        "source_type": source_type,
        "compression": compression,
        "priority": priority,
        "file_name": fp.name,
    }


def gunzip_to_temp(gz_path: Path, temp_dir: Path):
    out_name = gz_path.name[:-3]
    out_path = temp_dir / out_name

    if out_path.exists():
        return out_path

    with gzip.open(gz_path, "rb") as f_in:
        with open(out_path, "wb") as f_out:
            f_out.write(f_in.read())

    return out_path


def run_wgrib2_command(cmd):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = WGRIB2_THREADS
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return result.returncode, result.stdout, result.stderr


def monthly_outputs_exist(year: str, month: str):
    expected = [
        OUTPUT_DIR / f"Philadelphia_StageIV_hourly_{year}{month}.csv",
        OUTPUT_DIR / f"Atlanta_StageIV_hourly_{year}{month}.csv",
        OUTPUT_DIR / f"Austin_StageIV_hourly_{year}{month}.csv",
    ]
    return all(fp.exists() for fp in expected)


def load_precip_grid_with_pygrib(raw_fp: Path):
    """
    Try to read precipitation directly with pygrib.
    Returns (grid_mm, method_string)
    """
    if not HAS_PYGRIB:
        raise RuntimeError("pygrib not available")

    with pygrib.open(str(raw_fp)) as grbs:
        msgs = list(grbs)

        preferred = []
        for g in msgs:
            name = str(getattr(g, "name", "") or "")
            shortName = str(getattr(g, "shortName", "") or "")
            if (
                "precip" in name.lower()
                or "precip" in shortName.lower()
                or shortName.upper() == "APCP"
            ):
                preferred.append(g)

        if len(preferred) > 0:
            grb = preferred[0]
            msg_used = f"pygrib:{getattr(grb, 'shortName', 'unknown')}"
        else:
            grb = msgs[0]
            msg_used = "pygrib:first_message"

        vals = np.asarray(grb.values, dtype=float)

    if vals.ndim != 2:
        raise RuntimeError(f"Unexpected pygrib array shape {vals.shape}")

    if FLIP_Y_AXIS:
        vals = np.flipud(vals)

    return vals, msg_used


def parse_wgrib2_text_file(txt_fp: Path):
    """
    Parse numeric values from a wgrib2 text dump.
    """
    vals = []
    with open(txt_fp, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # pull out all floats in the line
            found = re.findall(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?", line)
            if found:
                vals.extend(found)

    if len(vals) == 0:
        raise RuntimeError(f"No numeric values found in {txt_fp}")

    arr = np.array(vals, dtype=float)
    return arr


def reshape_stageiv_array(arr_1d: np.ndarray):
    """
    Reshape to the known Stage IV HRAP grid.
    """
    expected = STAGEIV_NX * STAGEIV_NY
    if arr_1d.size != expected:
        raise RuntimeError(
            f"Unexpected Stage IV grid size: found {arr_1d.size}, expected {expected}"
        )

    grid = arr_1d.reshape((STAGEIV_NY, STAGEIV_NX))

    if FLIP_Y_AXIS:
        grid = np.flipud(grid)

    return grid


def load_precip_grid_with_wgrib2_text(raw_fp: Path):
    """
    Legacy fallback:
    use wgrib2 to write a text dump, then reshape to Stage IV HRAP grid.

    Returns (grid_mm, method_string)
    """
    txt_fp = TEMP_DIR / f"{raw_fp.name}.txt"

    if txt_fp.exists():
        try:
            txt_fp.unlink()
        except Exception:
            pass

    candidate_cmds = [
        ["wgrib2", str(raw_fp), "-match", "APCP", "-no_header", "-text", str(txt_fp)],
        ["wgrib2", str(raw_fp), "-match", "precip", "-no_header", "-text", str(txt_fp)],
        ["wgrib2", str(raw_fp), "-d", "1", "-no_header", "-text", str(txt_fp)],
    ]

    last_error = None
    for cmd in candidate_cmds:
        code, out, err = run_wgrib2_command(cmd)
        if code == 0 and txt_fp.exists() and txt_fp.stat().st_size > 0:
            try:
                arr = parse_wgrib2_text_file(txt_fp)
                grid = reshape_stageiv_array(arr)
                try:
                    txt_fp.unlink()
                except Exception:
                    pass
                return grid, f"wgrib2_text:{' '.join(cmd[2:4])}"
            except Exception as e:
                last_error = e
        else:
            last_error = RuntimeError(f"wgrib2 failed for {raw_fp.name}: {err[:200]}")

    if txt_fp.exists():
        try:
            txt_fp.unlink()
        except Exception:
            pass

    raise RuntimeError(f"wgrib2 text fallback failed for {raw_fp.name}. Last error: {last_error}")


def load_precip_grid_auto(raw_fp: Path, source_type: str):
    """
    Auto-detect loading strategy.
    """
    # 1) Try pygrib first for everything, including legacy after gunzip
    try:
        grid, method = load_precip_grid_with_pygrib(raw_fp)
        return grid, method
    except Exception as e_pygrib:
        # 2) Fallback to wgrib2 text path
        try:
            grid, method = load_precip_grid_with_wgrib2_text(raw_fp)
            return grid, method
        except Exception as e_wgrib2:
            raise RuntimeError(
                f"Both loaders failed for {raw_fp.name} | "
                f"pygrib: {e_pygrib} | wgrib2_text: {e_wgrib2}"
            )


# ----------------------------------------------------------
# LOAD MASKS
# ----------------------------------------------------------
masks = {}
for site, fp in MASK_FILES.items():
    if not fp.exists():
        print(f"Skipping {site}: missing mask file -> {fp}")
        continue

    df = pd.read_csv(fp)

    if "WatershedFraction" in df.columns:
        weight_col = "WatershedFraction"
    elif "CellAreaFraction" in df.columns:
        weight_col = "CellAreaFraction"
    else:
        raise ValueError(
            f"{site} mask must contain 'WatershedFraction' or 'CellAreaFraction'."
        )

    required = ["Cell_I", "Cell_J", weight_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{site} mask missing columns: {missing}")

    cell_i = df["Cell_I"].astype(int).to_numpy()
    cell_j = df["Cell_J"].astype(int).to_numpy()

    if SUBTRACT_ONE_FROM_MASK_INDICES:
        cell_i = cell_i - 1
        cell_j = cell_j - 1

    weights = df[weight_col].astype(float).to_numpy()
    weight_sum = np.nansum(weights)

    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError(f"{site} mask weights invalid.")

    if not np.isclose(weight_sum, 1.0, atol=1e-3):
        print(f"{site}: normalizing weights from sum={weight_sum:.6f} to 1.0")
        weights = weights / weight_sum
        weight_sum = weights.sum()

    masks[site] = {
        "cell_i": cell_i,
        "cell_j": cell_j,
        "weights": weights,
        "weight_col_used": weight_col,
    }

    print(
        f"{site}: loaded {len(df)} cells; "
        f"weight_col={weight_col}; weight sum={weight_sum:.6f}"
    )

if not masks:
    raise ValueError("No valid mask files loaded.")

# ----------------------------------------------------------
# MASTER LOOP
# ----------------------------------------------------------
master_summary_rows = []

for YEAR in YEARS:
    year_dir = ROOT_DIR / YEAR
    if not year_dir.exists():
        print(f"\nSkipping missing year folder: {year_dir}")
        continue

    print(f"\n====================")
    print(f"PROCESSING YEAR {YEAR}")
    print(f"====================")

    for MONTH in MONTHS:
        start_time = time.time()

        if (not FORCE_RERUN) and monthly_outputs_exist(YEAR, MONTH):
            print(f"\nSkipping {YEAR}{MONTH}: outputs already exist")
            master_summary_rows.append({
                "year": YEAR,
                "month": MONTH,
                "status": "skipped_existing",
                "n_candidates": np.nan,
                "n_selected": np.nan,
                "n_failed": np.nan,
                "runtime_min": 0.0
            })
            continue

        month_folder = year_dir / f"{YEAR}{MONTH}"
        if not month_folder.exists():
            print(f"\nSkipping missing month folder: {month_folder}")
            master_summary_rows.append({
                "year": YEAR,
                "month": MONTH,
                "status": "missing_month_folder",
                "n_candidates": 0,
                "n_selected": 0,
                "n_failed": 0,
                "runtime_min": np.nan
            })
            continue

        print(f"\n--- Processing {YEAR}{MONTH} ---")
        print(f"Month folder: {month_folder}")

        candidates = []
        for fp in month_folder.rglob("*"):
            meta = classify_hourly_file(fp)
            if meta is not None:
                candidates.append(meta)

        if not candidates:
            print(f"No hourly Stage IV candidates found in {month_folder}")
            master_summary_rows.append({
                "year": YEAR,
                "month": MONTH,
                "status": "no_candidates",
                "n_candidates": 0,
                "n_selected": 0,
                "n_failed": 0,
                "runtime_min": round((time.time() - start_time) / 60, 2)
            })
            continue

        cand_df = pd.DataFrame(candidates).sort_values(["timestamp", "priority", "file_name"])
        cand_csv = OUTPUT_DIR / f"StageIV_candidates_auto_{YEAR}{MONTH}.csv"
        cand_df.to_csv(cand_csv, index=False)

        # one file per timestamp
        selected_rows = []
        dup_rows = []

        for ts, group in cand_df.groupby("timestamp", sort=True):
            group_sorted = group.sort_values(["priority", "file_name"]).reset_index(drop=True)
            chosen = group_sorted.iloc[0]
            selected_rows.append(chosen.to_dict())

            if len(group_sorted) > 1:
                for _, row in group_sorted.iterrows():
                    dup_rows.append({
                        "timestamp": ts,
                        "file_name": row["file_name"],
                        "source_type": row["source_type"],
                        "compression": row["compression"],
                        "priority": row["priority"],
                        "selected": row["file_name"] == chosen["file_name"]
                    })

        selected_df = pd.DataFrame(selected_rows).sort_values("timestamp").reset_index(drop=True)
        selected_csv = OUTPUT_DIR / f"StageIV_selected_auto_{YEAR}{MONTH}.csv"
        selected_df.to_csv(selected_csv, index=False)

        if dup_rows:
            dup_csv = OUTPUT_DIR / f"StageIV_duplicates_auto_{YEAR}{MONTH}.csv"
            pd.DataFrame(dup_rows).to_csv(dup_csv, index=False)

        print(f"Candidates: {len(cand_df)} | Selected timestamps: {len(selected_df)}")

        records = {site: [] for site in masks.keys()}
        failed_rows = []

        for _, row in tqdm(selected_df.iterrows(), total=len(selected_df), desc=f"Extracting {YEAR}{MONTH}"):
            fp = Path(row["path"])
            ts = pd.to_datetime(row["timestamp"], utc=True)
            source_type = row["source_type"]
            compression = row["compression"]

            raw_fp = None
            temp_created = None

            try:
                if compression == "gz":
                    raw_fp = gunzip_to_temp(fp, TEMP_DIR)
                    temp_created = raw_fp
                else:
                    raw_fp = fp

                precip_mm, method_used = load_precip_grid_auto(raw_fp, source_type)

                for site, info in masks.items():
                    vals_mm = precip_mm[info["cell_j"], info["cell_i"]]
                    watershed_precip_mm = np.sum(vals_mm * info["weights"])
                    watershed_precip_in = watershed_precip_mm * MM_TO_IN

                    records[site].append({
                        "DateTimeUTC": ts,
                        "StageIV_Precip_in": watershed_precip_in,
                        "SelectedFile": raw_fp.name,
                        "SourceType": source_type,
                        "WeightColUsed": info["weight_col_used"],
                        "LoadMethod": method_used
                    })

            except Exception as e:
                failed_rows.append({
                    "timestamp": ts,
                    "file_name": row["file_name"],
                    "path": row["path"],
                    "source_type": source_type,
                    "compression": compression,
                    "error": str(e)
                })

            finally:
                if temp_created is not None and temp_created.exists():
                    try:
                        temp_created.unlink()
                    except Exception:
                        pass

        for site, rows in records.items():
            if not rows:
                print(f"No rows for {site} in {YEAR}{MONTH}")
                continue

            df = pd.DataFrame(rows).sort_values("DateTimeUTC").drop_duplicates(subset=["DateTimeUTC"])
            out_csv = OUTPUT_DIR / f"{site}_StageIV_hourly_{YEAR}{MONTH}.csv"
            df.to_csv(out_csv, index=False)

        if failed_rows:
            fail_csv = OUTPUT_DIR / f"StageIV_failed_auto_{YEAR}{MONTH}.csv"
            pd.DataFrame(failed_rows).to_csv(fail_csv, index=False)
            print(f"Saved failed-file log to: {fail_csv}")

        runtime_min = round((time.time() - start_time) / 60, 2)
        master_summary_rows.append({
            "year": YEAR,
            "month": MONTH,
            "status": "done",
            "n_candidates": len(cand_df),
            "n_selected": len(selected_df),
            "n_failed": len(failed_rows),
            "runtime_min": runtime_min
        })

# ----------------------------------------------------------
# SAVE MASTER SUMMARY
# ----------------------------------------------------------
summary_df = pd.DataFrame(master_summary_rows)
summary_csv = OUTPUT_DIR / "StageIV_master_summary_auto.csv"
summary_df.to_csv(summary_csv, index=False)

print(f"\nSaved master summary to:\n{summary_csv}")
print("\nDone.")