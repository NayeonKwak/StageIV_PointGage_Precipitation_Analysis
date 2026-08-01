# ==========================================================
# GRIDDED VS POINT GAGE PRECIP FIGURES
# Generates:
#   - Annual 1:1 scatter, US + metric
#   - Annual difference bars + percent difference dotted lines, US + metric
#       with percent y-axis 0–50% and 0–100%
#   - Event duration Q-Q plot
#   - Event metric boxplots, US + metric for precip metrics
#     and one version for time-only metrics
# ==========================================================

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# USER SETTINGS
# ==========================================================

BASE_DIR = Path("/Volumes/StageIV/StageIV2017-2025/paired_site_files")
EVENT_DIR = BASE_DIR / "event_analysis_WY2018_2025"
OUT_DIR = BASE_DIR / "paper_final_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVENT_FILE = EVENT_DIR / "all_events_master_WY2018_2025.csv"

PAIRED_FILES = {
    "Austin": BASE_DIR / "Austin_PointGage_StageIV_paired.csv",
    "Atlanta": BASE_DIR / "Atlanta_PointGage_StageIV_paired.csv",
    "Philadelphia": BASE_DIR / "Philadelphia_PointGage_StageIV_paired.csv",
}

SITE_ORDER = ["Austin", "Atlanta", "Philadelphia"]
SOURCE_ORDER = ["Stage IV", "Point Gage"]

WY_START = 2018
WY_END = 2025

# Dark = Stage IV / lines; light = point gage / bars
COLORS = {
    ("Austin", "Stage IV"): "#9BBB59",
    ("Austin", "Point Gage"): "#D8E9C6",
    ("Atlanta", "Stage IV"): "#7030A0",
    ("Atlanta", "Point Gage"): "#B4A7FF",
    ("Philadelphia", "Stage IV"): "#2F70C0",
    ("Philadelphia", "Point Gage"): "#B8D2EE",
}

SITE_DARK = {
    "Austin": "#6E9E2E",
    "Atlanta": "#5B1E8C",
    "Philadelphia": "#0F5BB5",
}

SITE_LIGHT = {
    "Austin": "#D8E9C6",
    "Atlanta": "#D8CCF4",
    "Philadelphia": "#B8D2EE",
}

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 1.0,
    "savefig.dpi": 300,
})

# ==========================================================
# HELPERS
# ==========================================================

def normalize_col(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())

def find_col(df, possible_names):
    norm_lookup = {normalize_col(c): c for c in df.columns}

    for name in possible_names:
        key = normalize_col(name)
        if key in norm_lookup:
            return norm_lookup[key]

    for name in possible_names:
        key = normalize_col(name)
        for norm, original in norm_lookup.items():
            if key in norm or norm in key:
                return original

    raise KeyError(
        f"Could not find any of these columns:\n{possible_names}\n\n"
        f"Available columns are:\n{list(df.columns)}"
    )

def read_paired_site_file(site):
    path = PAIRED_FILES[site]
    if not path.exists():
        raise FileNotFoundError(f"Missing paired file for {site}:\n{path}")

    print(f"\nReading paired file for {site}: {path.name}")
    df = pd.read_csv(path)
    print("Columns:", list(df.columns))

    dt_col = find_col(df, [
        "DateTimeUTC", "Date Time UTC", "DatetimeUTC", "datetime_utc",
        "DateTime", "Datetime", "timestamp", "Timestamp"
    ])

    stage_col = find_col(df, [
        "Stage IV Precip (in)", "StageIV Precip (in)", "Stage IV",
        "StageIV", "Stage_IV", "Stage IV Total (in)",
        "stageiv_total_in", "stageiv_precip_in"
    ])

    point_col = find_col(df, [
        "Point Gage Precip (in)", "Point Gauge Precip (in)",
        "Point Gage", "PointGage", "Point_Gage",
        "Point Gage Total (in)", "point_total_in", "pointgage_precip_in"
    ])

    df[dt_col] = pd.to_datetime(df[dt_col], utc=True, errors="coerce")
    df = df.dropna(subset=[dt_col]).copy()

    df[stage_col] = pd.to_numeric(df[stage_col], errors="coerce").fillna(0.0)
    df[point_col] = pd.to_numeric(df[point_col], errors="coerce").fillna(0.0)

    clean = pd.DataFrame({
        "DateTimeUTC": df[dt_col],
        "Stage IV Precip (in)": df[stage_col],
        "Point Gage Precip (in)": df[point_col],
        "site": site,
    })

    clean["water_year"] = (
        clean["DateTimeUTC"].dt.year
        + (clean["DateTimeUTC"].dt.month >= 10).astype(int)
    )

    clean = clean[clean["water_year"].between(WY_START, WY_END)].copy()
    return clean

def build_annual_totals():
    rows = []

    for site in SITE_ORDER:
        df = read_paired_site_file(site)

        annual = (
            df.groupby("water_year", as_index=False)
            .agg(
                stageiv_total_in=("Stage IV Precip (in)", "sum"),
                point_total_in=("Point Gage Precip (in)", "sum"),
            )
        )

        annual["site"] = site
        annual["diff_in"] = annual["stageiv_total_in"] - annual["point_total_in"]
        annual["pct_diff"] = np.where(
            annual["point_total_in"] != 0,
            annual["diff_in"] / annual["point_total_in"] * 100,
            np.nan
        )

        rows.append(annual)

    annual_all = pd.concat(rows, ignore_index=True)
    annual_all = annual_all[annual_all["water_year"].between(WY_START, WY_END)].copy()

    annual_all.to_csv(OUT_DIR / "annual_totals_WY2018_2025_generated.csv", index=False)
    return annual_all

def quantile_compare(x, y, n_quantiles=100):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) == 0 or len(y) == 0:
        return np.array([]), np.array([])

    qs = np.linspace(0.01, 0.99, n_quantiles)
    return np.quantile(x, qs), np.quantile(y, qs)

def safe_metric_values(events, site, source, metric_col):
    sub = events[(events["site"] == site) & (events["source"] == source)]

    if metric_col not in sub.columns:
        raise KeyError(f"Metric column missing from event file: {metric_col}")

    return pd.to_numeric(sub[metric_col], errors="coerce").dropna().values

# ==========================================================
# LOAD DATA
# ==========================================================

print("\nLoading event file...")
if not EVENT_FILE.exists():
    raise FileNotFoundError(f"Missing event file:\n{EVENT_FILE}")

events = pd.read_csv(EVENT_FILE)

events["site"] = events["site"].replace({
    "Austin, TX": "Austin",
    "Atlanta, GA": "Atlanta",
    "Philadelphia, PA": "Philadelphia",
})

events["source"] = events["source"].replace({
    "StageIV": "Stage IV",
    "Stage IV": "Stage IV",
    "Point": "Point Gage",
    "PointGage": "Point Gage",
    "Point Gage": "Point Gage",
})

print("Event file sites:", sorted(events["site"].dropna().unique()))
print("Event file sources:", sorted(events["source"].dropna().unique()))

annual_all = build_annual_totals()

# ==========================================================
# UNIT SETTINGS
# ==========================================================

UNIT_SYSTEMS = {
    "US": {
        "factor": 1.0,
        "depth_unit": "in",
        "intensity_unit": "in/hr",
    },
    "METRIC": {
        "factor": 2.54,
        "depth_unit": "cm",
        "intensity_unit": "cm/hr",
    }
}

# ==========================================================
# FIGURE 1: ANNUAL 1:1 SCATTER — NO YEAR LABELS
# ==========================================================

def make_annual_one_to_one(unit_key):
    cfg = UNIT_SYSTEMS[unit_key]
    factor = cfg["factor"]
    unit = cfg["depth_unit"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)

    global_max = max(
        annual_all["stageiv_total_in"].max() * factor,
        annual_all["point_total_in"].max() * factor
    ) * 1.08

    for ax, site in zip(axes, SITE_ORDER):
        sub = annual_all[annual_all["site"] == site].sort_values("water_year")

        ax.scatter(
            sub["point_total_in"] * factor,
            sub["stageiv_total_in"] * factor,
            s=62,
            color=SITE_DARK[site],
            alpha=0.9,
            edgecolor="white",
            linewidth=0.7
        )

        ax.plot([0, global_max], [0, global_max], "--", color="0.35", linewidth=1.3)

        ax.set_title(site)
        ax.set_xlim(0, global_max)
        ax.set_ylim(0, global_max)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel(f"Point Gage Total ({unit})")

    axes[0].set_ylabel(f"Stage IV Total ({unit})")
    fig.suptitle("Water Year Totals: Stage IV vs. Point Gage", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(
        OUT_DIR / f"Annual_OneToOne_3Panel_{unit_key}_NO_YEAR_LABELS.png",
        bbox_inches="tight",
        dpi=300
    )
    plt.close()

# ==========================================================
# FIGURE 2: ANNUAL DIFFERENCE BARS + PERCENT DIFFERENCE LINES
# ==========================================================

PCT_AXIS_OPTIONS = {
    "PCT_0_50": (0, 50),
    "PCT_0_100": (0, 100),
}

def make_annual_difference_bar_line(unit_key, pct_axis_key):
    cfg = UNIT_SYSTEMS[unit_key]
    factor = cfg["factor"]
    unit = cfg["depth_unit"]
    pct_ylim = PCT_AXIS_OPTIONS[pct_axis_key]

    fig, ax1 = plt.subplots(figsize=(10.5, 5.2))
    ax2 = ax1.twinx()

    years = sorted(annual_all["water_year"].unique())
    x = np.arange(len(years))

    bar_width = 0.22
    offsets = {
        "Austin": -bar_width,
        "Atlanta": 0,
        "Philadelphia": bar_width,
    }

    all_diff = annual_all["diff_in"].values * factor
    ymin = np.nanmin(all_diff)
    ymax = np.nanmax(all_diff)
    yrange = ymax - ymin if ymax != ymin else 1
    ypad = 0.12 * yrange

    # Always show a negative portion on the difference axis
    y_lower = min(0, ymin) - ypad
    y_upper = ymax + ypad

    for site in SITE_ORDER:
        sub = annual_all[annual_all["site"] == site].sort_values("water_year")
        xpos = x + offsets[site]

        ax1.bar(
            xpos,
            sub["diff_in"] * factor,
            width=bar_width,
            color=SITE_LIGHT[site],
            edgecolor=SITE_DARK[site],
            linewidth=1.0,
            alpha=0.95,
            label=f"{site}: Δ {unit}",
            zorder=2
        )

        # clip_on=False keeps circles fully visible at y=0 boundary
        ax2.plot(
            xpos,
            sub["pct_diff"],
            linestyle=":",
            linewidth=2.8,
            marker="o",
            markersize=8.5,
            color=SITE_DARK[site],
            markerfacecolor=SITE_DARK[site],
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=f"{site}: Δ %",
            zorder=5,
            clip_on=False
        )

    ax1.axhline(0, color="0.35", linewidth=1.0, zorder=1)

    ax1.set_xticks(x)
    ax1.set_xticklabels(years)
    ax1.set_xlabel("Water Year")
    ax1.set_ylabel(f"Difference: Stage IV − Point Gage ({unit})")
    ax2.set_ylabel("Percent Difference (%)")

    ax1.set_ylim(y_lower, y_upper)
    ax2.set_ylim(pct_ylim)

    ax1.grid(axis="y", alpha=0.25)

    ax1.set_title(
        "Annual Difference and Percent Difference by Water Year",
        fontsize=14,
        fontweight="bold",
        pad=12
    )

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()

    ax2.legend(
        h1 + h2,
        l1 + l2,
        loc="upper left",
        bbox_to_anchor=(1.03, 1.0),
        frameon=False
    )

    plt.tight_layout()

    outname = f"Annual_Difference_Bars_PercentDifference_Line_{unit_key}_{pct_axis_key}.png"
    plt.savefig(OUT_DIR / outname, bbox_inches="tight", dpi=300)
    plt.close()

# ==========================================================
# FIGURE 3: EVENT DURATION Q-Q — TIME ONLY
# ==========================================================

def make_duration_qq():
    duration_col = "duration_hr"

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)

    qq_max = 0.0
    for site in SITE_ORDER:
        stage_vals = safe_metric_values(events, site, "Stage IV", duration_col)
        point_vals = safe_metric_values(events, site, "Point Gage", duration_col)
        xq, yq = quantile_compare(point_vals, stage_vals)
        if len(xq) > 0:
            qq_max = max(qq_max, np.nanmax(xq), np.nanmax(yq))

    qq_max *= 1.08

    for ax, site in zip(axes, SITE_ORDER):
        stage_vals = safe_metric_values(events, site, "Stage IV", duration_col)
        point_vals = safe_metric_values(events, site, "Point Gage", duration_col)
        xq, yq = quantile_compare(point_vals, stage_vals)

        ax.scatter(
            xq, yq,
            s=22,
            color=SITE_DARK[site],
            alpha=0.82,
            edgecolor="white",
            linewidth=0.3
        )

        ax.plot([0, qq_max], [0, qq_max], "--", color="0.35", linewidth=1.2)

        ax.set_title(site)
        ax.set_xlim(0, qq_max)
        ax.set_ylim(0, qq_max)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("Point Gage Event Duration Quantiles (hr)")

    axes[0].set_ylabel("Stage IV Event Duration Quantiles (hr)")
    fig.suptitle("Event Duration Q-Q Comparison", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "QQ_EventDuration_3Panel_hr.png", bbox_inches="tight", dpi=300)
    plt.close()

# ==========================================================
# FIGURE 4: EVENT METRIC BOXPLOTS
# ==========================================================

METRICS = {
    "depth_in": {
        "label_us": "Event Depth (in)",
        "label_metric": "Event Depth (cm)",
        "factor_metric": 2.54,
        "unit_type": "precip"
    },
    "duration_hr": {
        "label_us": "Event Duration (hr)",
        "label_metric": "Event Duration (hr)",
        "factor_metric": 1.0,
        "unit_type": "time"
    },
    "time_to_peak_hr": {
        "label_us": "Time to Peak Rainfall Intensity (hr)",
        "label_metric": "Time to Peak Rainfall Intensity (hr)",
        "factor_metric": 1.0,
        "unit_type": "time"
    },
    "avg_intensity_inhr": {
        "label_us": "Average Intensity (in/hr)",
        "label_metric": "Average Intensity (cm/hr)",
        "factor_metric": 2.54,
        "unit_type": "precip"
    },
    "peak_hourly_intensity_inhr": {
        "label_us": "Peak Hourly Intensity (in/hr)",
        "label_metric": "Peak Hourly Intensity (cm/hr)",
        "factor_metric": 2.54,
        "unit_type": "precip"
    },
    "dry_time_prior_hr": {
        "label_us": "Dry Time Prior (hr)",
        "label_metric": "Dry Time Prior (hr)",
        "factor_metric": 1.0,
        "unit_type": "time"
    },
    "wet_duration_hr": {
        "label_us": "Wet Duration (hr)",
        "label_metric": "Wet Duration (hr)",
        "factor_metric": 1.0,
        "unit_type": "time"
    },
    "max_internal_dry_gap_hr": {
        "label_us": "Max Internal Dry Gap (hr)",
        "label_metric": "Max Internal Dry Gap (hr)",
        "factor_metric": 1.0,
        "unit_type": "time"
    },
}

def make_metric_boxplot(metric_col, info, unit_key, log_scale=False):
    if unit_key == "US":
        factor = 1.0
        ylabel = info["label_us"]
    elif unit_key == "METRIC":
        factor = info["factor_metric"]
        ylabel = info["label_metric"]
    else:
        raise ValueError("unit_key must be US or METRIC")

    data = []
    labels = []
    box_colors = []
    positions = []

    pos = 1.0

    for site in SITE_ORDER:
        for source in SOURCE_ORDER:
            vals = safe_metric_values(events, site, source, metric_col)
            vals = vals * factor

            if log_scale:
                vals = vals[vals > 0]

            data.append(vals)
            labels.append(f"{site}\n{source}")
            box_colors.append(COLORS[(site, source)])
            positions.append(pos)

            pos += 0.75

        pos += 0.42

    fig, ax = plt.subplots(figsize=(7.3, 4.7))

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.45,
        patch_artist=True,
        showfliers=True,
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(color="0.4", linewidth=0.9),
        capprops=dict(color="0.4", linewidth=0.9),
        boxprops=dict(color="0.4", linewidth=0.9),
        flierprops=dict(
            marker="o",
            markersize=2.2,
            markerfacecolor="white",
            markeredgecolor="0.55",
            alpha=0.45
        )
    )

    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.88)

    if log_scale:
        ax.set_yscale("log")
        scale_text = "Log Scale"
    else:
        scale_text = "Regular Scale"

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8.1)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by Site and Source ({scale_text})", fontsize=12.5, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)

    for sep in [(positions[1] + positions[2]) / 2, (positions[3] + positions[4]) / 2]:
        ax.axvline(sep, color="0.82", linewidth=0.8)

    plt.tight_layout()

    scale_suffix = "LOG" if log_scale else "REGULAR"
    outname = f"Boxplot_{metric_col}_{unit_key}_{scale_suffix}.png"

    plt.savefig(OUT_DIR / outname, bbox_inches="tight", dpi=300)
    plt.close()

# ==========================================================
# GENERATE EVERYTHING
# ==========================================================

for unit_key in ["US", "METRIC"]:
    make_annual_one_to_one(unit_key)

    for pct_axis_key in ["PCT_0_50", "PCT_0_100"]:
        make_annual_difference_bar_line(unit_key, pct_axis_key)

# Time-only Q-Q once
make_duration_qq()

for metric_col, info in METRICS.items():
    if metric_col not in events.columns:
        print(f"Skipping missing metric column: {metric_col}")
        continue

    # US version always
    make_metric_boxplot(metric_col, info, "US", log_scale=False)

    vals_all = pd.to_numeric(events[metric_col], errors="coerce").dropna()
    if (vals_all > 0).any():
        make_metric_boxplot(metric_col, info, "US", log_scale=True)

    # Metric version only for precipitation-based metrics
    if info["unit_type"] == "precip":
        make_metric_boxplot(metric_col, info, "METRIC", log_scale=False)

        if (vals_all > 0).any():
            make_metric_boxplot(metric_col, info, "METRIC", log_scale=True)

print("\nDONE — generated all figures.")
print(OUT_DIR)