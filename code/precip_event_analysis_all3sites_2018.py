# ==========================================================
# PRECIP EVENT ANALYSIS FOR 3 SITES, 2018
#
# Input files:
#   Philadelphia_2018.csv
#   Atlanta_2018.csv
#   Austin_2018.csv
#
# Each file must contain:
#   DateTimeUTC
#   Stage IV Precip (in)
#   Point Gage Precip (in)
#
# Features:
#   - Site-specific event definitions
#   - Saves event tables
#   - Generates two figure sets:
#       (1) season-aware figures
#       (2) simplified figures (source only)
# ==========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu, ks_2samp

# ----------------------------------------------------------
# USER INPUTS
# ----------------------------------------------------------
base_dir = "/Users/nayeonkwak/Downloads/PrecipEventComparison2018"
output_dir = os.path.join(base_dir, "outputs_JOO_0.1")
os.makedirs(output_dir, exist_ok=True)

site_files = {
    "Philadelphia": os.path.join(base_dir, "Philadelphia_2018.csv"),
    "Atlanta": os.path.join(base_dir, "Atlanta_2018.csv"),
    "Austin": os.path.join(base_dir, "Austin_2018.csv"),
}

datetime_col = "DateTimeUTC"
stageiv_col = "Stage IV Precip (in)"
point_col = "Point Gage Precip (in)"

# ----------------------------------------------------------
# SITE-SPECIFIC EVENT DEFINITIONS
# ----------------------------------------------------------
# You can change these independently for each site
site_event_settings = {
    "Philadelphia": {
        "IETD_HOURS": 9.24,
        "RAIN_THRESHOLD_IN": 0.0,
        "MIN_EVENT_DEPTH_IN": 0.1,
    },
    "Atlanta": {
        "IETD_HOURS": 10.01,
        "RAIN_THRESHOLD_IN": 0.0,
        "MIN_EVENT_DEPTH_IN": 0.1,
    },
    "Austin": {
        "IETD_HOURS": 5.29,
        "RAIN_THRESHOLD_IN": 0.0,
        "MIN_EVENT_DEPTH_IN": 0.1,
    },
}

# ----------------------------------------------------------
# STYLE MAPS
# ----------------------------------------------------------
source_marker_map = {
    "Stage IV": "o",
    "Point Gage": "^",
}

source_color_map = {
    "Stage IV": "tab:blue",
    "Point Gage": "tab:orange",
}

season_color_map = {
    "Winter": "tab:blue",
    "Spring": "tab:green",
    "Summer": "tab:orange",
    "Fall": "tab:brown",
}

source_color_for_boxplot = {
    "Stage IV": "#4C78A8",
    "Point Gage": "#F58518",
}

site_order = ["Philadelphia", "Atlanta", "Austin"]
source_order = ["Stage IV", "Point Gage"]
season_order = ["Winter", "Spring", "Summer", "Fall"]

# ----------------------------------------------------------
# HELPERS
# ----------------------------------------------------------
def get_season(dt):
    m = dt.month
    if m in [12, 1, 2]:
        return "Winter"
    elif m in [3, 4, 5]:
        return "Spring"
    elif m in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"

def ecdf(values):
    x = np.sort(np.asarray(values))
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y

def build_season_source_legend_handles():
    season_handles = [
        Line2D([0], [0], marker="o", linestyle="None",
               color=season_color_map[s], markersize=8, label=s)
        for s in season_order
    ]
    source_handles = [
        Line2D([0], [0], marker=source_marker_map[src], linestyle="None",
               color="black", markersize=8, label=src)
        for src in source_order
    ]
    return season_handles, source_handles

def build_source_only_legend_handles():
    return [
        Line2D([0], [0], marker=source_marker_map[src], linestyle="None",
               color=source_color_map[src], markersize=8, label=src)
        for src in source_order
    ]

# ----------------------------------------------------------
# EVENT DETECTION
# ----------------------------------------------------------
def detect_events(df, precip_col, source_name, site_name,
                  ietd_hours=6,
                  rain_threshold=0.0,
                  min_event_depth=0.25):
    """
    Event definition:
    - dry periods EXCEEDING ietd_hours separate events
    - dry gaps <= ietd_hours stay in the same event
    - retain only events with cumulative depth >= min_event_depth
    """
    work = df[[datetime_col, precip_col]].copy()
    work = work.rename(columns={precip_col: "precip_in"})
    work = work.sort_values(datetime_col).reset_index(drop=True)

    full_index = pd.date_range(
        work[datetime_col].min(),
        work[datetime_col].max(),
        freq="h"
    )

    work = (
        work.set_index(datetime_col)
        .reindex(full_index)
        .rename_axis(datetime_col)
        .reset_index()
    )

    work["precip_in"] = pd.to_numeric(work["precip_in"], errors="coerce").fillna(0.0)
    work["is_wet"] = work["precip_in"] > rain_threshold

    wet_idx = np.where(work["is_wet"].values)[0]

    if len(wet_idx) == 0:
        return pd.DataFrame(columns=[
            "site", "source", "season", "event_id",
            "event_start", "event_end",
            "duration_hr", "wet_duration_hr",
            "depth_in", "avg_intensity_inhr",
            "peak_hourly_intensity_inhr",
            "dry_time_prior_hr", "time_to_peak_hr",
            "max_internal_dry_gap_hr"
        ])

    raw_events = []
    current_start = wet_idx[0]
    current_end = wet_idx[0]

    for idx in wet_idx[1:]:
        gap_hours = idx - current_end - 1

        if gap_hours <= ietd_hours:
            current_end = idx
        else:
            raw_events.append((current_start, current_end))
            current_start = idx
            current_end = idx

    raw_events.append((current_start, current_end))

    rows = []
    prev_event_end = None
    retained_event_id = 0

    for start_idx, end_idx in raw_events:
        event_df = work.iloc[start_idx:end_idx + 1].copy()

        event_start = event_df[datetime_col].iloc[0]
        event_end = event_df[datetime_col].iloc[-1]
        duration_hr = len(event_df)
        wet_duration_hr = int(event_df["is_wet"].sum())
        depth_in = event_df["precip_in"].sum()
        avg_intensity_inhr = depth_in / duration_hr if duration_hr > 0 else np.nan
        peak_hourly_intensity_inhr = event_df["precip_in"].max()

        dry_time_prior_hr = np.nan
        if prev_event_end is not None:
            dry_time_prior_hr = (event_start - prev_event_end).total_seconds() / 3600.0 - 1

        peak_idx_within_event = event_df["precip_in"].idxmax()
        peak_time = work.loc[peak_idx_within_event, datetime_col]
        time_to_peak_hr = (peak_time - event_start).total_seconds() / 3600.0

        wet_positions = np.where(event_df["is_wet"].values)[0]
        if len(wet_positions) <= 1:
            max_internal_dry_gap_hr = 0
        else:
            internal_gaps = []
            for a, b in zip(wet_positions[:-1], wet_positions[1:]):
                internal_gaps.append(b - a - 1)
            max_internal_dry_gap_hr = max(internal_gaps) if internal_gaps else 0

        prev_event_end = event_end

        if depth_in >= min_event_depth:
            retained_event_id += 1
            rows.append({
                "site": site_name,
                "source": source_name,
                "season": get_season(event_start),
                "event_id": retained_event_id,
                "event_start": event_start,
                "event_end": event_end,
                "duration_hr": duration_hr,
                "wet_duration_hr": wet_duration_hr,
                "depth_in": depth_in,
                "avg_intensity_inhr": avg_intensity_inhr,
                "peak_hourly_intensity_inhr": peak_hourly_intensity_inhr,
                "dry_time_prior_hr": dry_time_prior_hr,
                "time_to_peak_hr": time_to_peak_hr,
                "max_internal_dry_gap_hr": max_internal_dry_gap_hr
            })

    return pd.DataFrame(rows)

# ----------------------------------------------------------
# FIGURE FUNCTIONS — SEASON AWARE
# ----------------------------------------------------------
def make_cdf_figure_by_site_seasonal(events_master, site, metric_col, x_label, title_stub, x_max, output_name):
    fig, ax = plt.subplots(figsize=(7, 5))

    for source in source_order:
        for season in season_order:
            sub = events_master[
                (events_master["site"] == site) &
                (events_master["source"] == source) &
                (events_master["season"] == season)
            ]

            if len(sub) == 0:
                continue

            x, y = ecdf(sub[metric_col].dropna().values)

            ax.scatter(
                x, y,
                color=season_color_map[season],
                marker=source_marker_map[source],
                s=28,
                alpha=0.85
            )

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Cumulative probability")
    ax.set_title(f"{site}: {title_stub}")

    season_handles, source_handles = build_season_source_legend_handles()
    legend1 = ax.legend(handles=season_handles, title="Season", loc="lower right")
    ax.add_artist(legend1)
    ax.legend(handles=source_handles, title="Source", loc="lower center")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, output_name), dpi=300)
    plt.close()

def make_depth_duration_scatter_by_site_seasonal(events_master, site, x_max, y_max, output_name):
    fig, ax = plt.subplots(figsize=(7, 5))
    sub_site = events_master[events_master["site"] == site]

    for source in source_order:
        for season in season_order:
            sub = sub_site[
                (sub_site["source"] == source) &
                (sub_site["season"] == season)
            ]
            if len(sub) == 0:
                continue

            ax.scatter(
                sub["duration_hr"],
                sub["depth_in"],
                color=season_color_map[season],
                marker=source_marker_map[source],
                s=35,
                alpha=0.8
            )

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Event duration (hr)")
    ax.set_ylabel("Event depth (in)")
    ax.set_title(f"{site}: Event Depth vs Duration")

    season_handles, source_handles = build_season_source_legend_handles()
    legend1 = ax.legend(handles=season_handles, title="Season", loc="upper right")
    ax.add_artist(legend1)
    ax.legend(handles=source_handles, title="Source", loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, output_name), dpi=300)
    plt.close()

# ----------------------------------------------------------
# FIGURE FUNCTIONS — SOURCE ONLY
# ----------------------------------------------------------
def make_cdf_figure_by_site_simple(events_master, site, metric_col, x_label, title_stub, x_max, output_name):
    fig, ax = plt.subplots(figsize=(7, 5))

    for source in source_order:
        sub = events_master[
            (events_master["site"] == site) &
            (events_master["source"] == source)
        ]

        if len(sub) == 0:
            continue

        x, y = ecdf(sub[metric_col].dropna().values)

        ax.scatter(
            x, y,
            color=source_color_map[source],
            marker=source_marker_map[source],
            s=28,
            alpha=0.85
        )

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Cumulative probability")
    ax.set_title(f"{site}: {title_stub}")

    ax.legend(handles=build_source_only_legend_handles(), title="Source", loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, output_name), dpi=300)
    plt.close()

def make_depth_duration_scatter_by_site_simple(events_master, site, x_max, y_max, output_name):
    fig, ax = plt.subplots(figsize=(7, 5))
    sub_site = events_master[events_master["site"] == site]

    for source in source_order:
        sub = sub_site[sub_site["source"] == source]
        if len(sub) == 0:
            continue

        ax.scatter(
            sub["duration_hr"],
            sub["depth_in"],
            color=source_color_map[source],
            marker=source_marker_map[source],
            s=35,
            alpha=0.8
        )

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Event duration (hr)")
    ax.set_ylabel("Event depth (in)")
    ax.set_title(f"{site}: Event Depth vs Duration")

    ax.legend(handles=build_source_only_legend_handles(), title="Source", loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, output_name), dpi=300)
    plt.close()

# ----------------------------------------------------------
# BOXPLOT
# ----------------------------------------------------------
def make_time_to_peak_boxplot(events_master, output_name):
    fig, ax = plt.subplots(figsize=(9, 5))

    data = []
    labels = []
    positions = []
    colors = []

    pos = 1
    for site in site_order:
        for source in source_order:
            sub = events_master[
                (events_master["site"] == site) &
                (events_master["source"] == source)
            ]["time_to_peak_hr"].dropna().values

            data.append(sub)
            labels.append(f"{site}\n{source}")
            positions.append(pos)
            colors.append(source_color_for_boxplot[source])
            pos += 1

        pos += 0.5

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=True
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Time to peak (hr from event start)")
    ax.set_title("Time to Peak by Site and Source, 2018")

    source_handles = [
        Patch(facecolor=source_color_for_boxplot[src], edgecolor="black", alpha=0.7, label=src)
        for src in source_order
    ]
    ax.legend(handles=source_handles, title="Source")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, output_name), dpi=300)
    plt.close()

# ----------------------------------------------------------
# READ + CLEAN INPUT
# ----------------------------------------------------------
all_events = []

for site_name, file_path in site_files.items():
    print(f"\nProcessing {site_name}...")

    settings = site_event_settings[site_name]

    df = pd.read_csv(file_path)
    df[datetime_col] = pd.to_datetime(df[datetime_col], utc=True, errors="coerce")
    df = df.dropna(subset=[datetime_col]).sort_values(datetime_col).reset_index(drop=True)

    df[stageiv_col] = pd.to_numeric(df[stageiv_col], errors="coerce").fillna(0.0)
    df[point_col] = pd.to_numeric(df[point_col], errors="coerce").fillna(0.0)

    stage_events = detect_events(
        df=df,
        precip_col=stageiv_col,
        source_name="Stage IV",
        site_name=site_name,
        ietd_hours=settings["IETD_HOURS"],
        rain_threshold=settings["RAIN_THRESHOLD_IN"],
        min_event_depth=settings["MIN_EVENT_DEPTH_IN"]
    )

    point_events = detect_events(
        df=df,
        precip_col=point_col,
        source_name="Point Gage",
        site_name=site_name,
        ietd_hours=settings["IETD_HOURS"],
        rain_threshold=settings["RAIN_THRESHOLD_IN"],
        min_event_depth=settings["MIN_EVENT_DEPTH_IN"]
    )

    stage_out = os.path.join(output_dir, f"{site_name}_stageiv_events_2018.csv")
    point_out = os.path.join(output_dir, f"{site_name}_pointgage_events_2018.csv")
    stage_events.to_csv(stage_out, index=False)
    point_events.to_csv(point_out, index=False)

    print(f"Saved: {stage_out}")
    print(f"Saved: {point_out}")
    print(f"{site_name} Stage IV events: {len(stage_events)}")
    print(f"{site_name} Point Gage events: {len(point_events)}")

    all_events.append(stage_events)
    all_events.append(point_events)

events_master = pd.concat(all_events, ignore_index=True)
master_csv = os.path.join(output_dir, "all_events_master_2018.csv")
events_master.to_csv(master_csv, index=False)
print(f"\nSaved master event table: {master_csv}")

# ----------------------------------------------------------
# SUMMARY TABLE
# ----------------------------------------------------------
summary = (
    events_master
    .groupby(["site", "source"], as_index=False)
    .agg(
        n_events=("event_id", "count"),
        mean_depth_in=("depth_in", "mean"),
        median_depth_in=("depth_in", "median"),
        mean_duration_hr=("duration_hr", "mean"),
        median_duration_hr=("duration_hr", "median"),
        mean_peak_hourly_inhr=("peak_hourly_intensity_inhr", "mean"),
        median_peak_hourly_inhr=("peak_hourly_intensity_inhr", "median"),
        mean_time_to_peak_hr=("time_to_peak_hr", "mean"),
        median_time_to_peak_hr=("time_to_peak_hr", "median")
    )
)

summary_csv = os.path.join(output_dir, "event_summary_by_site_source_2018.csv")
summary.to_csv(summary_csv, index=False)
print(f"Saved summary table: {summary_csv}")

# ----------------------------------------------------------
# GLOBAL AXIS LIMITS
# ----------------------------------------------------------
global_depth_max = np.ceil(events_master["depth_in"].max() * 1.05) if len(events_master) else 1
global_duration_max = np.ceil(events_master["duration_hr"].max() * 1.05) if len(events_master) else 1
global_time_to_peak_max = np.ceil(events_master["time_to_peak_hr"].max() * 1.05) if len(events_master) else 1

# ----------------------------------------------------------
# SEASON-AWARE FIGURES
# ----------------------------------------------------------
for site in site_order:
    make_cdf_figure_by_site_seasonal(
        events_master, site,
        metric_col="depth_in",
        x_label="Event depth (in)",
        title_stub="CDF of Event Depth (Seasonal)",
        x_max=global_depth_max,
        output_name=f"{site}_CDF_event_depth_2018_seasonal.png"
    )

    make_cdf_figure_by_site_seasonal(
        events_master, site,
        metric_col="duration_hr",
        x_label="Event duration (hr)",
        title_stub="CDF of Event Duration (Seasonal)",
        x_max=global_duration_max,
        output_name=f"{site}_CDF_event_duration_2018_seasonal.png"
    )

    make_cdf_figure_by_site_seasonal(
        events_master, site,
        metric_col="time_to_peak_hr",
        x_label="Time to peak (hr from event start)",
        title_stub="CDF of Time to Peak (Seasonal)",
        x_max=global_time_to_peak_max,
        output_name=f"{site}_CDF_time_to_peak_2018_seasonal.png"
    )

    make_depth_duration_scatter_by_site_seasonal(
        events_master, site,
        x_max=global_duration_max,
        y_max=global_depth_max,
        output_name=f"{site}_Depth_vs_Duration_2018_seasonal.png"
    )

# ----------------------------------------------------------
# SIMPLIFIED FIGURES (SOURCE ONLY)
# ----------------------------------------------------------
for site in site_order:
    make_cdf_figure_by_site_simple(
        events_master, site,
        metric_col="depth_in",
        x_label="Event depth (in)",
        title_stub="CDF of Event Depth",
        x_max=global_depth_max,
        output_name=f"{site}_CDF_event_depth_2018_simple.png"
    )

    make_cdf_figure_by_site_simple(
        events_master, site,
        metric_col="duration_hr",
        x_label="Event duration (hr)",
        title_stub="CDF of Event Duration",
        x_max=global_duration_max,
        output_name=f"{site}_CDF_event_duration_2018_simple.png"
    )

    make_cdf_figure_by_site_simple(
        events_master, site,
        metric_col="time_to_peak_hr",
        x_label="Time to peak (hr from event start)",
        title_stub="CDF of Time to Peak",
        x_max=global_time_to_peak_max,
        output_name=f"{site}_CDF_time_to_peak_2018_simple.png"
    )

    make_depth_duration_scatter_by_site_simple(
        events_master, site,
        x_max=global_duration_max,
        y_max=global_depth_max,
        output_name=f"{site}_Depth_vs_Duration_2018_simple.png"
    )

# ----------------------------------------------------------
# BOXPLOT
# ----------------------------------------------------------
make_time_to_peak_boxplot(
    events_master=events_master,
    output_name="AllSites_Boxplot_TimeToPeak_2018.png"
)

# ----------------------------------------------------------
# 3-PANEL FIGURE FUNCTIONS — SEASONAL
# ----------------------------------------------------------
def make_cdf_3panel_seasonal(events_master, metric_col, x_label, title_text, x_max, output_name):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax, site in zip(axes, site_order):
        for source in source_order:
            for season in season_order:
                sub = events_master[
                    (events_master["site"] == site) &
                    (events_master["source"] == source) &
                    (events_master["season"] == season)
                ]

                if len(sub) == 0:
                    continue

                x, y = ecdf(sub[metric_col].dropna().values)

                ax.scatter(
                    x, y,
                    color=season_color_map[season],
                    marker=source_marker_map[source],
                    s=24,
                    alpha=0.85
                )

        ax.set_xlim(0, x_max)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(x_label)
        ax.set_title(site)

    axes[0].set_ylabel("Cumulative probability")

    season_handles, source_handles = build_season_source_legend_handles()
    legend1 = fig.legend(handles=season_handles, title="Season",
                         loc="lower center", bbox_to_anchor=(0.40, -0.02), ncol=4)
    fig.add_artist(legend1)
    fig.legend(handles=source_handles, title="Source",
               loc="lower center", bbox_to_anchor=(0.86, -0.02), ncol=2)

    fig.suptitle(title_text, y=0.98)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(os.path.join(output_dir, output_name), dpi=300, bbox_inches="tight")
    plt.close()

def make_depth_duration_3panel_seasonal(events_master, x_max, y_max, title_text, output_name):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax, site in zip(axes, site_order):
        sub_site = events_master[events_master["site"] == site]

        for source in source_order:
            for season in season_order:
                sub = sub_site[
                    (sub_site["source"] == source) &
                    (sub_site["season"] == season)
                ]
                if len(sub) == 0:
                    continue

                ax.scatter(
                    sub["duration_hr"],
                    sub["depth_in"],
                    color=season_color_map[season],
                    marker=source_marker_map[source],
                    s=28,
                    alpha=0.8
                )

        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
        ax.set_xlabel("Event duration (hr)")
        ax.set_title(site)

    axes[0].set_ylabel("Event depth (in)")

    season_handles, source_handles = build_season_source_legend_handles()
    legend1 = fig.legend(handles=season_handles, title="Season",
                         loc="lower center", bbox_to_anchor=(0.40, -0.02), ncol=4)
    fig.add_artist(legend1)
    fig.legend(handles=source_handles, title="Source",
               loc="lower center", bbox_to_anchor=(0.86, -0.02), ncol=2)

    fig.suptitle(title_text, y=0.98)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(os.path.join(output_dir, output_name), dpi=300, bbox_inches="tight")
    plt.close()

# ----------------------------------------------------------
# 3-PANEL FIGURE FUNCTIONS — SIMPLE / SOURCE ONLY
# ----------------------------------------------------------
def make_cdf_3panel_simple(events_master, metric_col, x_label, title_text, x_max, output_name):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax, site in zip(axes, site_order):
        for source in source_order:
            sub = events_master[
                (events_master["site"] == site) &
                (events_master["source"] == source)
            ]

            if len(sub) == 0:
                continue

            x, y = ecdf(sub[metric_col].dropna().values)

            ax.scatter(
                x, y,
                color=source_color_map[source],
                marker=source_marker_map[source],
                s=24,
                alpha=0.85
            )

        ax.set_xlim(0, x_max)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(x_label)
        ax.set_title(site)

    axes[0].set_ylabel("Cumulative probability")

    fig.legend(
        handles=build_source_only_legend_handles(),
        title="Source",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2
    )

    fig.suptitle(title_text, y=0.98)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(os.path.join(output_dir, output_name), dpi=300, bbox_inches="tight")
    plt.close()

def make_depth_duration_3panel_simple(events_master, x_max, y_max, title_text, output_name):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax, site in zip(axes, site_order):
        sub_site = events_master[events_master["site"] == site]

        for source in source_order:
            sub = sub_site[sub_site["source"] == source]
            if len(sub) == 0:
                continue

            ax.scatter(
                sub["duration_hr"],
                sub["depth_in"],
                color=source_color_map[source],
                marker=source_marker_map[source],
                s=28,
                alpha=0.8
            )

        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
        ax.set_xlabel("Event duration (hr)")
        ax.set_title(site)

    axes[0].set_ylabel("Event depth (in)")

    fig.legend(
        handles=build_source_only_legend_handles(),
        title="Source",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2
    )

    fig.suptitle(title_text, y=0.98)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(os.path.join(output_dir, output_name), dpi=300, bbox_inches="tight")
    plt.close()

# ----------------------------------------------------------
# 3-PANEL FIGURES — SEASONAL
# ----------------------------------------------------------
make_cdf_3panel_seasonal(
    events_master=events_master,
    metric_col="depth_in",
    x_label="Event depth (in)",
    title_text="CDF of Event Depth by Watershed (Seasonal), 2018",
    x_max=global_depth_max,
    output_name="AllSites_3Panel_CDF_EventDepth_2018_seasonal.png"
)

make_cdf_3panel_seasonal(
    events_master=events_master,
    metric_col="duration_hr",
    x_label="Event duration (hr)",
    title_text="CDF of Event Duration by Watershed (Seasonal), 2018",
    x_max=global_duration_max,
    output_name="AllSites_3Panel_CDF_EventDuration_2018_seasonal.png"
)

make_cdf_3panel_seasonal(
    events_master=events_master,
    metric_col="time_to_peak_hr",
    x_label="Time to peak (hr from event start)",
    title_text="CDF of Time to Peak by Watershed (Seasonal), 2018",
    x_max=global_time_to_peak_max,
    output_name="AllSites_3Panel_CDF_TimeToPeak_2018_seasonal.png"
)

make_depth_duration_3panel_seasonal(
    events_master=events_master,
    x_max=global_duration_max,
    y_max=global_depth_max,
    title_text="Event Depth vs Duration by Watershed (Seasonal), 2018",
    output_name="AllSites_3Panel_DepthVsDuration_2018_seasonal.png"
)

# ----------------------------------------------------------
# 3-PANEL FIGURES — SIMPLE / SOURCE ONLY
# ----------------------------------------------------------
make_cdf_3panel_simple(
    events_master=events_master,
    metric_col="depth_in",
    x_label="Event depth (in)",
    title_text="CDF of Event Depth by Watershed, 2018",
    x_max=global_depth_max,
    output_name="AllSites_3Panel_CDF_EventDepth_2018_simple.png"
)

make_cdf_3panel_simple(
    events_master=events_master,
    metric_col="duration_hr",
    x_label="Event duration (hr)",
    title_text="CDF of Event Duration by Watershed, 2018",
    x_max=global_duration_max,
    output_name="AllSites_3Panel_CDF_EventDuration_2018_simple.png"
)

make_cdf_3panel_simple(
    events_master=events_master,
    metric_col="time_to_peak_hr",
    x_label="Time to peak (hr from event start)",
    title_text="CDF of Time to Peak by Watershed, 2018",
    x_max=global_time_to_peak_max,
    output_name="AllSites_3Panel_CDF_TimeToPeak_2018_simple.png"
)

make_depth_duration_3panel_simple(
    events_master=events_master,
    x_max=global_duration_max,
    y_max=global_depth_max,
    title_text="Event Depth vs Duration by Watershed, 2018",
    output_name="AllSites_3Panel_DepthVsDuration_2018_simple.png"
)

# ----------------------------------------------------------
# STATISTICAL COMPARISONS
# ----------------------------------------------------------
def calculate_significance_tests(events_master, output_dir, alpha=0.05):
    """
    Compare Stage IV vs Point Gage event metrics for each site.

    Tests used:
    - Mann-Whitney U test
    - Kolmogorov-Smirnov two-sample test

    Output:
    - CSV summary of p-values and significance flags
    """
    metrics_to_test = {
        "depth_in": "Event Depth (in)",
        "duration_hr": "Event Duration (hr)",
        "time_to_peak_hr": "Time to Peak (hr)",
        "avg_intensity_inhr": "Average Intensity (in/hr)",
        "peak_hourly_intensity_inhr": "Peak Hourly Intensity (in/hr)",
        "dry_time_prior_hr": "Dry Time Prior (hr)",
        "wet_duration_hr": "Wet Duration (hr)",
        "max_internal_dry_gap_hr": "Max Internal Dry Gap (hr)",
    }

    rows = []

    for site in site_order:
        site_df = events_master[events_master["site"] == site]

        stage_df = site_df[site_df["source"] == "Stage IV"]
        point_df = site_df[site_df["source"] == "Point Gage"]

        for metric_col, metric_label in metrics_to_test.items():
            x = stage_df[metric_col].dropna().values
            y = point_df[metric_col].dropna().values

            row = {
                "site": site,
                "metric_column": metric_col,
                "metric_label": metric_label,
                "n_stageiv": len(x),
                "n_pointgage": len(y),
                "stageiv_mean": np.mean(x) if len(x) > 0 else np.nan,
                "pointgage_mean": np.mean(y) if len(y) > 0 else np.nan,
                "stageiv_median": np.median(x) if len(x) > 0 else np.nan,
                "pointgage_median": np.median(y) if len(y) > 0 else np.nan,
            }

            if len(x) >= 2 and len(y) >= 2:
                # Mann-Whitney U
                try:
                    mw_stat, mw_p = mannwhitneyu(x, y, alternative="two-sided")
                except Exception:
                    mw_stat, mw_p = np.nan, np.nan

                # KS test
                try:
                    ks_stat, ks_p = ks_2samp(x, y, alternative="two-sided", mode="auto")
                except Exception:
                    ks_stat, ks_p = np.nan, np.nan
            else:
                mw_stat, mw_p = np.nan, np.nan
                ks_stat, ks_p = np.nan, np.nan

            row["mannwhitney_u_stat"] = mw_stat
            row["mannwhitney_p"] = mw_p
            row["mannwhitney_significant_alpha_0.05"] = (
                bool(mw_p < alpha) if pd.notna(mw_p) else np.nan
            )

            row["ks_stat"] = ks_stat
            row["ks_p"] = ks_p
            row["ks_significant_alpha_0.05"] = (
                bool(ks_p < alpha) if pd.notna(ks_p) else np.nan
            )

            rows.append(row)

    stats_df = pd.DataFrame(rows)
    out_csv = os.path.join(output_dir, "event_metric_significance_by_site_2018.csv")
    stats_df.to_csv(out_csv, index=False)
    print(f"Saved statistical significance table: {out_csv}")

    return stats_df

# ----------------------------------------------------------
# STATISTICAL SIGNIFICANCE TESTS
# ----------------------------------------------------------
stats_df = calculate_significance_tests(
    events_master=events_master,
    output_dir=output_dir,
    alpha=0.05
)

print("\nStatistical significance summary (Mann-Whitney p-values):")
print(stats_df[["site", "metric_label", "mannwhitney_p", "mannwhitney_significant_alpha_0.05"]])


print("\nDone.")