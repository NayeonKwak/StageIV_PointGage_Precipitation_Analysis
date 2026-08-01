from __future__ import annotations

from pathlib import Path
import re
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


# ============================================================
# USER SETTINGS
# ============================================================

BASE_DIR = Path("/Users/nayeonkwak/Downloads")

EVENTS_CSV = (
    BASE_DIR
    / "event_outputs"
    / "stageiv_events_2018_ALLSITES_ALLDEFS.csv"
)

PRECIP_DIR = BASE_DIR / "precip2018_UTC_bothsources"
FLOW_DIR = BASE_DIR / "WDFN_flow_clean_UTC"

# New output directory so earlier figures are preserved
OUT_DIR = BASE_DIR / "event_figures_metadata_strip"

# Must exactly match a def_tag in EVENTS_CSV
DEF_TAG_TO_PLOT = "P0p10_PD12_IET6"

# Time displayed before and after the Stage IV-defined event
PAD_BEFORE_H = 12
PAD_AFTER_H = 24

# Point-gage cumulative precipitation window around Stage IV event
GAGE_PAD_H = 5

# Check for Stage IV precipitation after the event
CHECK_AFTER_H = [6, 12]

# Sample discharge after precipitation ends
FLOW_AFTER_H = [1, 3, 6]

# False retains 15-minute discharge resolution
RESAMPLE_FLOW_TO_HOURLY = False

# Threshold for identifying measurable post-event precipitation
RAIN_THRESHOLD_IN = 0.001

# Figure settings
FIG_WIDTH_IN = 10.0
FIG_HEIGHT_IN = 7.0
RASTER_DPI = 600

SAVE_SVG = True
SHOW_METADATA_STRIP = True


# ============================================================
# UNIT CONVERSIONS
# ============================================================

IN_TO_CM = 2.54
CFS_TO_CMS = 0.028316846592


# ============================================================
# CITY-SPECIFIC COLOR PALETTE
# ============================================================

# Dark shade = Stage IV
# Light shade = point gage

CITY_COLORS = {
    "Austin": {
        "stage4": "#91B64C",
        "gage": "#D6E4BD",
    },
    "Atlanta": {
        "stage4": "#7B3FA2",
        "gage": "#BDA7DC",
    },
    "Philadelphia": {
        "stage4": "#3F7FC4",
        "gage": "#B5D0EA",
    },
}

DEFAULT_COLORS = {
    "stage4": "#666666",
    "gage": "#D0D0D0",
}


# ============================================================
# MATPLOTLIB STYLE
# ============================================================

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial",
            "Helvetica",
            "DejaVu Sans",
        ],
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def _find_col(
    df: pd.DataFrame,
    candidates: list[str],
) -> str:
    """Return the first matching column name."""
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        "None of the expected columns were found.\n"
        f"Expected one of: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


def _to_hourly_increment(
    cumulative: pd.Series,
) -> pd.Series:
    """
    Convert cumulative precipitation to nonnegative increments.

    Negative differences are clipped to zero so that cumulative resets
    are not interpreted as negative precipitation.
    """
    cumulative = pd.to_numeric(
        cumulative,
        errors="coerce",
    ).fillna(0.0)

    increments = cumulative.diff()

    if len(cumulative) > 0:
        increments.iloc[0] = cumulative.iloc[0]

    return increments.clip(lower=0.0)


def position_label(position: str) -> str:
    """Convert US/DS abbreviations to full labels."""
    labels = {
        "US": "Upstream",
        "DS": "Downstream",
    }

    return labels.get(position, position)


def safe_float(
    value,
    default: float = np.nan,
) -> float:
    """Safely convert a value to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_flow(value_cms: float) -> str:
    """Format discharge in cubic meters per second."""
    if pd.isna(value_cms):
        return "NA"

    return f"{value_cms:.2f}"


def format_precip_cm(value_cm: float) -> str:
    """Format precipitation depth in centimeters."""
    if pd.isna(value_cm):
        return "NA"

    return f"{value_cm:.2f}"


def format_duration(value_h: float) -> str:
    """Format duration in hours."""
    if pd.isna(value_h):
        return "NA"

    return f"{value_h:.1f}"


# ============================================================
# FILE IDENTIFICATION
# ============================================================

def infer_city_and_pos_from_filename(
    path: Path,
) -> tuple[str, str]:
    """Infer city and watershed position from a filename."""
    stem = path.stem.lower()

    if any(token in stem for token in ["austin", "waller"]):
        city = "Austin"

    elif any(
        token in stem
        for token in ["atl", "atlanta", "proctor"]
    ):
        city = "Atlanta"

    elif any(
        token in stem
        for token in [
            "phil",
            "philadelphia",
            "ttf",
            "tacony",
            "frankford",
        ]
    ):
        city = "Philadelphia"

    else:
        city = stem.split("_")[0].title()

    position = "UNK"

    upstream_patterns = [
        r"(^|[_\-])us($|[_\-])",
        r"(^|[_\-])up($|[_\-])",
    ]

    downstream_patterns = [
        r"(^|[_\-])ds($|[_\-])",
        r"(^|[_\-])down($|[_\-])",
    ]

    if (
        any(re.search(pattern, stem) for pattern in upstream_patterns)
        or "upstream" in stem
    ):
        position = "US"

    if (
        any(re.search(pattern, stem) for pattern in downstream_patterns)
        or "downstream" in stem
    ):
        position = "DS"

    return city, position


def find_matching_file(
    directory: Path,
    city: str,
    position: str,
) -> Path:
    """Locate the CSV matching a city and watershed position."""
    files = sorted(directory.glob("*.csv"))

    matches: list[Path] = []

    for file_path in files:
        inferred_city, inferred_position = (
            infer_city_and_pos_from_filename(file_path)
        )

        if (
            inferred_city == city
            and inferred_position == position
        ):
            matches.append(file_path)

    if not matches:
        raise FileNotFoundError(
            f"No matching CSV found for {city}_{position} "
            f"in {directory}"
        )

    if len(matches) > 1:
        warnings.warn(
            f"Multiple files matched {city}_{position}. "
            f"Using {matches[0].name}"
        )

    return matches[0]


# ============================================================
# DATA LOADING
# ============================================================

def load_precip_hourly_increments(
    city: str,
    position: str,
) -> pd.DataFrame:
    """
    Load Stage IV and point-gage precipitation.

    Original values remain available in inches and additional columns
    are created in centimeters.
    """
    file_path = find_matching_file(
        PRECIP_DIR,
        city,
        position,
    )

    print(f"Precipitation file: {file_path.name}")

    df = pd.read_csv(file_path)

    datetime_col = _find_col(
        df,
        [
            "DateTimeUTC",
            "DatetimeUTC",
            "datetime_utc",
            "datetime",
        ],
    )

    df[datetime_col] = pd.to_datetime(
        df[datetime_col],
        utc=True,
        errors="coerce",
    )

    df = (
        df.dropna(subset=[datetime_col])
        .sort_values(datetime_col)
        .drop_duplicates(subset=[datetime_col], keep="last")
        .set_index(datetime_col)
    )

    stage4_col = _find_col(
        df,
        [
            "Stage IV Precipitation (in)",
            "Stage IV Precip (in)",
            "Stage IV Precipitation(in)",
            "Stage IV Precip(in)",
        ],
    )

    gage_col = None

    for candidate in [
        "Point Gage Precipitation (in)",
        "Point Gage Precip (in)",
        "Point Gage Precipitation(in)",
        "Point Gage Precip(in)",
    ]:
        if candidate in df.columns:
            gage_col = candidate
            break

    output = pd.DataFrame(
        {
            "stage4_inc_in": _to_hourly_increment(
                df[stage4_col]
            )
        }
    )

    if gage_col is not None:
        output["gage_inc_in"] = _to_hourly_increment(
            df[gage_col]
        )
    else:
        output["gage_inc_in"] = np.nan

    output = output.resample("1h").sum(min_count=1)

    output["stage4_inc_in"] = (
        output["stage4_inc_in"].fillna(0.0)
    )

    if gage_col is not None:
        output["gage_inc_in"] = (
            output["gage_inc_in"].fillna(0.0)
        )

    output = output.reset_index()

    output = output.rename(
        columns={output.columns[0]: "datetime"}
    )

    output["stage4_inc_cm"] = (
        output["stage4_inc_in"] * IN_TO_CM
    )

    output["gage_inc_cm"] = (
        output["gage_inc_in"] * IN_TO_CM
    )

    return output


def load_flow(
    city: str,
    position: str,
    hourly: bool,
) -> pd.DataFrame:
    """Load streamflow and convert cfs to cubic meters per second."""
    file_path = find_matching_file(
        FLOW_DIR,
        city,
        position,
    )

    print(f"Flow file: {file_path.name}")

    df = pd.read_csv(file_path)

    datetime_col = _find_col(
        df,
        [
            "datetime_utc",
            "DateTimeUTC",
            "DatetimeUTC",
            "datetime",
        ],
    )

    flow_col = _find_col(
        df,
        [
            "flow_cfs",
            "flow",
            "discharge_cfs",
        ],
    )

    df[datetime_col] = pd.to_datetime(
        df[datetime_col],
        utc=True,
        errors="coerce",
    )

    df[flow_col] = pd.to_numeric(
        df[flow_col],
        errors="coerce",
    )

    df = (
        df.dropna(subset=[datetime_col])
        .sort_values(datetime_col)
        .drop_duplicates(subset=[datetime_col], keep="last")
        .set_index(datetime_col)
    )

    if hourly:
        flow = df[[flow_col]].resample("1h").mean()

    else:
        flow = (
            df[[flow_col]]
            .resample("15min")
            .mean()
            .interpolate(
                method="time",
                limit=4,
                limit_area="inside",
            )
        )

    flow = flow.reset_index()

    flow = flow.rename(
        columns={
            datetime_col: "datetime",
            flow_col: "flow_cfs",
        }
    )

    flow["flow_cms"] = (
        flow["flow_cfs"] * CFS_TO_CMS
    )

    return flow[
        [
            "datetime",
            "flow_cfs",
            "flow_cms",
        ]
    ]


# ============================================================
# HYDROLOGIC CALCULATIONS
# ============================================================

def flow_at_time(
    flow_df: pd.DataFrame,
    timestamp: pd.Timestamp,
    value_col: str = "flow_cms",
) -> float:
    """Return time-interpolated flow at a specified timestamp."""
    if pd.isna(timestamp):
        return np.nan

    series = (
        flow_df
        .dropna(subset=["datetime", value_col])
        .set_index("datetime")[value_col]
        .sort_index()
    )

    if series.empty:
        return np.nan

    timestamp = pd.to_datetime(
        timestamp,
        utc=True,
    )

    if (
        timestamp < series.index.min()
        or timestamp > series.index.max()
    ):
        return np.nan

    expanded_index = series.index.union(
        pd.DatetimeIndex([timestamp])
    )

    interpolated = (
        series.reindex(expanded_index)
        .sort_index()
        .interpolate(method="time")
    )

    value = interpolated.loc[timestamp]

    return float(value) if pd.notna(value) else np.nan


def stage4_any_rain_in_window(
    precip_df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> bool:
    """Check for measurable Stage IV precipitation within a window."""
    series = (
        precip_df
        .set_index("datetime")["stage4_inc_in"]
        .sort_index()
    )

    window = series.loc[
        (series.index > start)
        & (series.index <= end)
    ]

    return bool(
        (window > RAIN_THRESHOLD_IN).any()
    )


# ============================================================
# FIGURE CREATION
# ============================================================

def create_event_figure(
    *,
    city: str,
    position: str,
    sequence: int,
    definition_tag: str,
    precip_start: pd.Timestamp,
    precip_end: pd.Timestamp,
    peak_time: pd.Timestamp,
    precip_window: pd.DataFrame,
    flow_window: pd.DataFrame,
    stage4_depth_cm: float,
    stage4_duration_h: float,
    stage4_intensity_cm_per_h: float,
    pre_dry_required_h: int,
    pre_dry_actual_h: float,
    flow_start_cms: float,
    flow_end_cms: float,
    peak_flow_cms: float,
    flow_after_cms: dict[int, float],
    after_flags: dict[int, bool],
    gage_cumulative_cm: float,
    png_path: Path,
    pdf_path: Path,
    svg_path: Path | None,
) -> None:
    """
    Create a hydrograph–hyetograph with:

    - an unobstructed main plotting panel;
    - a dedicated legend panel at upper right;
    - a separate metadata strip at the bottom.
    """

    palette = CITY_COLORS.get(
        city,
        DEFAULT_COLORS,
    )

    stage4_color = palette["stage4"]
    gage_color = palette["gage"]

    # --------------------------------------------------------
    # FIGURE LAYOUT
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        facecolor="white",
    )

    if SHOW_METADATA_STRIP:
        grid = fig.add_gridspec(
            nrows=2,
            ncols=1,
            height_ratios=[4.8, 1.65],
            hspace=0.42,
            left=0.09,
            right=0.97,
            top=0.88,
            bottom=0.07,
        )
    else:
        grid = fig.add_gridspec(
            nrows=1,
            ncols=1,
            left=0.09,
            right=0.97,
            top=0.88,
            bottom=0.14,
        )

    ax_flow = fig.add_subplot(grid[0, 0])

    if SHOW_METADATA_STRIP:
        ax_metadata = fig.add_subplot(grid[1, 0])
    else:
        ax_metadata = None

    # --------------------------------------------------------
    # HYDROGRAPH
    # --------------------------------------------------------

    ax_flow.plot(
        flow_window["datetime"],
        flow_window["flow_cms"],
        color="black",
        linewidth=2.0,
        label="Observed discharge",
        zorder=5,
    )

    ax_flow.set_ylabel(
        r"Discharge (m$^3$ s$^{-1}$)"
    )

    ax_flow.grid(
        axis="y",
        which="major",
        color="0.86",
        linewidth=0.65,
        zorder=0,
    )

    ax_flow.spines["top"].set_visible(False)

    # --------------------------------------------------------
    # HYETOGRAPHS
    # --------------------------------------------------------

    ax_precip = ax_flow.twinx()

    # Stage IV hourly values are end-of-hour labeled.
    bar_times = (
        precip_window["datetime"]
        - pd.Timedelta(hours=1)
    )

    bar_width_days = 0.82 / 24.0

    ax_precip.bar(
        bar_times,
        precip_window["stage4_inc_cm"],
        width=bar_width_days,
        align="edge",
        color=stage4_color,
        edgecolor="0.20",
        linewidth=0.4,
        label="Stage IV precipitation",
        zorder=3,
    )

    gage_available = (
        "gage_inc_cm" in precip_window.columns
        and precip_window["gage_inc_cm"].notna().any()
    )

    if gage_available:
        ax_precip.bar(
            bar_times,
            precip_window["gage_inc_cm"],
            width=bar_width_days,
            align="edge",
            facecolor=gage_color,
            edgecolor="0.20",
            linewidth=0.4,
            hatch="///",
            label="Point-gage precipitation",
            zorder=2,
        )

    ax_precip.set_ylabel("Precipitation (cm)")

    precipitation_maxima = [
        precip_window["stage4_inc_cm"].max()
    ]

    if gage_available:
        precipitation_maxima.append(
            precip_window["gage_inc_cm"].max()
        )

    precip_max = np.nanmax(precipitation_maxima)

    if not np.isfinite(precip_max) or precip_max <= 0:
        precip_max = 0.1

    ax_precip.set_ylim(
        precip_max * 1.35,
        0.0,
    )

    ax_precip.spines["top"].set_visible(False)

    # --------------------------------------------------------
    # EVENT BOUNDARIES
    # --------------------------------------------------------

    boundary_style = {
        "color": "0.25",
        "linestyle": ":",
        "linewidth": 1.1,
        "zorder": 6,
    }

    ax_flow.axvline(
        precip_start,
        **boundary_style,
    )

    ax_flow.axvline(
        precip_end,
        **boundary_style,
    )

    # --------------------------------------------------------
    # DATE AXIS
    # --------------------------------------------------------

    ax_flow.xaxis.set_major_locator(
        mdates.AutoDateLocator(
            minticks=4,
            maxticks=7,
        )
    )

    ax_flow.xaxis.set_major_formatter(
        mdates.DateFormatter(
            "%b %d\n%H:%M",
            tz=mdates.UTC,
        )
    )

    ax_flow.set_xlabel(
        "Date and time (UTC)",
        labelpad=14,
    )

    for tick_label in ax_flow.get_xticklabels():
        tick_label.set_rotation(30)
        tick_label.set_horizontalalignment("right")

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    if pd.notna(peak_time):
        peak_title = pd.to_datetime(
            peak_time
        ).strftime("%d %b %Y %H:%M UTC")
    else:
        peak_title = "Peak time unavailable"

    ax_flow.set_title(
        f"{city} ({position_label(position)}) — Event {sequence}\n"
        f"Peak discharge: {peak_title}",
        loc="left",
        pad=10,
        fontweight="semibold",
    )

    # --------------------------------------------------------
    # DEDICATED LEGEND PANEL
    # --------------------------------------------------------
    legend_handles = [
        Line2D([0],[0],color="black",linewidth=2.0,label="Observed discharge"),
        Patch(facecolor=stage4_color,edgecolor="0.20",linewidth=0.4,label="Stage IV precipitation"),
    ]
    if gage_available:
        legend_handles.append(Patch(facecolor=gage_color,edgecolor="0.20",linewidth=0.4,hatch="///",label="Point-gage precipitation"))
    legend_handles.append(Line2D([0],[0],color="0.25",linestyle=":",linewidth=1.1,label="Stage IV event boundaries"))
    ax_flow.legend(handles=legend_handles,loc="upper right",bbox_to_anchor=(0.985,0.985),
                   frameon=True,framealpha=0.97,facecolor="white",edgecolor="0.72",
                   borderpad=0.6,labelspacing=0.6,handlelength=2.4)

    # --------------------------------------------------------
    # METADATA STRIP
    # --------------------------------------------------------

    if ax_metadata is not None:
        ax_metadata.set_xlim(0, 1)
        ax_metadata.set_ylim(0, 1)
        ax_metadata.axis("off")
        if pd.notna(peak_time):
            peak_time_text = pd.to_datetime(
                peak_time
            ).strftime("%d %b %Y %H:%M UTC")
        else:
            peak_time_text = "NA"

        left_column = [
            f"Event definition: {definition_tag}",
            (
                f"Stage IV precipitation: "
                f"{format_precip_cm(stage4_depth_cm)} cm"
            ),
            (
                f"Event duration: "
                f"{format_duration(stage4_duration_h)} h"
            ),
            (
                f"Mean intensity: "
                f"{stage4_intensity_cm_per_h:.3f} "
                r"cm h$^{-1}$"
            ),
        ]

        middle_column = [
            (
                "Dry period (required/actual):\n"
                f"  {pre_dry_required_h} / "
                f"{pre_dry_actual_h:.1f} h"
            ),
            (
                "Discharge at precipitation start/end:\n"
                f"  {format_flow(flow_start_cms)} / "
                f"{format_flow(flow_end_cms)} "
                r"m$^3$ s$^{-1}$"
            ),
            (
                f"Peak discharge: "
                f"{format_flow(peak_flow_cms)} "
                r"m$^3$ s$^{-1}$"
            ),
            f"Peak time: {peak_time_text}",
        ]

        right_column = [
            (
                "Discharge +1/+3/+6 h:\n"
                f"  {format_flow(flow_after_cms.get(1, np.nan))} / "
                f"{format_flow(flow_after_cms.get(3, np.nan))} / "
                f"{format_flow(flow_after_cms.get(6, np.nan))} "
                r"m$^3$ s$^{-1}$"
            ),
            (
                "Precipitation after event (6/12 h):\n"
                f"  {'Yes' if after_flags.get(6, False) else 'No'} / "
                f"{'Yes' if after_flags.get(12, False) else 'No'}"
            ),
            (
                f"Point-gage total (±{GAGE_PAD_H} h): "
                f"{format_precip_cm(gage_cumulative_cm)} cm"
            ),
        ]

        metadata_style = {
            "ha": "left",
            "va": "top",
            "fontsize": 9.1,
            "linespacing": 1.28,
            "transform": ax_metadata.transAxes,
        }

        ax_metadata.text(
            0.01,
            0.96,
            "\n".join(left_column),
            **metadata_style,
        )

        ax_metadata.text(
            0.34,
            0.96,
            "\n".join(middle_column),
            **metadata_style,
        )

        ax_metadata.text(
            0.70,
            0.96,
            "\n".join(right_column),
            **metadata_style,
        )

    # --------------------------------------------------------
    # SAVE FIGURES
    # --------------------------------------------------------

    fig.savefig(
        png_path,
        dpi=RASTER_DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
    )

    if svg_path is not None:
        fig.savefig(
            svg_path,
            bbox_inches="tight",
            facecolor="white",
        )

    plt.close(fig)


# ============================================================
# MAIN WORKFLOW
# ============================================================

def main() -> None:
    """Generate figures for all selected Stage IV-defined events."""
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    events = pd.read_csv(EVENTS_CSV)

    datetime_columns = [
        "event_start_hrlabel_utc",
        "event_end_hrlabel_utc",
        "precip_start_utc",
        "precip_end_utc",
    ]

    for column in datetime_columns:
        if column in events.columns:
            events[column] = pd.to_datetime(
                events[column],
                utc=True,
                errors="coerce",
            )

    if "def_tag" not in events.columns:
        raise KeyError(
            "The events CSV does not contain a 'def_tag' column."
        )

    events = events.loc[
        events["def_tag"] == DEF_TAG_TO_PLOT
    ].copy()

    if events.empty:
        all_events = pd.read_csv(EVENTS_CSV)

        available_definitions = (
            all_events["def_tag"]
            .dropna()
            .astype(str)
            .unique()
        )

        raise SystemExit(
            f"No events were found for "
            f"def_tag='{DEF_TAG_TO_PLOT}'.\n"
            f"Available definitions:\n"
            f"{sorted(available_definitions)}"
        )

    required_columns = [
        "site_key",
        "city",
        "position",
        "precip_start_utc",
        "precip_end_utc",
        "event_seq_within_site_def",
        "stage4_depth_in",
        "stage4_duration_h",
        "stage4_intensity_in_per_h",
        "pre_dry_hours_req",
        "pre_dry_hours_actual",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in events.columns
    ]

    if missing_columns:
        raise KeyError(
            "The following required columns are missing "
            f"from the events CSV: {missing_columns}"
        )

    all_event_rows: list[dict] = []

    for site_key, site_events in events.groupby(
        "site_key",
        sort=True,
    ):
        city = str(site_events["city"].iloc[0])
        position = str(site_events["position"].iloc[0])

        print(
            "\n"
            + "=" * 70
            + f"\nLoading data for {site_key}"
            + "\n"
            + "=" * 70
        )

        precip_hourly = load_precip_hourly_increments(
            city,
            position,
        )

        flow_df = load_flow(
            city,
            position,
            hourly=RESAMPLE_FLOW_TO_HOURLY,
        )

        assert "datetime" in flow_df.columns
        assert "flow_cms" in flow_df.columns
        assert flow_df["datetime"].notna().all()
        assert flow_df["flow_cms"].notna().any()

        print(
            f"{site_key}: flow rows = {len(flow_df):,}\n"
            f"Time span = "
            f"{flow_df['datetime'].min()} to "
            f"{flow_df['datetime'].max()}"
        )

        site_output_dir = (
            OUT_DIR
            / DEF_TAG_TO_PLOT
            / site_key
        )

        site_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        site_events = site_events.sort_values(
            "precip_start_utc"
        )

        for _, event in site_events.iterrows():
            sequence = int(
                event["event_seq_within_site_def"]
            )

            try:
                precip_start = pd.to_datetime(
                    event["precip_start_utc"],
                    utc=True,
                )

                precip_end = pd.to_datetime(
                    event["precip_end_utc"],
                    utc=True,
                )

                plot_start = (
                    precip_start
                    - pd.Timedelta(hours=PAD_BEFORE_H)
                )

                plot_end = (
                    precip_end
                    + pd.Timedelta(hours=PAD_AFTER_H)
                )

                # --------------------------------------------
                # PRECIPITATION WINDOW
                # --------------------------------------------

                precip_window = precip_hourly.loc[
                    (
                        precip_hourly["datetime"]
                        >= plot_start
                    )
                    & (
                        precip_hourly["datetime"]
                        <= plot_end
                    )
                ].copy()

                precip_window = (
                    precip_window
                    .set_index("datetime")
                    .resample("1h")
                    .sum(min_count=1)
                    .reset_index()
                )

                precip_window["stage4_inc_in"] = (
                    precip_window["stage4_inc_in"]
                    .fillna(0.0)
                )

                precip_window["stage4_inc_cm"] = (
                    precip_window["stage4_inc_cm"]
                    .fillna(0.0)
                )

                # --------------------------------------------
                # FLOW WINDOW
                # --------------------------------------------

                flow_window = flow_df.loc[
                    (
                        flow_df["datetime"]
                        >= plot_start
                    )
                    & (
                        flow_df["datetime"]
                        <= plot_end
                    )
                ].copy()

                # --------------------------------------------
                # EVENT PRECIPITATION METRICS
                # --------------------------------------------

                stage4_depth_in = safe_float(
                    event["stage4_depth_in"]
                )

                stage4_duration_h = safe_float(
                    event["stage4_duration_h"]
                )

                stage4_intensity_in_per_h = safe_float(
                    event["stage4_intensity_in_per_h"]
                )

                stage4_depth_cm = (
                    stage4_depth_in * IN_TO_CM
                )

                stage4_intensity_cm_per_h = (
                    stage4_intensity_in_per_h
                    * IN_TO_CM
                )

                pre_dry_required_h = int(
                    event["pre_dry_hours_req"]
                )

                pre_dry_actual_h = safe_float(
                    event["pre_dry_hours_actual"]
                )

                min_depth_in = safe_float(
                    event.get("min_depth_in", np.nan)
                )

                iet_hours = safe_float(
                    event.get("iet_hours", np.nan)
                )

                # --------------------------------------------
                # FLOW METRICS
                # --------------------------------------------

                flow_start_cms = flow_at_time(
                    flow_df,
                    precip_start,
                    value_col="flow_cms",
                )

                flow_end_cms = flow_at_time(
                    flow_df,
                    precip_end,
                    value_col="flow_cms",
                )

                if (
                    flow_window.empty
                    or flow_window["flow_cms"].dropna().empty
                ):
                    peak_flow_cms = np.nan
                    peak_time = pd.NaT

                else:
                    peak_index = (
                        flow_window["flow_cms"].idxmax()
                    )

                    peak_flow_cms = float(
                        flow_window.loc[
                            peak_index,
                            "flow_cms",
                        ]
                    )

                    peak_time = flow_window.loc[
                        peak_index,
                        "datetime",
                    ]

                flow_after_cms: dict[int, float] = {}

                for hours_after in FLOW_AFTER_H:
                    flow_after_cms[hours_after] = (
                        flow_at_time(
                            flow_df,
                            precip_end
                            + pd.Timedelta(
                                hours=hours_after
                            ),
                            value_col="flow_cms",
                        )
                    )

                # --------------------------------------------
                # POST-EVENT PRECIPITATION CHECKS
                # --------------------------------------------

                after_flags: dict[int, bool] = {}

                for hours_after in CHECK_AFTER_H:
                    after_flags[hours_after] = (
                        stage4_any_rain_in_window(
                            precip_hourly,
                            precip_end,
                            precip_end
                            + pd.Timedelta(
                                hours=hours_after
                            ),
                        )
                    )

                # --------------------------------------------
                # POINT-GAGE TOTAL
                # --------------------------------------------

                gage_window_start = (
                    precip_start
                    - pd.Timedelta(hours=GAGE_PAD_H)
                )

                gage_window_end = (
                    precip_end
                    + pd.Timedelta(hours=GAGE_PAD_H)
                )

                gage_window = precip_hourly.loc[
                    (
                        precip_hourly["datetime"]
                        > gage_window_start
                    )
                    & (
                        precip_hourly["datetime"]
                        <= gage_window_end
                    )
                ].copy()

                if (
                    "gage_inc_cm" in gage_window.columns
                    and gage_window["gage_inc_cm"]
                    .notna()
                    .any()
                ):
                    gage_cumulative_cm = float(
                        gage_window["gage_inc_cm"].sum()
                    )
                else:
                    gage_cumulative_cm = np.nan

                # --------------------------------------------
                # EVENT ID AND OUTPUT PATHS
                # --------------------------------------------

                peak_datetime_for_id = (
                    pd.to_datetime(peak_time)
                    if pd.notna(peak_time)
                    else precip_end
                )

                peak_stamp_readable = (
                    peak_datetime_for_id.strftime(
                        "%Y-%m-%d_%Hh"
                    )
                )

                event_id = (
                    f"{site_key}_"
                    f"{DEF_TAG_TO_PLOT}_"
                    f"E{sequence:03d}_"
                    f"Peak{peak_stamp_readable}"
                )

                png_path = (
                    site_output_dir
                    / f"{event_id}.png"
                )

                pdf_path = (
                    site_output_dir
                    / f"{event_id}.pdf"
                )

                svg_path = (
                    site_output_dir
                    / f"{event_id}.svg"
                    if SAVE_SVG
                    else None
                )

                # --------------------------------------------
                # CREATE FIGURE
                # --------------------------------------------

                create_event_figure(
                    city=city,
                    position=position,
                    sequence=sequence,
                    definition_tag=DEF_TAG_TO_PLOT,
                    precip_start=precip_start,
                    precip_end=precip_end,
                    peak_time=peak_time,
                    precip_window=precip_window,
                    flow_window=flow_window,
                    stage4_depth_cm=stage4_depth_cm,
                    stage4_duration_h=stage4_duration_h,
                    stage4_intensity_cm_per_h=(
                        stage4_intensity_cm_per_h
                    ),
                    pre_dry_required_h=(
                        pre_dry_required_h
                    ),
                    pre_dry_actual_h=(
                        pre_dry_actual_h
                    ),
                    flow_start_cms=flow_start_cms,
                    flow_end_cms=flow_end_cms,
                    peak_flow_cms=peak_flow_cms,
                    flow_after_cms=flow_after_cms,
                    after_flags=after_flags,
                    gage_cumulative_cm=(
                        gage_cumulative_cm
                    ),
                    png_path=png_path,
                    pdf_path=pdf_path,
                    svg_path=svg_path,
                )

                # --------------------------------------------
                # OUTPUT TABLE ROW
                # --------------------------------------------

                row = {
                    "event_id": event_id,
                    "def_tag": DEF_TAG_TO_PLOT,
                    "site_key": site_key,
                    "city": city,
                    "position": position,
                    "event_seq_within_site_def": sequence,
                    "peak_stamp_readable": peak_stamp_readable,

                    "precip_start_utc": precip_start,
                    "precip_end_utc": precip_end,
                    "plot_start_utc": plot_start,
                    "plot_end_utc": plot_end,

                    "min_depth_in_filter": min_depth_in,
                    "min_depth_cm_filter": (
                        min_depth_in * IN_TO_CM
                        if pd.notna(min_depth_in)
                        else np.nan
                    ),
                    "iet_hours": iet_hours,
                    "pre_dry_hours_req": pre_dry_required_h,
                    "pre_dry_hours_actual": pre_dry_actual_h,

                    "stage4_depth_in": stage4_depth_in,
                    "stage4_depth_cm": stage4_depth_cm,
                    "stage4_duration_h": stage4_duration_h,
                    "stage4_intensity_in_per_h": (
                        stage4_intensity_in_per_h
                    ),
                    "stage4_intensity_cm_per_h": (
                        stage4_intensity_cm_per_h
                    ),

                    f"gage_cum_pm{GAGE_PAD_H}h_cm": (
                        gage_cumulative_cm
                    ),

                    "stage4_rain_within_6h_after_end": (
                        bool(after_flags.get(6, False))
                    ),
                    "stage4_rain_within_12h_after_end": (
                        bool(after_flags.get(12, False))
                    ),

                    "flow_at_precip_start_cms": (
                        flow_start_cms
                    ),
                    "peak_flow_cms": peak_flow_cms,
                    "peak_flow_time_utc": peak_time,
                    "flow_at_precip_end_cms": (
                        flow_end_cms
                    ),
                    "flow_end_plus_1h_cms": (
                        flow_after_cms.get(1, np.nan)
                    ),
                    "flow_end_plus_3h_cms": (
                        flow_after_cms.get(3, np.nan)
                    ),
                    "flow_end_plus_6h_cms": (
                        flow_after_cms.get(6, np.nan)
                    ),

                    "figure_png": str(png_path),
                    "figure_pdf": str(pdf_path),
                    "figure_svg": (
                        str(svg_path)
                        if svg_path is not None
                        else ""
                    ),

                    "error": "",
                }

                all_event_rows.append(row)

                print(f"Saved: {event_id}")

            except Exception as error:
                error_message = repr(error)

                print(
                    f"[SKIP] {site_key}, event {sequence}: "
                    f"{error_message}"
                )

                all_event_rows.append(
                    {
                        "event_id": (
                            f"{site_key}_"
                            f"{DEF_TAG_TO_PLOT}_"
                            f"E{sequence:03d}_FAILED"
                        ),
                        "def_tag": DEF_TAG_TO_PLOT,
                        "site_key": site_key,
                        "city": city,
                        "position": position,
                        "event_seq_within_site_def": sequence,
                        "error": error_message,
                    }
                )

    # ========================================================
    # WRITE OUTPUT TABLE
    # ========================================================

    output_csv_directory = (
        OUT_DIR / DEF_TAG_TO_PLOT
    )

    output_csv_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_csv = (
        output_csv_directory
        / f"event_metrics_{DEF_TAG_TO_PLOT}_metadata_strip.csv"
    )

    output_table = pd.DataFrame(all_event_rows)

    output_table.to_csv(
        output_csv,
        index=False,
    )

    if "error" in output_table.columns:
        successful_events = int(
            output_table["error"]
            .fillna("")
            .eq("")
            .sum()
        )
    else:
        successful_events = len(output_table)

    failed_events = (
        len(output_table) - successful_events
    )

    print(
        "\n"
        + "=" * 70
        + "\nProcessing complete"
        + "\n"
        + "=" * 70
        + f"\nSuccessful figures: {successful_events}"
        + f"\nFailed events: {failed_events}"
        + f"\nMetrics table: {output_csv}"
        + f"\nFigure directory: {output_csv_directory}"
    )


if __name__ == "__main__":
    main()