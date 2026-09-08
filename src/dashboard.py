"""
dashboard.py - the public-facing RiverWatch dashboard.

Shows predicted ecological health for rivers across England, written as part of
the COMP7039 MSc dissertation at Oxford Brookes. Everything here is aimed at
people who aren't water-quality experts - plain English, no machine-learning
jargon.

To run it:  streamlit run src/dashboard.py
"""

import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import plotly.express as px
from pyproj import Transformer
import json
import os
from config import RESULTS_DIR
from db_loader import get_conn


st.set_page_config(
    page_title="RiverWatch | England Freshwater Risk",
    page_icon="🌊",
    layout="wide",
)

st.markdown("""
<style>
/* Make all Streamlit column rows stack vertically on narrow screens */
@media (max-width: 768px) {
    .stHorizontalBlock {
        flex-direction: column !important;
    }
    .stHorizontalBlock > div {
        width: 100% !important;
    }
    /* Shrink the hero banner padding and font for phones */
    .riverwatch-hero {
        padding: 16px 18px !important;
    }
    .riverwatch-hero h1 {
        font-size: 22px !important;
    }
    .riverwatch-hero p {
        font-size: 14px !important;
    }
    /* Give the map a bit less height on phones */
    .stPlotlyChart {
        max-height: 400px;
    }
    /* Metric cards: reduce padding so they don't waste space */
    .stMetric {
        padding: 4px 0 !important;
    }
    /* Make sidebar text slightly smaller to fit more */
    .stSidebar .stMarkdown {
        font-size: 14px;
    }
}
/* Hover tooltip for tour mode */
.rw-tour {
    position: relative;
    display: inline-block;
    margin-left: 6px;
    color: #4A8CB5;
    cursor: help;
    font-size: 0.8em;
    vertical-align: super;
}
.rw-tour .rw-tour-tip {
    visibility: hidden;
    width: 280px;
    background-color: #1a3a4a;
    color: white;
    text-align: left;
    border-radius: 8px;
    padding: 12px 16px;
    position: absolute;
    z-index: 9999;
    bottom: 140%;
    left: 50%;
    margin-left: -140px;
    opacity: 0;
    transition: opacity 0.25s;
    font-size: 13px;
    line-height: 1.5;
    font-weight: normal;
    box-shadow: 0 4px 16px rgba(0,0,0,0.25);
    pointer-events: none;
}
.rw-tour .rw-tour-tip::after {
    content: "";
    position: absolute;
    top: 100%;
    left: 50%;
    margin-left: -6px;
    border-width: 6px;
    border-style: solid;
    border-color: #1a3a4a transparent transparent transparent;
}
.rw-tour:hover .rw-tour-tip {
    visibility: visible;
    opacity: 1;
}
</style>
""", unsafe_allow_html=True)

COL_GOOD     = "#2E7D32"
COL_MODERATE = "#E8A33D"
COL_POOR     = "#B33A3A"
COLOURS = {"Good": COL_GOOD, "Moderate": COL_MODERATE, "Poor": COL_POOR}

FEATURE_LABELS = {
    "nitrate_mid":     "Nitrate in the water",
    "phosphate_mid":   "Phosphate in the water",
    "spills_per_pipe": "Spills per sewage pipe",
    "avg_spills":      "Typical yearly sewage spills",
    "n_overflows":     "Sewage overflow pipes nearby",
}

STATUS_MEANING = {
    "Good":     "The water here supports healthy wildlife and plant life.",
    "Moderate": "This water shows signs of pollution pressure. Wildlife is affected, but not severely.",
    "Poor":     "This water is significantly polluted. Wildlife and plant life are struggling here.",
}

LC_KIND_NAMES = {
    "woodland": "woodland",
    "arable":   "farmland",
    "grass":    "grassland",
    "wetland":  "wetland",
    "urban":    "built-up area",
    "water":    "open water",
}

GENERIC_SITE_NAMES = {"other", "n/a", "unknown", "unnamed", ""}

# Area of a circle with the given radius - used to convert land cover
# percentages into actual km² figures, a genuine number rather than a
# percentage.
BUFFER_AREA_KM2 = {
    "1km": 3.14159 * 1**2,
    "5km": 3.14159 * 5**2,
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def describe_lc_factor(feat_name, value):
    parts = feat_name.split("_")
    kind = LC_KIND_NAMES.get(parts[1], parts[1])
    radius = parts[-1]
    area_km2 = value / 100 * BUFFER_AREA_KM2.get(radius, 0)
    if value < 0.5:
        return f"No {kind} within {radius}"
    return f"{area_km2:.1f} km² of {kind} within {radius}"


def describe_level(feat_name, value, percentiles_df):
    if feat_name not in percentiles_df.index:
        return ""
    p25, p75 = percentiles_df.loc[feat_name, "p25"], percentiles_df.loc[feat_name, "p75"]
    if value <= p25:
        return "low compared to other places"
    elif value >= p75:
        return "high compared to other places"
    return "a typical level"


def render_location_chart(local_df):
    """
    A bar chart showing every factor's real SHAP value for this specific
    location - a visual companion to the summary text below, replacing
    the old per-factor text list.
    """
    chart_df = local_df.copy()
    chart_df["label"] = chart_df.apply(
        lambda r: describe_lc_factor(r["feat"], r["val"]) if r["feat"].startswith("lc_")
        else FEATURE_LABELS.get(r["feat"], r["feat"]),
        axis=1
    )
    chart_df["direction"] = chart_df["shap"].apply(lambda v: "Polluted" if v > 0 else "Healthy")

    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(labelFontSize=11, labelLimit=250)),
            x=alt.X("shap:Q", title="← Healthier          Polluted →",
                    axis=alt.Axis(labels=False, ticks=False, grid=False)),
            color=alt.Color("direction:N",
                             scale=alt.Scale(domain=["Polluted", "Healthy"], range=["#B33A3A", "#2E7D32"]),
                             legend=alt.Legend(title=None)),
            tooltip=[alt.Tooltip("label:N", title="Factor"), alt.Tooltip("shap:Q", title="Pull", format="+.3f")],
        )
        .properties(height=380)
    )
    return chart


def clean_site_name(name, county=None):
    """
    Turn raw database site names into something readable.

    Some sites have placeholder names like "other" or "n/a". When that
    happens we fall back to the county name so the user still sees
    something meaningful on the map and in the search box.
    """
    if isinstance(name, str) and name.strip().lower() in GENERIC_SITE_NAMES:
        if pd.notna(county) and county:
            return f"Unnamed site, {county}"
        return "Unnamed site"
    return name


def make_display_name(row, site_counts):
    name = row["site_name"]
    county = row.get("county")
    clean_name = clean_site_name(name, county)
    if clean_name != name:
        return clean_name
    if site_counts.get(name, 0) <= 1:
        return name
    return f"{name}, {county}" if pd.notna(county) and county else name


def get_location_history(site_name, county):
    """
    Finds all historical samples at the same physical location (matching
    site_name and county), ordered by date - lets us show a genuine
    before/after comparison for one specific place, rather than a
    potentially misleading national trend across inconsistently-sampled
    locations.
    """
    conn = get_conn()
    history = pd.read_sql("""
        SELECT fww_id, sample_date, nitrate_mid, phosphate_mid, wfd_status
        FROM feat_matrix
        WHERE site_name = ? AND county = ?
        ORDER BY sample_date
    """, conn, params=(site_name, county))
    conn.close()
    return history


def get_regional_comparison(county):
    """All real samples in the same county, for a fair local comparison."""
    conn = get_conn()
    df = pd.read_sql(
        "SELECT wfd_status FROM feat_matrix WHERE county = ?",
        conn, params=(county,)
    )
    conn.close()
    return df


def get_county_summary():
    """% Poor by county, across England, for counties with enough real data."""
    conn = get_conn()
    df = pd.read_sql("""
        SELECT county, wfd_status FROM feat_matrix
        WHERE county IS NOT NULL AND county != 'Unknown' AND county != ''
    """, conn)
    conn.close()
    grouped = df.groupby("county")
    summary = grouped["wfd_status"].apply(lambda s: (s == "Poor").mean() * 100).reset_index()
    summary.columns = ["county", "pct_poor"]
    summary["n"] = grouped.size().values
    return summary[summary["n"] >= 20].sort_values("pct_poor", ascending=False)


def get_yearly_trend():
    """% Poor by year across all samples - see the caveat shown alongside this chart."""
    conn = get_conn()
    df = pd.read_sql("SELECT sample_date, wfd_status FROM feat_matrix WHERE sample_date IS NOT NULL", conn)
    conn.close()
    df["year"] = pd.to_datetime(df["sample_date"], errors="coerce").dt.year
    df = df.dropna(subset=["year"])
    summary = df.groupby("year")["wfd_status"].apply(lambda s: (s == "Poor").mean() * 100).reset_index()
    summary.columns = ["year", "pct_poor"]
    summary["n_samples"] = df.groupby("year").size().values
    return summary


@st.cache_data(ttl=300)  # pull fresh data from the database every 5 minutes
def load_all_data():
    conn = get_conn()
    preds = pd.read_sql("""
        SELECT p.fww_id, p.site_name, p.easting, p.northing, p.wb_id,
               p.predicted_status, p.prob_moderate, p.prob_poor,
               p.data_source, p.predicted_at, f.county
        FROM predictions p
        INNER JOIN (
            SELECT fww_id,
                   MAX(CASE WHEN data_source = 'live' THEN predicted_at END) as live_time,
                   MAX(predicted_at) as any_time
            FROM predictions
            GROUP BY fww_id
        ) latest
        ON p.fww_id = latest.fww_id
        AND p.predicted_at = COALESCE(latest.live_time, latest.any_time)
        LEFT JOIN feat_matrix f ON p.fww_id = f.fww_id
        ORDER BY p.fww_id
    """, conn)
    conn.close()

    proj = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    lons, lats = proj.transform(preds["easting"].values, preds["northing"].values)
    preds["lat"], preds["lon"] = lats, lons

    try:
        shap_vals = np.load(os.path.join(RESULTS_DIR, "shap_values.npy"))
        shap_feats = pd.read_csv(os.path.join(RESULTS_DIR, "shap_input_features.csv"))
        with open(os.path.join(RESULTS_DIR, "shap_global_importance.json")) as f:
            global_imp = json.load(f)

        conn2 = get_conn()
        fm_ids = pd.read_sql("SELECT fww_id FROM feat_matrix", conn2)["fww_id"]
        conn2.close()
        shap_id_to_pos = {str(fww_id): i for i, fww_id in enumerate(fm_ids)}

    except FileNotFoundError:
        shap_vals, shap_feats, global_imp, shap_id_to_pos = None, None, {}, {}

    return preds, shap_vals, shap_feats, global_imp, shap_id_to_pos


@st.cache_data(ttl=3600)  # percentiles barely move, so checking once an hour is plenty
def load_feature_percentiles():
    conn = get_conn()
    fm = pd.read_sql("SELECT * FROM feat_matrix", conn)
    conn.close()

    feat_cols = [c for c in fm.columns if c not in
                 ("id", "fww_id", "site_name", "sample_date", "easting", "northing",
                  "wb_id", "county", "rbd", "wfd_dist_m", "match_q", "wfd_status", "loaded_at")]

    return pd.DataFrame({
        "p25": fm[feat_cols].quantile(0.25),
        "p75": fm[feat_cols].quantile(0.75),
    })


# ============================================================================
# MAIN PROGRAM FLOW
# ============================================================================

preds, shap_vals, shap_feats, global_imp, shap_id_to_pos = load_all_data()
percentiles = load_feature_percentiles()


# ---- Sidebar ----------------------------------------------------------------

st.sidebar.markdown("# 🌊 RiverWatch")
st.sidebar.caption("England Freshwater Risk Dashboard")
st.sidebar.markdown("---")

st.sidebar.markdown("### Info tour")
tour_enabled = st.sidebar.toggle(
    "Show info tips", value=False,
    help="Turn on to see helpful tips when you hover over the info icons next to each section"
)
if tour_enabled:
    st.sidebar.caption("Info tips are on. Hover over the \u2139\ufe0f icons to learn about each section.")
st.sidebar.markdown("---")

st.sidebar.markdown("### Show me")

status_filter = st.sidebar.multiselect(
    "Water health rating",
    options=["Good", "Moderate", "Poor"],
    default=["Good", "Moderate", "Poor"],
    label_visibility="collapsed",
)
st.sidebar.caption("Tick or untick to show different ratings on the map.")

st.sidebar.markdown("---")
st.sidebar.markdown("### Find a place")

site_counts = preds["site_name"].value_counts()
preds["display_name"] = preds.apply(lambda row: make_display_name(row, site_counts), axis=1)

site_lookup = preds[["fww_id", "display_name"]].drop_duplicates(subset="fww_id")
site_lookup = site_lookup.rename(columns={"display_name": "display"})

display_to_fww_id = dict(zip(site_lookup["display"], site_lookup["fww_id"]))
all_site_names = sorted(site_lookup["display"].tolist())

if st.session_state.pop("_clear_search_flag", False):
    st.session_state["site_search"] = []

search_pick = st.sidebar.multiselect(
    "Search by name",
    options=all_site_names,
    max_selections=1,
    label_visibility="collapsed",
    key="site_search",
    placeholder="Type to search...",
)
searched_fww_id = display_to_fww_id.get(search_pick[0]) if search_pick else None
st.sidebar.caption("Start typing a river or site name - matching places appear as you type.")

st.sidebar.markdown("### What am I looking at?")
st.sidebar.markdown(
    "Each dot is a place where volunteers have tested river water quality. "
    "The colour shows how healthy we predict that stretch of water is, based on "
    "what's in the water, nearby sewage discharges, and the surrounding landscape."
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Built for the COMP7039 MSc dissertation, Oxford Brookes University. "
    "Not an official Environment Agency tool."
)


# ---- Hero banner --------------------------------------------------------------

tour_hero = (
    "No English river reached Good ecological status in the latest 2022 assessment. "
    "This tool predicts health for places between official assessments, and explains "
    "why each place gets its rating."
)
tour_hero_icon = (
    " <span class='rw-tour'>ℹ️<span class='rw-tour-tip'>"
    f"{tour_hero}</span></span>"
) if tour_enabled else ""

st.markdown(
    "<div class='riverwatch-hero' style='background: linear-gradient(135deg, #1a3a4a 0%, #2E7D32 100%); "
    "padding: 28px 32px; border-radius: 10px; margin-bottom: 8px;'>"
    "<h1 style='color: white; font-size: 28px; margin: 0 0 8px 0;'>"
    "🌊 How healthy are England's rivers?"
    "</h1>"
    "<p style='color: rgba(255,255,255,0.9); font-size: 16px; margin: 0; line-height: 1.5;'>"
    "Welcome to <strong>RiverWatch</strong> - an AI-powered tool that predicts the health "
    "of rivers and streams across England. We combine water test results from "
    "volunteers, records of sewage discharges, and information about the surrounding "
    "land to estimate how each stretch of water is doing - and importantly, "
    "<strong>why</strong>."
    "</p>"
    "</div>",
    unsafe_allow_html=True,
)


# ---- Summary metrics ------------------------------------------------------------

filtered = preds[preds["predicted_status"].isin(status_filter)]

n_good = (preds["predicted_status"] == "Good").sum()
n_mod  = (preds["predicted_status"] == "Moderate").sum()
n_poor = (preds["predicted_status"] == "Poor").sum()

m1, m2, m3, m4 = st.columns(4, gap="small")
m1.metric("Places tested", f"{len(preds):,}")
m2.metric("Healthy (Good)", f"{n_good:,}")
m3.metric("Under pressure (Moderate)", f"{n_mod:,}")
m4.metric("Polluted (Poor)", f"{n_poor:,}")

if n_good == 0:
    st.info(
        "**None of England's rivers currently reach 'Good' health.** "
        "This isn't a gap in our data - it reflects the official Environment Agency "
        "assessment, where no English river waterbody achieved Good ecological status "
        "in the most recent 2022 assessment."
    )

st.markdown("---")


# ---- Map and detail panel -------------------------------------------------------

map_col, detail_col = st.columns([3, 2], gap="small")

if searched_fww_id is not None and searched_fww_id != st.session_state.get("_last_shown_search_id"):
    st.session_state["_active_selection"] = ("search", searched_fww_id)
    st.session_state["_last_shown_search_id"] = searched_fww_id
    st.session_state["_force_recenter"] = True
    st.session_state["_recenter_source"] = "search"

selection = st.session_state.get("_active_selection")
selected_fww_id = None
if selection:
    if selection[0] in ("search", "click_id"):
        selected_fww_id = selection[1]

tour_map = (
    "Each dot is a volunteer-tested site. The colour shows predicted health - "
    "green is Good, amber is Moderate, red is Poor. Click a dot to see the "
    "explanation panel on the right."
)
tour_map_icon = (
    " <span class='rw-tour'>ℹ️<span class='rw-tour-tip'>"
    f"{tour_map}</span></span>"
) if tour_enabled else ""

with map_col:
    st.markdown(f"<h3>Explore the map{tour_map_icon}</h3>", unsafe_allow_html=True)
    st.caption("Click any dot to find out why that stretch of water got its rating.")

    if filtered.empty:
        st.warning(
            f"There are no locations with a **{', '.join(status_filter)}** rating. "
            "Try ticking a different rating in the sidebar."
        )
    else:
        shown = filtered.reset_index(drop=True)

        DEFAULT_CENTER = {"lat": 52.8, "lon": -1.6}
        DEFAULT_ZOOM = 5

        if "_map_center" not in st.session_state:
            st.session_state["_map_center"] = DEFAULT_CENTER
        if "_map_zoom" not in st.session_state:
            st.session_state["_map_zoom"] = DEFAULT_ZOOM
        if "_map_uirevision" not in st.session_state:
            st.session_state["_map_uirevision"] = "freshwater-risk-map"
        if "_map_at_home" not in st.session_state:
            st.session_state["_map_at_home"] = True

        force_recenter = st.session_state.pop("_force_recenter", False)
        recenter_source = st.session_state.pop("_recenter_source", None)

        searched_row = None
        if selected_fww_id is not None:
            match_for_map = preds[preds["fww_id"] == selected_fww_id]
            if not match_for_map.empty:
                searched_row = match_for_map.iloc[0]

        should_recenter = False
        if force_recenter and searched_row is not None:
            if recenter_source == "search":
                should_recenter = True
            elif recenter_source == "click" and st.session_state["_map_at_home"]:
                should_recenter = True

        if should_recenter:
            st.session_state["_map_center"] = {
                "lat": float(searched_row["lat"]),
                "lon": float(searched_row["lon"]),
            }
            st.session_state["_map_zoom"] = 11
            st.session_state["_map_at_home"] = False
            st.session_state["_map_uirevision"] = "freshwater-risk-map-recenter"

        map_center = st.session_state["_map_center"]
        map_zoom = st.session_state["_map_zoom"]
        uirevision = st.session_state["_map_uirevision"]

        fig = px.scatter_map(
            shown,
            lat="lat", lon="lon",
            color="predicted_status",
            color_discrete_map=COLOURS,
            hover_name="display_name",
            hover_data={"predicted_status": True, "lat": ":.4f", "lon": ":.4f"},
            zoom=map_zoom, center=map_center,
            height=520,
            map_style="carto-voyager",
            category_orders={"predicted_status": ["Good", "Moderate", "Poor"]},
            labels={"predicted_status": "Water health"},
        )
        fig.update_traces(marker={"size": 7, "opacity": 0.8})

        if searched_row is not None:
            halo_lat, halo_lon = [searched_row["lat"]], [searched_row["lon"]]
            dot_colour = COLOURS.get(searched_row["predicted_status"], "#888")
            dot_hover = searched_row["display_name"]
        else:
            halo_lat, halo_lon = [], []
            dot_colour = "#888"
            dot_hover = ""

        fig.add_scattermap(
            lat=halo_lat, lon=halo_lon,
            mode="markers",
            marker={"size": 16, "color": "#222222", "opacity": 0.9},
            showlegend=False,
            hoverinfo="skip",
        )
        fig.add_scattermap(
            lat=halo_lat, lon=halo_lon,
            mode="markers",
            marker={"size": 9, "color": dot_colour},
            showlegend=False,
            hovertext=dot_hover,
            hoverinfo="text",
        )

        fig.update_layout(
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            uirevision=uirevision,
            map={"uirevision": uirevision},
            legend={
                "yanchor": "bottom", "y": 0.03,
                "xanchor": "left",   "x": 0.02,
                "bgcolor": "rgba(255,255,255,0.92)",
                "bordercolor": "rgba(0,0,0,0.15)",
                "borderwidth": 1,
                "title": {"text": "Water health", "font": {"size": 12, "color": "#1A1A1A"}},
                "font": {"size": 12, "color": "#1A1A1A"},
            },
        )

        st.caption(f"Showing {len(shown):,} of {len(filtered):,} places tested.")

        if not st.session_state["_map_at_home"]:
            if st.button("Reset map view", help="Return to the full England view"):
                st.session_state["_map_center"] = DEFAULT_CENTER
                st.session_state["_map_zoom"] = DEFAULT_ZOOM
                st.session_state["_map_at_home"] = True
                st.session_state["_map_uirevision"] = "freshwater-risk-map-home"
                st.rerun()

        click_result = st.plotly_chart(
            fig, use_container_width=True, key="riskmap",
            on_select="rerun", selection_mode="points",
        )

        click_points = (click_result or {}).get("selection", {}).get("points", [])
        if click_points:
            pt = click_points[0]
            click_lat, click_lon = pt.get("lat"), pt.get("lon")
            clicked_fww_id = None
            if click_lat is not None and click_lon is not None:
                dist = ((shown["lat"] - click_lat) ** 2 + (shown["lon"] - click_lon) ** 2)
                nearest_idx = dist.idxmin()
                clicked_fww_id = shown.loc[nearest_idx, "fww_id"]

            if clicked_fww_id is not None:
                click_key = f"click:{clicked_fww_id}"
                if click_key != st.session_state.get("_last_shown_click"):
                    st.session_state["_active_selection"] = ("click_id", clicked_fww_id)
                    st.session_state["_last_shown_click"] = click_key
                    st.session_state["_last_shown_search_id"] = None
                    st.session_state["_clear_search_flag"] = True
                    st.session_state["_force_recenter"] = True
                    st.session_state["_recenter_source"] = "click"
                    st.rerun()

tour_detail = (
    "When you click a dot, this panel shows the predicted rating, how confident "
    "we are, and which factors pushed the rating up or down. Red bars mean "
    "factors pushing toward Poor, green toward Good."
)
tour_detail_icon = (
    " <span class='rw-tour'>ℹ️<span class='rw-tour-tip'>"
    f"{tour_detail}</span></span>"
) if tour_enabled else ""

with detail_col:
    st.markdown(f"<h3> About this place{tour_detail_icon}</h3>", unsafe_allow_html=True)

    match = preds[preds["fww_id"] == selected_fww_id] if selected_fww_id is not None else pd.DataFrame()

    if not match.empty:
        pos = shap_id_to_pos.get(str(selected_fww_id))
        row = match.iloc[0]
        status = row["predicted_status"]
        colour = COLOURS.get(status, "#888")

        st.markdown(f"#### {row['display_name']}")
        st.markdown(
            f"<div style='background:{colour}15; border-left:4px solid {colour}; "
            f"padding:12px 16px; border-radius:4px; margin-bottom:12px;'>"
            f"<div style='color:{colour}; font-weight:600; font-size:18px;'>{status}</div>"
            f"<div style='font-size:14px; margin-top:4px;'>{STATUS_MEANING.get(status,'')}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        conf = max(row["prob_poor"], row["prob_moderate"])
        if conf > 0.75:
            conf_text = "We're fairly confident about this rating."
        elif conf > 0.6:
            conf_text = "We're reasonably confident about this rating."
        else:
            conf_text = (
                "This one's borderline - the water here sits close to the line "
                "between two ratings, so treat it with some caution."
            )
        st.caption(conf_text)

        # --- Location history (only shown if this exact place has repeat visits) ---
        history = get_location_history(row["site_name"], row.get("county"))
        if len(history) > 1:
            st.markdown("---")
            st.markdown("**How has this exact place changed over time?**")
            st.caption(f"This location has been sampled {len(history)} times by volunteers.")

            history["sample_date"] = pd.to_datetime(history["sample_date"])
            history_chart = (
                alt.Chart(history)
                .transform_fold(["nitrate_mid", "phosphate_mid"], as_=["Chemical", "Level"])
                .mark_line(point=True)
                .encode(
                    x=alt.X("sample_date:T", title="Sample date"),
                    y=alt.Y("Level:Q", title="mg/L measured"),
                    color=alt.Color("Chemical:N", title=None),
                    tooltip=["sample_date:T", "Level:Q", "Chemical:N"],
                )
                .properties(height=250)
            )
            st.altair_chart(history_chart, use_container_width=True)

        st.markdown("---")
        st.markdown("**Why did this place get this rating?**")

        if shap_vals is not None and pos is not None and pos < len(shap_vals):
            vals = shap_vals[pos]
            names = shap_feats.columns.tolist()
            fvals = shap_feats.iloc[pos].values

            local = pd.DataFrame({"feat": names, "shap": vals, "val": fvals})
            local["abs"] = local["shap"].abs()

            st.altair_chart(render_location_chart(local), use_container_width=True)
            st.caption("All 17 factors for this location, strongest pull either direction.")

            n_total = len(local)
            n_bad  = int((local["shap"] > 0).sum())
            n_good_n = int((local["shap"] < 0).sum())

            if status == "Poor":
                st.caption(
                    f"**{n_bad} of {n_total}** factors point toward Polluted - "
                    f"the remaining {n_total - n_bad} pointed the other way but were outweighed."
                )
            elif status == "Good":
                st.caption(
                    f"**{n_good_n} of {n_total}** factors point toward Healthy - "
                    f"the remaining {n_total - n_good_n} pointed the other way but were outweighed."
                )
            else:
                st.caption(
                    f"**{n_bad} of {n_total}** factors point toward Polluted, "
                    f"**{n_good_n} of {n_total}** point toward Healthy - a genuine balance."
                )

            pull_poor = local[local["shap"] > 0]["shap"].sum()
            pull_healthy = local[local["shap"] < 0]["shap"].sum()

            st.caption(
                f"Total pull toward Polluted: {pull_poor:+.3f} · "
                f"Total pull toward Healthy: {pull_healthy:+.3f}. "
                "The final rating reflects the sum of every factor, not just the strongest few."
            )
            st.warning(
                "**Read these carefully.** These show statistical patterns the "
                "system found, not proven cause and effect. Sewage monitors are "
                "mostly installed in towns and cities, so raw spill numbers alone "
                "can be misleading - we correct for this by looking at spills per "
                "monitored pipe rather than the total count.",
                icon="⚠️",
            )

            # --- Regional comparison - genuinely location-specific, so it
            # lives here in the detail panel, not in the map column. ---
            county = row.get("county")
            if county:
                regional = get_regional_comparison(county)
                if len(regional) > 5:
                    st.markdown("---")
                    st.markdown(f"**How does this compare to other rivers in {county}?**")
                    pct_poor_here = 100 if status == "Poor" else 0
                    pct_poor_region = (regional["wfd_status"] == "Poor").mean() * 100
                    comp_df = pd.DataFrame({
                        "Location": ["This river", f"{county} average"],
                        "% rated Poor": [pct_poor_here, pct_poor_region],
                    })
                    comp_chart = (
                        alt.Chart(comp_df)
                        .mark_bar(color="#B33A3A")
                        .encode(x="Location:N", y="% rated Poor:Q")
                        .properties(height=200)
                    )
                    st.altair_chart(comp_chart, use_container_width=True)
                    st.caption(f"Based on {len(regional)} tested locations across {county}.")
        else:
            st.caption("Detailed explanation not available for this location.")
    else:
        st.info("👈 Click a dot on the map, or search for a place, to see what's affecting that stretch of water.")


# ---- Global feature importance chart -----------------------------------------------

tour_global = (
    "This chart shows which factors matter most across all of England. "
    "Sewage spills and nitrate levels tend to be the biggest drivers. "
    "This is global importance - the same factors shown per-site on the right panel."
)
tour_global_icon = (
    " <span class='rw-tour'>ℹ️<span class='rw-tour-tip'>"
    f"{tour_global}</span></span>"
) if tour_enabled else ""

st.markdown("---")
st.markdown(f"<h3> What affects river health most across England?{tour_global_icon}</h3>", unsafe_allow_html=True)
st.caption(
    "Across all 36,000+ places we looked at, these are the factors that most "
    "influence whether water is healthy or polluted. Longer bars mean more influence."
)

if global_imp:
    display_names = {}
    for k, v in list(global_imp.items())[:8]:
        if k.startswith("lc_"):
            parts = k.split("_")
            kind = LC_KIND_NAMES.get(parts[1], parts[1])
            radius = parts[-1]
            display_names[f"{kind.title()} ({radius})"] = v
        else:
            display_names[FEATURE_LABELS.get(k, k)] = v

    imp_df = pd.DataFrame({
        "Factor": list(display_names.keys()),
        "Influence": list(display_names.values()),
    }).sort_values("Influence", ascending=True)

    chart = (
        alt.Chart(imp_df)
        .mark_bar(color="#4A7C8C")
        .encode(
            y=alt.Y("Factor:N", sort="-x", title=None,
                    axis=alt.Axis(labelFontSize=13, labelLimit=300, labelPadding=8)),
            x=alt.X("Influence:Q", title="Influence",
                    axis=alt.Axis(labelFontSize=12, titleFontSize=13)),
            tooltip=[
                alt.Tooltip("Factor:N", title="Factor"),
                alt.Tooltip("Influence:Q", title="Influence", format=".3f"),
            ],
        )
        .properties(height=420)
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.caption("Run shap_analysis.py to generate this chart.")


# ---- County comparison chart --------------------------------------------------------

st.markdown("---")
st.markdown("### Which parts of England are most affected?")

county_summary = get_county_summary()
top_bottom = pd.concat([county_summary.head(10), county_summary.tail(10)])

st.caption(
    "Showing the 10 highest and 10 lowest counties by % Poor (out of "
    f"{len(county_summary)} counties with at least 20 tests) - not every "
    "English county has enough data to include here."
)

county_chart = (
    alt.Chart(top_bottom)
    .mark_bar()
    .encode(
        y=alt.Y("county:N", sort="-x", title=None),
        x=alt.X("pct_poor:Q", title="% rated Poor"),
        color=alt.condition(alt.datum.pct_poor > 50, alt.value("#B33A3A"), alt.value("#2E7D32")),
        tooltip=[
            alt.Tooltip("county:N", title="County"),
            alt.Tooltip("pct_poor:Q", title="% rated Poor", format=".1f"),
            alt.Tooltip("n:Q", title="Number of tests"),
        ],
    )
    .properties(height=500)
)
county_text = (
    alt.Chart(top_bottom)
    .mark_text(align="left", dx=3)
    .encode(
        y=alt.Y("county:N", sort="-x"),
        x="pct_poor:Q",
        text=alt.Text("pct_poor:Q", format=".0f"),
        tooltip=[
            alt.Tooltip("county:N", title="County"),
            alt.Tooltip("pct_poor:Q", title="% rated Poor", format=".1f"),
            alt.Tooltip("n:Q", title="Number of tests"),
        ],
    )
)
st.altair_chart(county_chart + county_text, use_container_width=True)


# ---- Yearly trend ---------------------------------------------------------------------

st.markdown("---")
st.markdown("### Has water quality changed over the years?")

yearly = get_yearly_trend()
if len(yearly) > 1:
    yearly_chart = (
        alt.Chart(yearly)
        .mark_line(point=True, color="#B33A3A")
        .encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("pct_poor:Q", title="% rated Poor"),
            tooltip=[
                alt.Tooltip("year:O", title="Year"),
                alt.Tooltip("pct_poor:Q", title="% rated Poor", format=".1f"),
                alt.Tooltip("n_samples:Q", title="Number of samples"),
            ],
        )
        .properties(height=250)
    )
    st.altair_chart(yearly_chart, use_container_width=True)
    st.caption(
        "⚠️ Different locations are sampled by volunteers each year, so this "
        "partly reflects which places happened to be tested, not a fully "
        "controlled measurement of real change over time."
    )


# ---- Calls to action ----------------------------------------------------------------

tour_action = (
    "These are concrete actions people can take - joining FreshWater Watch, "
    "reporting pollution, or finding a local river group."
)
tour_action_icon = (
    " <span class='rw-tour'>ℹ️<span class='rw-tour-tip'>"
    f"{tour_action}</span></span>"
) if tour_enabled else ""

st.markdown("---")
st.markdown(f"<h3>What can I do?{tour_action_icon}</h3>", unsafe_allow_html=True)
st.markdown("If you're concerned about river pollution, here are some ways to get involved:")

action_cols = st.columns(3, gap="small")
with action_cols[0]:
    st.markdown("""
    **Join the volunteer network**

    [FreshWater Watch](https://freshwaterwatch.org/) trains people across the UK to
    test their local rivers. No experience needed - you'll get a free kit and simple
    instructions.
    """)
with action_cols[1]:
    st.markdown("""
    **Report pollution**

    If you see pollution in a river, report it to the
    [Environment Agency](https://www.gov.uk/report-an-environmental-incident)
    or call **0800 80 70 60**. For sewage spills specifically,
    [Surfers Against Sewage](https://www.sas.org.uk/) runs a pollution alert map.
    """)
with action_cols[2]:
    st.markdown("""
    **Find your local river group**

    [The Rivers Trust](https://www.theriverstrust.org/) connects local river and
    catchment groups across England. Many run clean-up days, monitoring programmes,
    and campaigns you can join.
    """)


# ---- FAQ / methodology expanders -----------------------------------------------------

st.markdown("---")
e1, e2 = st.columns(2, gap="small")

with e1:
    with st.expander("What do the ratings mean?"):
        st.markdown("""
        Rivers in England are officially assessed by the Environment Agency using a
        European standard called the Water Framework Directive. Each stretch of water
        gets a rating:

        - **Good** - the water supports healthy wildlife and plant life
        - **Moderate** - there are signs of pollution pressure, and wildlife is affected
        - **Poor** - the water is significantly polluted and wildlife is struggling

        Official assessments happen only every few years and don't cover every stretch
        of water. This tool predicts ratings for places between official assessments,
        using patterns learned from thousands of locations.
        """)

    with st.expander("Where does the information come from?"):
        st.markdown("""
        We bring together four public sources:

        **Water tests by volunteers** - thousands of people across England test their
        local rivers for nitrate and phosphate, two chemicals that indicate pollution
        from farming and sewage. This comes from the
        [FreshWater Watch project](https://freshwaterwatch.org/).

        **Sewage discharge records** - water companies must record every time they
        release untreated sewage into rivers. We use these records to know how much
        sewage pressure each stretch of water is under.

        **Official water quality assessments** - the
        [Environment Agency's](https://www.gov.uk/government/organisations/environment-agency)
        formal ratings, which we use to teach the system what healthy and unhealthy
        water looks like.

        **Land maps** - what the land around each river looks like: farmland, towns,
        woodland or wetland. This comes from the
        [UK Centre for Ecology & Hydrology](https://www.ceh.ac.uk/). This matters
        because rainwater carries pollution off farmland and streets into rivers.
        """)

    with st.expander("What do the factors mean?"):
        st.markdown("""
        **Spills per sewage pipe** - on average, how many times each nearby
        sewage overflow pipe has discharged. Higher usually means more
        sewage pressure on the water.

        **Sewage overflow pipes nearby** - how many monitored sewage
        overflow points are near this location. More pipes usually means
        a more built-up drainage network nearby, not necessarily worse
        water on its own.

        **Farmland / Woodland / Grassland / Wetland / Built-up area** -
        shown as the actual area (km²) within 1 km or 5 km. Farmland and
        built-up areas tend to add pollution pressure; woodland and
        wetland tend to help filter it out.

        **Open water within 1 km / 5 km** - how much lake, pond or other
        open water is nearby. This can dilute pollution, so its absence
        can sometimes be a mild negative sign.

        **Nitrate / Phosphate in the water** - chemicals mainly from
        farming and sewage. Higher levels generally mean more pollution
        pressure.
        """)

with e2:
    with st.expander("How does the prediction work?"):
        st.markdown("""
        The system learns patterns from over 36,000 places where we know both the
        water test results and the official health rating. It works out which
        combinations of chemistry, sewage pressure, and surrounding land tend to go
        with healthy or unhealthy water.

        Once it's learned those patterns, it can estimate the health of a stretch of
        water - and explain which factors pushed the rating up or down.

        We tested the system carefully by splitting England into separate regions,
        training on some regions and testing on others it had never seen. This makes
        sure it genuinely learns real patterns rather than memorising specific places.
        """)

    with st.expander("How reliable is this?"):
        st.markdown("""
        This is a research tool, not an official assessment. It gets the rating right
        about two thirds of the time when tested against official Environment Agency
        assessments.

        It's better at spotting water under moderate pressure than at catching the
        most polluted stretches, so treat a 'Moderate' rating as a reason to look more
        closely rather than a clean bill of health.

        **This tool does not tell you whether water is safe to swim in, drink, or let
        pets into.** Always check official Environment Agency guidance for that.
        """)

st.markdown("---")
st.caption(
    "Created as part of an MSc Data Science and Artificial Intelligence dissertation "
    "at Oxford Brookes University. Uses public data from the "
    "[Environment Agency](https://www.gov.uk/government/organisations/environment-agency), "
    "[FreshWater Watch / Earthwatch](https://freshwaterwatch.org/), and the "
    "[UK Centre for Ecology & Hydrology](https://www.ceh.ac.uk/)."
)