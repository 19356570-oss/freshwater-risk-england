"""
dashboard.py
Public-facing dashboard - Freshwater Ecological Risk, England.
COMP7039 MSc Dissertation

Written for non-specialist users: plain language, no ML jargon.
Run:  streamlit run src/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pyproj import Transformer
import json
import os
from config import RESULTS_DIR
from db_loader import get_conn


st.set_page_config(
    page_title="Is My River Healthy? | England Freshwater Risk",
    page_icon="🌊",
    layout="wide",
)

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


def describe_lc_factor(feat_name, value):
    parts = feat_name.split("_")
    kind = LC_KIND_NAMES.get(parts[1], parts[1])
    radius = parts[-1]
    if value < 0.5:  # rounds to 0% anyway - "No X" reads more naturally than "0% X"
        return f"No {kind} within {radius}"
    return f"{value:.0f}% {kind} within {radius}"


def describe_level(feat_name, value, percentiles_df):
    if feat_name not in percentiles_df.index:
        return ""
    p25, p75 = percentiles_df.loc[feat_name, "p25"], percentiles_df.loc[feat_name, "p75"]
    if value <= p25:
        return "low compared to other places"
    elif value >= p75:
        return "high compared to other places"
    return "a typical level"


@st.cache_data(ttl=300)  # refresh from DB every 5 minutes, not just on restart
def load_all_data():
    conn = get_conn()
    # For each fww_id, prefer the most recent LIVE prediction if one exists
    # (reflects current sewage discharge conditions), otherwise fall back
    # to the historical prediction. This is what makes the dashboard
    # genuinely "AI-integrated" - predictions update as new data arrives,
    # not just a static one-time snapshot.
    # County is joined in from feat_matrix so every part of the dashboard
    # (map hover, search, detail panel) can consistently disambiguate
    # points that share the same site_name, or replace unhelpful generic
    # names like "Other" with something meaningful.
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

        # shap_values.npy has no fww_id column of its own - it was computed
        # against feat_matrix in that table's row order. We cannot assume
        # preds (built from the predictions table, which can have several
        # rows per fww_id from historical+live) has the SAME row order as
        # feat_matrix - it does not. Build an explicit fww_id -> row lookup
        # here, once, so every location is matched to its SHAP values by
        # ID, never by position, regardless of how differently the two
        # tables happen to be ordered.
        conn2 = get_conn()
        fm_ids = pd.read_sql("SELECT fww_id FROM feat_matrix", conn2)["fww_id"]
        conn2.close()
        # normalise to string - feat_matrix stores fww_id as text, but
        # predictions may store it differently. A silent type mismatch
        # here (e.g. int vs str) makes every single lookup fail, which is
        # exactly what caused the SHAP/prediction mismatch bug.
        shap_id_to_pos = {str(fww_id): i for i, fww_id in enumerate(fm_ids)}

    except FileNotFoundError:
        shap_vals, shap_feats, global_imp, shap_id_to_pos = None, None, {}, {}

    return preds, shap_vals, shap_feats, global_imp, shap_id_to_pos


@st.cache_data(ttl=3600)  # percentiles change rarely - hourly is enough
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


preds, shap_vals, shap_feats, global_imp, shap_id_to_pos = load_all_data()
percentiles = load_feature_percentiles()


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

# Multiple FWW sampling points can share the same site_name (e.g. several
# "River Windrush" locations along its length, tested on different dates
# and in different counties). Disambiguate using county and sample date -
# real, meaningful details a person would naturally use to tell two
# visits to "the same-named river" apart.
site_counts = preds["site_name"].value_counts()

# Some FWW volunteers entered a generic placeholder rather than a real
# name (about 180 rows say just "Other"). Showing that alone is not
# meaningful to a user, so use the county in its place when possible.
GENERIC_SITE_NAMES = {"other", "n/a", "unknown", "unnamed", ""}

def clean_site_name(name, county=None):
    """Return a user-friendly name, falling back to county for generic/placeholder names."""
    if isinstance(name, str) and name.strip().lower() in GENERIC_SITE_NAMES:
        if pd.notna(county) and county:
            return f"Unnamed site, {county}"
        return "Unnamed site"
    return name

def make_display_name(row):
    name = row["site_name"]
    county = row.get("county")
    clean_name = clean_site_name(name, county)

    if clean_name != name:
        return clean_name  # already includes county if relevant
    if site_counts.get(name, 0) <= 1:
        return name
    return f"{name}, {county}" if pd.notna(county) and county else name

# Add a display_name to preds itself so the map hover tooltip shows the
# EXACT SAME disambiguated label used in search and the detail panel -
# previously the map showed raw site_name while search/detail showed a
# cleaned-up name, which looked like a mismatch even when the underlying
# point selection was correct.
preds["display_name"] = preds.apply(make_display_name, axis=1)

site_lookup = preds[["fww_id", "display_name"]].drop_duplicates(subset="fww_id")
site_lookup = site_lookup.rename(columns={"display_name": "display"})

display_to_fww_id = dict(zip(site_lookup["display"], site_lookup["fww_id"]))
all_site_names = sorted(site_lookup["display"].tolist())

# If a map click just took over the selection, clear the search box before
# it gets created this run - Streamlit does not allow changing a widget's
# state after it has already been instantiated, so this must happen first.
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

st.sidebar.markdown("---")
max_points = st.sidebar.slider(
    "Number of locations to show", 200, 5000, 1500, step=100,
    help="Showing fewer locations makes the map load faster."
)

st.sidebar.markdown("---")
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


st.title("How healthy are England's rivers?")
st.markdown(
    "This map shows the predicted health of rivers and streams across England. "
    "We combine water test results from volunteers, records of sewage discharges, "
    "and information about the surrounding land to estimate how each stretch of "
    "water is doing - and importantly, **why**."
)

filtered = preds[preds["predicted_status"].isin(status_filter)]

n_good = (preds["predicted_status"] == "Good").sum()
n_mod  = (preds["predicted_status"] == "Moderate").sum()
n_poor = (preds["predicted_status"] == "Poor").sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Places tested", f"{len(preds):,}")
c2.metric("Healthy (Good)", f"{n_good:,}")
c3.metric("Under pressure (Moderate)", f"{n_mod:,}")
c4.metric("Polluted (Poor)", f"{n_poor:,}")

if n_good == 0:
    st.info(
        "**None of England's rivers currently reach 'Good' health.** "
        "This isn't a gap in our data - it reflects the official Environment Agency "
        "assessment, where no English river waterbody achieved Good ecological status "
        "in the most recent 2022 assessment."
    )

st.markdown("---")

map_col, detail_col = st.columns([3, 2])

# Step 1: apply a fresh search pick immediately (no click involved yet).
if searched_fww_id is not None and searched_fww_id != st.session_state.get("_last_shown_search_id"):
    st.session_state["_active_selection"] = ("search", searched_fww_id)
    st.session_state["_last_shown_search_id"] = searched_fww_id

selection = st.session_state.get("_active_selection")
selected_fww_id = None
if selection:
    if selection[0] in ("search", "click_id"):
        selected_fww_id = selection[1]

with map_col:
    st.markdown("### Explore the map")
    st.caption("Click any dot to find out why that stretch of water got its rating.")

    if filtered.empty:
        st.warning(
            f"There are no locations with a **{', '.join(status_filter)}** rating. "
            "Try ticking a different rating in the sidebar."
        )
    else:
        # shown is built once, deterministically, and NEVER modified based
        # on the current selection - point_index from a click must always
        # mean the same row here, run to run, or clicks resolve to the
        # wrong location entirely. The selected point is made visible via
        # a separate highlight overlay below instead, not by injecting it
        # into this base layer.
        shown = filtered.sample(max_points, random_state=42) if len(filtered) > max_points else filtered
        shown = shown.reset_index(drop=True)

        map_center = {"lat": 52.8, "lon": -1.6}
        map_zoom = 5
        searched_row = None

        if selected_fww_id is not None:
            match_for_map = preds[preds["fww_id"] == selected_fww_id]
            if not match_for_map.empty:
                searched_row = match_for_map.iloc[0]
                map_center = {"lat": searched_row["lat"], "lon": searched_row["lon"]}
                map_zoom = 11  # close enough to clearly distinguish the single point
                # NOTE: deliberately not adding this point to `shown` - see
                # comment above. It's drawn via the highlight overlay only.

        fig = px.scatter_map(
            shown,
            lat="lat", lon="lon",
            color="predicted_status",
            color_discrete_map=COLOURS,
            hover_name="display_name",  # matches search/detail panel exactly
            hover_data={"predicted_status": True, "lat": False, "lon": False},
            zoom=map_zoom, center=map_center,
            height=520,
            map_style="carto-positron",
            category_orders={"predicted_status": ["Good", "Moderate", "Poor"]},
            labels={"predicted_status": "Water health"},
        )
        fig.update_traces(marker={"size": 7, "opacity": 0.8})

        # draw a highlight ring around the selected point so it stands out
        # clearly from the surrounding dots, even at a zoomed-out view
        if searched_row is not None:
            fig.add_scattermap(
                lat=[searched_row["lat"]], lon=[searched_row["lon"]],
                mode="markers",
                marker={"size": 16, "color": "#222222", "opacity": 0.9},
                showlegend=False,
                hoverinfo="skip",
            )
            fig.add_scattermap(
                lat=[searched_row["lat"]], lon=[searched_row["lon"]],
                mode="markers",
                marker={
                    "size": 9,
                    "color": COLOURS.get(searched_row["predicted_status"], "#888"),
                },
                showlegend=False,
                hovertext=searched_row["display_name"],
                hoverinfo="text",
            )

        fig.update_layout(
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            legend={
                "yanchor": "bottom", "y": 0.03,
                "xanchor": "left",   "x": 0.02,
                "bgcolor": "rgba(255,255,255,0.92)",
                "bordercolor": "rgba(0,0,0,0.15)",
                "borderwidth": 1,
                "title": {"text": "Water health", "font": {"size": 12}},
                "font": {"size": 12},
            },
        )

        st.caption(f"Showing {len(shown):,} of {len(filtered):,} places tested.")

        # Step 2: render the map and capture any click in THIS SAME run -
        # st.plotly_chart's return value is fresh right now, unlike reading
        # session_state["riskmap"] which reflects the previous run's state.
        click_result = st.plotly_chart(
            fig, use_container_width=True, key="riskmap",
            on_select="rerun", selection_mode="points",
        )

        click_points = (click_result or {}).get("selection", {}).get("points", [])
        if click_points:
            pt = click_points[0]
            # Use the click's actual lat/lon rather than point_index - at
            # different zoom levels Plotly's index into the rendered view
            # was found to sometimes resolve to the wrong row (e.g. one
            # river's point clicked, a completely different one's data
            # shown). Coordinates are unambiguous regardless of zoom,
            # pan, or how many points are currently in view.
            click_lat, click_lon = pt.get("lat"), pt.get("lon")
            clicked_fww_id = None
            if click_lat is not None and click_lon is not None:
                dist = ((shown["lat"] - click_lat) ** 2 + (shown["lon"] - click_lon) ** 2)
                nearest_idx = dist.idxmin()
                clicked_fww_id = shown.loc[nearest_idx, "fww_id"]

            if clicked_fww_id is not None:
                click_key = f"click:{clicked_fww_id}"
                if click_key != st.session_state.get("_last_shown_click"):
                    # A genuinely new click just happened. The map figure
                    # above was already built and sent to the browser BEFORE
                    # we could know about this click, so its centre/zoom/
                    # highlight cannot reflect it - only a fresh run,
                    # starting from scratch with this selection already
                    # known, can draw the map correctly centred on it.
                    st.session_state["_active_selection"] = ("click_id", clicked_fww_id)
                    st.session_state["_last_shown_click"] = click_key
                    st.session_state["_last_shown_search_id"] = None
                    st.session_state["_clear_search_flag"] = True
                    st.rerun()

with detail_col:
    st.markdown("### About this place")

    # selection already resolved before the columns, shared with the map above
    match = preds[preds["fww_id"] == selected_fww_id] if selected_fww_id is not None else pd.DataFrame()

    if not match.empty:
        # Look up this point's SHAP row by fww_id, not by position - preds
        # and feat_matrix are not guaranteed to be in the same row order
        # (predictions has multiple rows per fww_id from historical+live,
        # restructured via the latest-per-id join, which reorders things).
        # Using position here previously showed one location's rating
        # alongside a COMPLETELY DIFFERENT location's SHAP explanation.
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

        st.markdown("---")
        st.markdown("**Why did this place get this rating?**")

        if shap_vals is not None and pos is not None and pos < len(shap_vals):
            vals = shap_vals[pos]
            names = shap_feats.columns.tolist()
            fvals = shap_feats.iloc[pos].values

            local = pd.DataFrame({"feat": names, "shap": vals, "val": fvals})
            local["abs"] = local["shap"].abs()

            n_total = len(local)
            n_bad  = int((local["shap"] > 0).sum())
            n_good_n = int((local["shap"] < 0).sum())

            if status == "Poor":
                st.caption(
                    f"**{n_bad} of {n_total}** factors point toward Polluted - "
                    f"shown below. The remaining {n_total - n_bad} pointed the "
                    "other way but were outweighed."
                )
            elif status == "Good":
                st.caption(
                    f"**{n_good_n} of {n_total}** factors point toward Healthy - "
                    f"shown below. The remaining {n_total - n_good_n} pointed the "
                    "other way but were outweighed."
                )
            else:
                st.caption(
                    f"**{n_bad} of {n_total}** factors point toward Polluted, "
                    f"**{n_good_n} of {n_total}** point toward Healthy - "
                    "a genuine balance, shown below."
                )

            show_all = st.checkbox("Show all factors, not just the top ones", key="show_all_shap")

            if show_all:
                local_display = local.sort_values("abs", ascending=False)
                st.caption(f"All {n_total} factors, strongest first:")
            elif status == "Poor":
                local_display = local[local["shap"] > 0].nlargest(5, "abs")
                st.caption("The five things that made this place polluted:")
            elif status == "Good":
                local_display = local[local["shap"] < 0].nlargest(5, "abs")
                st.caption("The five things that kept this place healthy:")
            else:
                bad  = local[local["shap"] > 0].nlargest(3, "abs")
                good = local[local["shap"] < 0].nlargest(3, "abs")
                local_display = pd.concat([bad, good])
                st.caption(
                    "This place is a balance of concerning and reassuring "
                    "factors. Here are the three strongest on each side:"
                )

            showing_balanced = (not show_all) and status == "Moderate"
            last_group = None

            for _, r in local_display.iterrows():
                worsens = r["shap"] > 0
                icon = "🔴" if worsens else "🔵"
                direction_text = (
                    "pushed this rating toward Polluted" if worsens
                    else "pushed this rating toward Healthy"
                )

                if showing_balanced:
                    group = "Concerning factors" if worsens else "Reassuring factors"
                    if group != last_group:
                        st.markdown(f"**{group}**")
                        last_group = group

                if r["feat"].startswith("lc_"):
                    label = describe_lc_factor(r["feat"], r["val"])
                    val_str = ""
                else:
                    label = FEATURE_LABELS.get(r["feat"], r["feat"])
                    if r["feat"] in ("nitrate_mid", "phosphate_mid"):
                        val_str = f"{r['val']:.2f} mg/L measured"
                    elif r["feat"] == "spills_per_pipe":
                        val_str = f"{r['val']:,.0f} spills per pipe on average"
                    else:
                        val_str = f"{r['val']:,.0f}"

                how_high = describe_level(r["feat"], r["val"], percentiles)
                how_high_str = f" ({how_high})" if how_high else ""

                if val_str:
                    detail = f"<span style='color:#666; font-size:13px;'>{val_str}{how_high_str}</span>"
                elif how_high:
                    detail = f"<span style='color:#666; font-size:13px;'>{how_high}</span>"
                else:
                    detail = ""

                st.markdown(
                    f"{icon} **{label}** - {direction_text}  \n{detail}",
                    unsafe_allow_html=True,
                )

            pull_poor = local[local["shap"] > 0]["shap"].sum()
            pull_healthy = local[local["shap"] < 0]["shap"].sum()

            st.caption("🔴 pushed toward Polluted · 🔵 pushed toward Healthy")
            st.caption(
                f"Total pull toward Polluted: {pull_poor:+.3f} · "
                f"Total pull toward Healthy: {pull_healthy:+.3f} "
                f"(across all {n_total} factors, not just those shown above). "
                "The final rating reflects the sum of every factor, not just "
                "the strongest few."
            )
            st.warning(
                "**Reading these carefully.** These show statistical patterns the "
                "system found, not proven cause and effect. Sewage monitors are "
                "mostly installed in towns and cities, so raw spill numbers alone "
                "can be misleading - we correct for this by looking at spills per "
                "monitored pipe rather than the total count.",
                icon="⚠️",
            )
        else:
            st.caption("Detailed explanation not available for this location.")
    else:
        st.info("👈 Click a dot on the map, or search for a place, to see what's affecting that stretch of water.")


st.markdown("---")
st.markdown("### What affects river health most across England?")
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

    st.bar_chart(imp_df.set_index("Factor"), horizontal=True, height=340, color="#4A7C8C")
else:
    st.caption("Run shap_analysis.py to generate this chart.")


st.markdown("---")
e1, e2 = st.columns(2)

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
        from farming and sewage. This comes from the FreshWater Watch project.

        **Sewage discharge records** - water companies must record every time they
        release untreated sewage into rivers. We use these records to know how much
        sewage pressure each stretch of water is under.

        **Official water quality assessments** - the Environment Agency's formal
        ratings, which we use to teach the system what healthy and unhealthy water
        looks like.

        **Land maps** - what the land around each river looks like: farmland, towns,
        woodland or wetland. This matters because rainwater carries pollution off
        farmland and streets into rivers.
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
        what percentage of the land within 1 km or 5 km is that type.
        Farmland and built-up areas tend to add pollution pressure; woodland
        and wetland tend to help filter it out.

        **Open water within 1 km / 5 km** - how much lake, pond or other
        open water is nearby. This can dilute pollution, so its absence
        can sometimes be a mild negative sign.

        **Nitrate / Phosphate in the water** - chemicals mainly from
        farming and sewage. Higher levels generally mean more pollution
        pressure.

        We describe each value as "high", "low", or "typical" by comparing
        it to all the other places we have data for - so "high" means
        higher than most other UK rivers we have tested, not a fixed
        official threshold.
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

with st.expander("🔧 System status (data pipeline monitoring)"):
    st.caption(
        "Read-only view of the automated data pipeline. Use the separate "
        "admin tool (pipeline_admin.py) to control the scheduler."
    )
    try:
        log_df = pd.read_sql("""
            SELECT source, status, records_fetched, records_new, run_at
            FROM ingestion_log
            ORDER BY run_at DESC
            LIMIT 15
        """, get_conn())

        if not log_df.empty:
            def status_icon(s):
                if s == "success":
                    return "✅"
                elif s == "no_new_data":
                    return "⏸️"
                elif s == "failed":
                    return "❌"
                return "⚠️"

            log_df["  "] = log_df["status"].apply(status_icon)
            log_df = log_df[["  ", "source", "status", "records_fetched", "records_new", "run_at"]]
            log_df.columns = ["", "Job", "Status", "Records Checked", "New Records", "Run At"]

            st.dataframe(log_df, hide_index=True, use_container_width=True)

            last_poopy = log_df[log_df["Job"].str.contains("edm_live", na=False)]
            if not last_poopy.empty:
                last_time = last_poopy.iloc[0]["Run At"]
                st.caption(f"Last live sewage data pull: {last_time}")
        else:
            st.caption("No pipeline runs logged yet.")
    except Exception as e:
        st.caption(f"Could not load pipeline status: {e}")

st.caption(
    "Created as part of an MSc Data Science and Artificial Intelligence dissertation "
    "at Oxford Brookes University. Uses public data from the Environment Agency, "
    "FreshWater Watch (Earthwatch), and the UK Centre for Ecology and Hydrology."
)