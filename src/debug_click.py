"""
debug_click.py
Minimal test to see EXACTLY what st.plotly_chart's click event returns.
Run: streamlit run src/debug_click.py
Click a few points and read the raw JSON printed on screen.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from db_loader import get_conn
from pyproj import Transformer

st.title("Click debug")

conn = get_conn()
df = pd.read_sql("SELECT fww_id, site_name, easting, northing FROM feat_matrix LIMIT 50", conn)
conn.close()

proj = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
lons, lats = proj.transform(df["easting"].values, df["northing"].values)
df["lat"], df["lon"] = lats, lons

fig = px.scatter_map(
    df, lat="lat", lon="lon", hover_name="site_name",
    zoom=6, height=500, map_style="carto-positron"
)

# reproduce the dashboard's extra highlight traces
fig.add_scattermap(
    lat=[df.iloc[5]["lat"]], lon=[df.iloc[5]["lon"]],
    mode="markers", marker={"size": 16, "color": "#222"},
    showlegend=False, hoverinfo="skip",
)
fig.add_scattermap(
    lat=[df.iloc[5]["lat"]], lon=[df.iloc[5]["lon"]],
    mode="markers", marker={"size": 9, "color": "red"},
    showlegend=False, hovertext="Highlighted point", hoverinfo="text",
)

result = st.plotly_chart(
    fig, use_container_width=True, key="debugmap",
    on_select="rerun", selection_mode="points"
)

st.write("### Raw click result:")
st.json(result)

if result and result.get("selection", {}).get("points"):
    pt = result["selection"]["points"][0]
    st.write("### First point's keys and values:")
    for k, v in pt.items():
        st.write(f"**{k}**: {v}")

    idx = pt.get("point_index")
    if idx is not None:
        st.write(f"### df.iloc[{idx}] (what point_index resolves to):")
        st.write(df.iloc[idx])