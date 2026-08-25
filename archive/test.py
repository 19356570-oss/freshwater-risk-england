"""
test_map.py
Minimal test to isolate why the Folium map shows no points.
Run: streamlit run src/test_map.py
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
from db_loader import get_conn
from pyproj import Transformer

st.title("Map debug test")

conn = get_conn()
df = pd.read_sql("SELECT * FROM predictions LIMIT 20", conn)
conn.close()

st.write(f"Rows loaded: {len(df)}")
st.write(df[["easting", "northing", "predicted_status"]].head())

proj = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)

m = folium.Map(location=[52.5, -1.5], zoom_start=6)

count_added = 0
for _, row in df.iterrows():
    try:
        lon, lat = proj.transform(row["easting"], row["northing"])
        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            color="red",
            fill=True,
            fill_color="red",
            fill_opacity=1.0,
        ).add_to(m)
        count_added += 1
    except Exception as e:
        st.error(f"Error on row: {e}")

st.write(f"Markers added: {count_added}")

st_folium(m, width=700, height=500)