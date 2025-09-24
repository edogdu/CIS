import streamlit as st
import pandas as pd
import plotly.express as px
from psycopg.rows import dict_row
import psycopg
from neo4j import GraphDatabase
from streamlit_agraph import agraph, Node, Edge, Config

st.set_page_config(page_title="Cyber Alerts", layout="wide")
st.title("🔐 Cybersecurity Alerts")

# ---------- Connections (cached) ----------
@st.cache_resource
def get_pg():
    return psycopg.connect(st.secrets["postgres"]["dsn"], row_factory=dict_row)

@st.cache_resource
def get_neo():
    drv = GraphDatabase.driver(
        st.secrets["neo4j"]["uri"],
        auth=(st.secrets["neo4j"]["user"], st.secrets["neo4j"]["password"])
    )
    return drv

# ---------- Queries (cached where useful) ----------
@st.cache_data(ttl=60)
def list_systems():
    sql = """
      SELECT DISTINCT system_id
      FROM assets
      ORDER BY system_id;
    """
    with get_pg().cursor() as cur:
        cur.execute(sql)
        return pd.DataFrame(cur.fetchall())

#@st.cache_data(ttl=60)
def list_alerts(system_id: str, limit: int = 300):
    return pd.DataFrame()

@st.cache_data(ttl=60)
def alert_details(alert_id: str):
    return pd.DataFrame()

@st.cache_data(ttl=60)
def xai_top_features(alert_id: str, method: str = "shap", k: int = 20):
    # Expect a table like: (alert_id, method, feature, importance)
    return pd.DataFrame()

@st.cache_data(ttl=300)
def mitre_for_alert(alert_id: str):
    return pd.DataFrame()

@st.cache_data(ttl=60)
def neo4j_graph_for_snapshot(snapshot_id: str):
    return pd.DataFrame()

# ---------- UI: Systems ----------
with st.sidebar:
    st.header("Filters")
systems_df = list_systems()
if systems_df.empty:
    st.info("No systems found.")
    st.stop()

left, right = st.columns([1, 3], gap="large")

with left:
    st.subheader("Systems")
    st.caption("Click to select a system")
    sys_choice = st.selectbox(
        "System",
        systems_df["system_id"],
        index=0,
        label_visibility="collapsed",
    )
    st.markdown("---")

with right:
    st.subheader(f"Alerts for System: {sys_choice}")
    alerts_df = list_alerts(sys_choice)
    if alerts_df.empty:
        st.info("No alerts found for this system.")
        st.stop()