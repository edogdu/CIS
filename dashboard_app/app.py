# app.py — Amiran Cyber Intelligence Dashboard (Wireframe + colored nodes + NVL optional)
import os
import pandas as pd
import streamlit as st
import plotly.express as px
from psycopg.rows import dict_row
import psycopg
from neo4j import GraphDatabase
from streamlit_agraph import agraph, Node, Edge, Config

# ---- TRY NVL (optional) -----------------------------------------------------
# If NVL (Neo4j Visualization Library for Python) is available, we'll use it.
# If not, we silently fall back to streamlit-agraph.
_HAS_NVL = False
try:
    # The package name may vary depending on the preview build.
    # If your install uses a different import path, change it here.
    # Example names people use in preview builds: `nvl`, `neo4j_nvl`, `nvl_python`
    import nvl  # noqa: F401
    _HAS_NVL = True
except Exception:
    _HAS_NVL = False

st.set_page_config(page_title="Amiran — Cyber Intelligence Dashboard", layout="wide")

# ------------------------ DSN helpers: Docker & Local ------------------------
def _pg_dsn():
    # 1) secrets.toml if present
    try:
        dsn = st.secrets["postgres"]["dsn"]
        if dsn:
            return dsn
    except Exception:
        pass
    # 2) env (Docker) or local fallback
    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT") or ("5445" if host == "localhost" else "5432")
    user = os.getenv("PGUSER", "postgres")
    pwd  = os.getenv("PGPASSWORD", "password")
    db   = os.getenv("PGDATABASE", "anomalydetection")
    ssl  = os.getenv("PGSSLMODE", "disable")
    return f"postgresql://{user}:{pwd}@{host}:{port}/{db}?sslmode={ssl}"

def _neo_conf():
    # 1) secrets.toml
    try:
        return (st.secrets["neo4j"]["uri"], st.secrets["neo4j"]["user"], st.secrets["neo4j"]["password"])
    except Exception:
        pass
    # 2) env or localhost
    return (
        os.getenv("NEO4J_URI", "bolt://localhost:7698"),
        os.getenv("NEO4J_USER", "neo4j"),
        os.getenv("NEO4J_PASSWORD", "changeme"),
    )

# ------------------------ Connections (cached) ------------------------
@st.cache_resource(show_spinner=False)
def get_pg():
    try:
        return psycopg.connect(_pg_dsn(), row_factory=dict_row)
    except Exception as e:
        st.sidebar.error(f"Postgres connection failed: {e}")
        return None

@st.cache_resource(show_spinner=False)
def get_neo():
    try:
        uri, user, pwd = _neo_conf()
        return GraphDatabase.driver(uri, auth=(user, pwd))
    except Exception as e:
        st.sidebar.error(f"Neo4j connection failed: {e}")
        return None

# ------------------------ Queries (cached) ------------------------
@st.cache_data(ttl=60, show_spinner=False)
def list_systems():
    sql = "SELECT DISTINCT system_id FROM assets ORDER BY system_id;"
    conn = get_pg()
    if not conn: return pd.DataFrame(columns=["system_id"])
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            return pd.DataFrame(cur.fetchall())
    except Exception:
        return pd.DataFrame(columns=["system_id"])

@st.cache_data(ttl=120, show_spinner=False)
def kpis_overview():
    conn = get_pg()
    out = {"active_systems": 0, "systems_with_alerts": 0, "critical_alerts": 0}
    if not conn: return out
    try:
        with conn.cursor() as cur:
            try:
                cur.execute("SELECT COUNT(DISTINCT system_id) AS c FROM assets WHERE active = TRUE;")
                out["active_systems"] = cur.fetchone()["c"]
            except Exception:
                cur.execute("SELECT COUNT(DISTINCT system_id) AS c FROM assets;")
                out["active_systems"] = cur.fetchone()["c"]
            try:
                cur.execute("SELECT COUNT(DISTINCT system_id) AS c FROM alerts;")
                out["systems_with_alerts"] = cur.fetchone()["c"]
            except Exception:
                out["systems_with_alerts"] = 0
            try:
                cur.execute("SELECT COUNT(*) AS c FROM alerts WHERE severity = 'CRITICAL';")
                out["critical_alerts"] = cur.fetchone()["c"]
            except Exception:
                out["critical_alerts"] = 0
    except Exception:
        pass
    return out

@st.cache_data(ttl=120, show_spinner=False)
def alerts_timeline(days=7):
    conn = get_pg()
    if not conn: return pd.DataFrame(columns=["ts", "cnt"])
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT date_trunc('hour', ts) AS ts, COUNT(*) AS cnt
                FROM alerts
                WHERE ts >= NOW() - INTERVAL %s
                GROUP BY 1
                ORDER BY 1;
                """,
                (f"{days} days",),
            )
            return pd.DataFrame(cur.fetchall())
    except Exception:
        return pd.DataFrame(columns=["ts", "cnt"])

@st.cache_data(ttl=120, show_spinner=False)
def list_alerts(system_id: str, limit: int = 400):
    conn = get_pg()
    if not conn: return pd.DataFrame()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id AS alert_id, ts, severity, src, prob, status, description, system_id, asset_id
                FROM alerts
                WHERE system_id = %s
                ORDER BY ts DESC
                LIMIT %s;
                """,
                (system_id, limit),
            )
            return pd.DataFrame(cur.fetchall())
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def mitre_for_alert(alert_id: str):
    conn = get_pg()
    if not conn: return pd.DataFrame(columns=["technique_id", "technique", "tactic"])
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT technique_id, technique, tactic
                FROM mitre_map
                WHERE alert_id = %s
                ORDER BY technique_id;
                """,
                (alert_id,),
            )
            return pd.DataFrame(cur.fetchall())
    except Exception:
        return pd.DataFrame(columns=["technique_id", "technique", "tactic"])

# --- New: which assets are "problematic" (for coloring) ----------------------
@st.cache_data(ttl=60, show_spinner=False)
def problem_assets(snapshot_id: str = "latest", min_severity=("HIGH", "CRITICAL")) -> set:
    """
    Returns a set of asset IDs that currently have alerts with severity in min_severity.
    Adjust to your schema as needed (e.g., join by snapshot_id if you store it).
    """
    conn = get_pg()
    if not conn: return set()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT asset_id
                FROM alerts
                WHERE severity = ANY(%s)
                """,
                (list(min_severity),),
            )
            rows = cur.fetchall()
            return {r["asset_id"] for r in rows if r.get("asset_id")}
    except Exception:
        return set()

# --- Graph retrieval from Neo4j ---------------------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def neo4j_graph_for_snapshot(snapshot_id: str):
    """
    Returns node/edge lists from Neo4j.
    Nodes include `id`, `label`, optional `color` (we add after this call).
    """
    drv = get_neo()
    if not drv: return [], []
    q = """
    MATCH (s:System {snapshot_id: $snap})-[:CONTAINS]->(a:Asset)
    OPTIONAL MATCH (a)-[r:CONNECTED_TO]->(b:Asset)
    RETURN collect(DISTINCT a) AS assets, collect(DISTINCT r) AS rels, collect(DISTINCT b) AS others
    LIMIT 2000
    """
    try:
        with drv.session() as ssn:
            rec = ssn.run(q, snap=snapshot_id).single()
        if not rec: return [], []
        assets = rec["assets"] or []; others = rec["others"] or []; rels = rec["rels"] or []
        def _nid(n):
            # neo4j driver v4/v5 compatibility
            return getattr(n, "element_id", None) or getattr(n, "id", None)

        node_map = {}
        for n in assets + others:
            nid = _nid(n)
            label = n.get("name") or n.get("id") or str(nid)
            # Keep raw node with basic fields; color will be applied later
            node_map[nid] = {"id": str(nid), "label": str(label)}

        edges = []
        for r in rels:
            sid = _nid(r.start_node); tid = _nid(r.end_node)
            edges.append({"source": str(sid), "target": str(tid)})
        return list(node_map.values()), edges
    except Exception:
        return [], []

# ------------------------ Top Nav ------------------------
st.markdown("## 🔐 Cyber Intelligence Dashboard — Amiran")
tabs = st.tabs(["🏠 Overview", "⚠️ Anomaly Alerts", "🧩 Knowledge Graph", "💽 System Health", "⚙️ Settings"])

# ------------------------ 🏠 Overview ------------------------
with tabs[0]:
    k = kpis_overview()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Active Systems", k["active_systems"])
    c2.metric("Systems w/ Alerts", k["systems_with_alerts"])
    c3.metric("Critical Alerts", k["critical_alerts"])
    c4.write("**Postgres**"); c4.success("Connected") if get_pg() else c4.error("Disconnected")
    c5.write("**Neo4j**");    c5.success("Connected") if get_neo() else c5.error("Disconnected")

    st.markdown("---")
    ts_df = alerts_timeline(7)
    if ts_df.empty:
        st.info("No alert timeline data available yet.")
    else:
        fig = px.line(ts_df, x="ts", y="cnt", title="Alerts over Time (last 7 days)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Latest Alerts")
    latest = pd.DataFrame()
    conn = get_pg()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id AS alert_id, ts, system_id, severity, src, status, description
                    FROM alerts ORDER BY ts DESC LIMIT 10;
                """)
                latest = pd.DataFrame(cur.fetchall())
        except Exception:
            pass
    st.dataframe(latest if not latest.empty else pd.DataFrame(columns=["No recent alerts"]),
                 use_container_width=True)

# ------------------------ ⚠️ Anomaly Alerts ------------------------
with tabs[1]:
    left, right = st.columns([1, 3], gap="large")
    with left:
        st.subheader("Filters")
        systems_df = list_systems()
        sys_choice = st.selectbox(
            "System",
            systems_df["system_id"] if not systems_df.empty else [],
            index=0 if not systems_df.empty else None,
            placeholder="Pick a system",
        )
        date_range = st.date_input("Date range", [])
        severity = st.multiselect("Severity", ["LOW", "MEDIUM", "HIGH", "CRITICAL"], ["HIGH", "CRITICAL"])

    with right:
        st.subheader("Alerts")
        if not sys_choice:
            st.info("Choose a system to view alerts.")
        else:
            alerts = list_alerts(sys_choice, 400)
            if alerts.empty:
                st.info("No alerts found for this system.")
            else:
                if "severity" in alerts and severity:
                    alerts = alerts[alerts["severity"].isin(severity)]
                if "ts" in alerts and date_range and len(date_range) == 2:
                    start = pd.to_datetime(date_range[0])
                    end = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
                    alerts = alerts[(pd.to_datetime(alerts["ts"]) >= start) &
                                    (pd.to_datetime(alerts["ts"]) < end)]

                st.dataframe(alerts, use_container_width=True, height=420)

                if {"ts", "severity"}.issubset(alerts.columns):
                    tmp = alerts.copy()
                    tmp["ts"] = pd.to_datetime(tmp["ts"]).dt.floor("H")
                    trend = tmp.groupby(["ts", "severity"]).size().reset_index(name="cnt")
                    fig = px.line(trend, x="ts", y="cnt", color="severity", title="Alerts by Severity over Time")
                    st.plotly_chart(fig, use_container_width=True)

                if "alert_id" in alerts and not alerts.empty:
                    st.markdown("##### Explainability (Top Features)")
                    sel_alert = st.selectbox("Alert", alerts["alert_id"], key="sel_alert_xai")
                    # NOTE: Hook your XAI table here when ready
                    st.caption("XAI hookup placeholder — add your `xai_top_features()` call when table is populated.")

# ------------------------ 🧩 Knowledge Graph ------------------------
with tabs[2]:
    st.subheader("Knowledge Graph (Neo4j)")
    snapshot_id = st.text_input("Snapshot ID", value="latest", help="Enter a snapshot id stored in Neo4j")

    # Color palette options
    st.markdown("**Node coloring**")
    palette = st.selectbox(
        "Choose palette",
        ["Default (red/orange/blue/gray)", "Protanopia-friendly"],
        index=0,
        help="Pick a color set for problem vs normal nodes."
    )

    # Define colors (easily adjustable)
    if palette == "Protanopia-friendly":
        COLOR_CRIT = "#6e016b"   # purple-ish
        COLOR_HIGH = "#88419d"   # lighter purple
        COLOR_NORM = "#2171b5"   # blue
        COLOR_OTHER = "#9ebcda"  # light blue/gray
    else:
        COLOR_CRIT = "#e11d48"   # red-600
        COLOR_HIGH = "#f59e0b"   # amber-500
        COLOR_NORM = "#2563eb"   # blue-600
        COLOR_OTHER = "#9ca3af"  # gray-400

    # Legend
    lc1, lc2, lc3, lc4 = st.columns(4)
    with lc1: st.markdown(f"<div style='height:12px;width:12px;background:{COLOR_CRIT};display:inline-block;border-radius:3px'></div> **CRITICAL**", unsafe_allow_html=True)
    with lc2: st.markdown(f"<div style='height:12px;width:12px;background:{COLOR_HIGH};display:inline-block;border-radius:3px'></div> **HIGH**", unsafe_allow_html=True)
    with lc3: st.markdown(f"<div style='height:12px;width:12px;background:{COLOR_NORM};display:inline-block;border-radius:3px'></div> **Normal**", unsafe_allow_html=True)
    with lc4: st.markdown(f"<div style='height:12px;width:12px;background:{COLOR_OTHER};display:inline-block;border-radius:3px'></div> **Other**", unsafe_allow_html=True)

    use_nvl = st.checkbox("Use NVL (if installed)", value=True,
                          help="Try Neo4j Visualization Library; fallback to agraph if unavailable.")

    if st.button("Load Graph", type="primary"):
        nodes_raw, edges_raw = neo4j_graph_for_snapshot(snapshot_id)
        if not nodes_raw and not edges_raw:
            st.info("No graph data for this snapshot (or Neo4j not connected).")
        else:
            # Determine problem nodes from alerts table
            problem = problem_assets(snapshot_id, min_severity=("HIGH", "CRITICAL"))
            # Assign colors
            # We don't know per-asset severity here; we'll mark CRITICAL if it has any CRITICAL alert,
            # otherwise HIGH if it has any HIGH alert. If you store per-asset-most-recent severity,
            # swap this with a query that returns exact severity per asset.
            # For now, treat every "problem" asset as HIGH; you can refine by joining severity later.
            problem_set = set(problem)

            # Build agraph nodes/edges with colors
            a_nodes = []
            id_to_color = {}
            for n in nodes_raw:
                nid = n["id"]
                label = n["label"]
                if nid in problem_set:
                    # Use HIGH color currently; you can split CRITICAL/HIGH with a more specific query
                    color = COLOR_HIGH
                else:
                    color = COLOR_NORM
                id_to_color[nid] = color
                a_nodes.append(Node(id=nid, label=label, size=18, color=color))

            a_edges = [Edge(source=e["source"], target=e["target"]) for e in edges_raw]

            # Try NVL if selected & available
            if use_nvl and _HAS_NVL:
                try:
                    # Minimal example: build an NVL graph and style nodes by our mapping.
                    # Adjust this block to match the exact NVL Python API you install (preview builds can differ).
                    import json
                    from streamlit.components.v1 import html

                    nvl_nodes = []
                    for n in nodes_raw:
                        nvl_nodes.append({
                            "id": n["id"],
                            "label": n["label"],
                            "style": {"color": id_to_color.get(n["id"], COLOR_OTHER)}
                        })
                    nvl_edges = [{"source": e["source"], "target": e["target"]} for e in edges_raw]

                    # Simple HTML emulation wrapper until NVL exposes a direct streamlit helper:
                    payload = {"nodes": nvl_nodes, "edges": nvl_edges}
                    html(f"""
                        <div id="nvl" style="height:650px;border:1px solid #e5e7eb;border-radius:8px"></div>
                        <script>
                          const payload = {json.dumps(payload)};
                          // If your NVL build exposes a global renderer, call it here.
                          // For preview builds you might do something like:
                          // NVL.render("#nvl", payload.nodes, payload.edges, {{ physics: true }});
                          // Fallback message:
                          document.getElementById('nvl').innerHTML =
                            '<div style="padding:12px;font-family:system-ui">NVL is installed, but the preview API wrapper is not configured. Using fallback agraph below.</div>';
                        </script>
                    """, height=680)

                    # Show agraph as a reliable fallback for now
                    st.caption("NVL preview shown above (if configured). Fallback interactive graph below:")
                    config = Config(height=620, width=1200, directed=True, physics=True, hierarchical=False)
                    agraph(nodes=a_nodes, edges=a_edges, config=config)
                except Exception as e:
                    st.warning(f"NVL rendering had an issue: {e}. Falling back to agraph.")
                    config = Config(height=620, width=1200, directed=True, physics=True, hierarchical=False)
                    agraph(nodes=a_nodes, edges=a_edges, config=config)
            else:
                # Pure agraph path
                config = Config(height=620, width=1200, directed=True, physics=True, hierarchical=False)
                agraph(nodes=a_nodes, edges=a_edges, config=config)

# ------------------------ 💽 System Health ------------------------
with tabs[3]:
    st.subheader("System Health")
    pg_ok = bool(get_pg()); neo_ok = bool(get_neo())
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Postgres**")
        if pg_ok:
            st.success("Connected")
            try:
                with get_pg().cursor() as cur:
                    cur.execute("SELECT COUNT(*) AS c FROM assets;")
                    st.metric("Assets rows", cur.fetchone()["c"])
            except Exception:
                st.caption("`assets` table not found or not accessible.")
        else:
            st.error("Disconnected")
    with c2:
        st.write("**Neo4j**")
        st.success("Connected") if neo_ok else st.error("Disconnected")
    st.caption("Tip: If startup is flaky, increase Docker Desktop resources and start DB → run init → start others.")

# ------------------------ ⚙️ Settings ------------------------
with tabs[4]:
    st.subheader("Settings / Connection Details (read-only)")
    st.write("**Postgres DSN (effective):**")
    st.code(_pg_dsn(), language="text")
    uri, user, _ = _neo_conf()
    st.write("**Neo4j URI / user:**")
    st.code(f"{uri}  |  user={user}", language="text")
