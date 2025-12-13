# app.py — Amiran Cyber Intelligence Dashboard (agraph pretty mode + NVL via CDN)
import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px
from psycopg.rows import dict_row
import psycopg
from neo4j import GraphDatabase
from streamlit_agraph import agraph, Node, Edge, Config
from streamlit.components.v1 import html

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
        return (
            st.secrets["neo4j"]["uri"],
            st.secrets["neo4j"]["user"],
            st.secrets["neo4j"]["password"],
        )
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
    if not conn:
        return pd.DataFrame(columns=["system_id"])
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
    if not conn:
        return out
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
    if not conn:
        return pd.DataFrame(columns=["ts", "cnt"])
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
    if not conn:
        return pd.DataFrame()
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
    if not conn:
        return pd.DataFrame(columns=["technique_id", "technique", "tactic"])
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

# --- Max severity per asset (for coloring) -----------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def asset_top_severity() -> dict:
    """
    Returns {asset_id: 'CRITICAL'|'HIGH'|'MEDIUM'|'LOW'} based on max severity.
    """
    conn = get_pg()
    if not conn:
        return {}
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT asset_id,
                       CASE
                         WHEN MAX(CASE WHEN severity='CRITICAL' THEN 1 ELSE 0 END)=1 THEN 'CRITICAL'
                         WHEN MAX(CASE WHEN severity='HIGH' THEN 1 ELSE 0 END)=1 THEN 'HIGH'
                         WHEN MAX(CASE WHEN severity='MEDIUM' THEN 1 ELSE 0 END)=1 THEN 'MEDIUM'
                         WHEN MAX(CASE WHEN severity='LOW' THEN 1 ELSE 0 END)=1 THEN 'LOW'
                         ELSE NULL
                       END AS top_sev
                FROM alerts
                GROUP BY asset_id
            """)
            rows = cur.fetchall()
            return {r["asset_id"]: r["top_sev"] for r in rows if r.get("asset_id")}
    except Exception:
        return {}

# ------------------------ Neo4j Helpers ------------------------
def _nid(n):
    # neo4j driver v4/v5 compatibility for node/relationship element id
    return getattr(n, "element_id", None) or getattr(n, "id", None)

def _prop(n, key):
    # Safely fetch node property from Neo4j Node
    try:
        return n.get(key)
    except Exception:
        try:
            return n[key]
        except Exception:
            try:
                return dict(n).get(key)
            except Exception:
                return None

# --- Graph retrieval from Neo4j (real edges; ID = asset_id when present) -----
@st.cache_data(ttl=60, show_spinner=False)
def neo4j_graph_for_snapshot(snapshot_id: str, rel_types: list[str]):
    """
    Returns (nodes, edges) from Neo4j for given snapshot and relationship types.
    nodes: {'id': <asset_id|name|element_id>, 'label': <string>, 'asset_id': <str|None>, 'labels':[...]}
    edges: {'source': <node_id>, 'target': <node_id>, 'type': <rel_type>}
    """
    drv = get_neo()
    if not drv:
        return [], []

    q = """
    MATCH (s:System {snapshot_id: $snap})-[:CONTAINS]->(a:Asset)
    OPTIONAL MATCH (a)-[r]->(b:Asset)
    WHERE type(r) IN $rels AND (s)-[:CONTAINS]->(b)
    RETURN collect(DISTINCT a) AS assets,
           collect(DISTINCT r) AS rels,
           collect(DISTINCT b) AS others
    LIMIT 5000
    """
    try:
        with drv.session() as ssn:
            rec = ssn.run(q, snap=snapshot_id, rels=rel_types).single()
        if not rec:
            return [], []

        assets = rec["assets"] or []
        others = rec["others"] or []
        rels   = rec["rels"]   or []

        def node_key(n):
            # Prefer asset_id → name → element id (string)
            return str(_prop(n, "asset_id") or _prop(n, "name") or _nid(n))

        def node_label(n):
            return str(_prop(n, "name") or _prop(n, "asset_id") or node_key(n))

        node_map = {}
        for n in assets + others:
            kid = node_key(n)
            node_map[kid] = {
                "id": kid,
                "label": node_label(n),
                "asset_id": _prop(n, "asset_id"),
                "labels": _prop(n, "labels") or [],
            }

        edges = []
        for r in rels:
            try:
                s_id = node_key(r.start_node)
                t_id = node_key(r.end_node)
                edges.append({"source": s_id, "target": t_id, "type": r.type})
            except Exception:
                continue

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
                    _ = st.selectbox("Alert", alerts["alert_id"], key="sel_alert_xai")
                    st.caption("XAI hookup placeholder — add your `xai_top_features()` call when table is populated.")

# ------------------------ 🧩 Knowledge Graph ------------------------
with tabs[2]:
    st.subheader("Knowledge Graph (Neo4j)")
    snapshot_id = st.text_input("Snapshot ID", value="latest", help="Enter a snapshot id stored in Neo4j")

    # Relationship types control
    rel_opts = st.multiselect(
        "Relationship types",
        ["FEEDS_TO", "FEEDS_THROUGH"],
        default=["FEEDS_TO", "FEEDS_THROUGH"],
        help="Select which edge types to include."
    )

    # Color palette options
    st.markdown("**Node coloring**")
    palette = st.selectbox(
        "Choose palette",
        ["Default (red/orange/blue/gray)", "Protanopia-friendly"],
        index=0,
        help="Pick a color set for problem vs normal nodes."
    )
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

    # View mode: Graph vs Table and renderer choice
    col_mode, col_renderer = st.columns([1,1])
    with col_mode:
        show_table = st.checkbox("Show as table (instead of graph)", value=False)
    with col_renderer:
        use_nvl = st.checkbox(
            "Use NVL renderer (via CDN)",
            value=True,
            help="If on, renders with @neo4j-nvl/base from a CDN. Else uses agraph pretty mode."
        )

    # Load action
    if st.button("Load", type="primary"):
        if not rel_opts:
            st.warning("Pick at least one relationship type.")
        else:
            nodes_raw, edges_raw = neo4j_graph_for_snapshot(snapshot_id, rel_opts)
            if not nodes_raw and not edges_raw:
                st.info("No graph data for this snapshot (or Neo4j not connected).")
            else:
                # Severity map & color function (used by both renderers)
                sev_map = asset_top_severity()

                def color_for(sev: str | None):
                    if sev == "CRITICAL": return COLOR_CRIT
                    if sev == "HIGH":      return COLOR_HIGH
                    return COLOR_NORM

                # Build dataframes for convenience
                nodes_df = pd.DataFrame([{
                    "id":       n["id"],
                    "label":    n["label"],
                    "asset_id": n.get("asset_id"),
                } for n in nodes_raw])

                if not nodes_df.empty:
                    nodes_df["sev_key"]  = nodes_df["asset_id"].fillna(nodes_df["id"])
                    nodes_df["severity"] = nodes_df["sev_key"].map(sev_map).fillna("normal")
                    nodes_df["color"]    = nodes_df["severity"].map(color_for)

                    # ✅ Force guaranteed label visibility (never blank)
                    nodes_df["display"] = (
                        nodes_df["label"].astype(str).str.strip()
                        .where(nodes_df["label"].astype(str).str.strip().ne(""), None)
                    )
                    nodes_df["display"] = nodes_df["display"].fillna(
                        nodes_df["asset_id"].astype(str).str.strip()
                        .where(nodes_df["asset_id"].astype(str).str.strip().ne(""), None)
                    )
                    nodes_df["display"] = nodes_df["display"].fillna(nodes_df["id"].astype(str))

                if not nodes_df.empty:
                    nodes_df["sev_key"]  = nodes_df["asset_id"].fillna(nodes_df["id"])
                    nodes_df["severity"] = nodes_df["sev_key"].map(sev_map).fillna("normal")
                    nodes_df["color"]    = nodes_df["severity"].map(color_for)

                edges_df = pd.DataFrame(
                    [{"source": e["source"], "target": e["target"], "type": e.get("type", "")} for e in edges_raw]
                )

                if show_table:
                    st.markdown("#### Nodes")
                    st.dataframe(nodes_df, use_container_width=True, height=360)
                    st.markdown("#### Edges")
                    st.dataframe(edges_df, use_container_width=True, height=360)
                else:
                    if use_nvl:
                        # ---------- NVL via CDN (labels + severity on node) ----------
                        nvl_nodes = [
                            {
                                "id":    str(r["id"]),
                                "label": f"{r['label']}\\n({str(r['severity']).upper()})",
                                "color": r["color"],
                            }
                            for _, r in nodes_df.iterrows()
                        ]
                        nvl_rels = [
                            {
                                "id":   f"{row.get('type','EDGE')}-{i}",
                                "from": str(row["source"]),
                                "to":   str(row["target"]),
                                "type": row.get("type", ""),
                            }
                            for i, row in edges_df.iterrows()
                        ]

                        html(f"""
                          <div id="nvl-container" style="height:650px;border:1px solid #e5e7eb;border-radius:8px"></div>
                          <script type="module">
                            import {{ NVL }} from 'https://esm.sh/@neo4j-nvl/base';
                            const nodes = {json.dumps(nvl_nodes)};
                            const relationships = {json.dumps(nvl_rels)};
                            const el = document.getElementById('nvl-container');
                            const nvl = new NVL(el, nodes, relationships, {{
                              physics: true,
                              showLabels: true
                            }});
                            setTimeout(() => {{
                              if (nvl && typeof nvl.zoomToFit === 'function') nvl.zoomToFit();
                            }}, 200);
                          </script>
                        """, height=680)
                        st.caption("Rendered with NVL (CDN).")
                    else:
                        # ---------- agraph Pretty Mode ----------
                        st.markdown("**Display options**")
                        colA, colB, colC = st.columns([1,1,1])
                        with colA: show_edge_labels = st.checkbox("Show edge labels", value=True)
                        with colB: physics_on      = st.checkbox("Physics layout", value=True)
                        with colC: scale_by_degree = st.checkbox("Scale nodes by degree", value=True)

                        # Degree map
                        deg = {nid: 0 for nid in nodes_df["id"]}
                        for _, e in edges_df.iterrows():
                            deg[e["source"]] = deg.get(e["source"], 0) + 1
                            deg[e["target"]] = deg.get(e["target"], 0) + 1

                        def size_for(nid):
                            if not scale_by_degree: return 18
                            d = deg.get(nid, 0)
                            return max(16, min(40, 14 + 3*d))

                        a_nodes = [
                            Node(
                                id=row["id"],
                                label=row["label"],
                                size=size_for(row["id"]),
                                color=row["color"],
                                title=f"{row['label']}\nSeverity: {row['severity']}\nDegree: {deg.get(row['id'],0)}",
                                shape="dot",
                                font={"size": 14, "multi": "html"},
                                borderWidth=1
                            )
                            for _, row in nodes_df.iterrows()
                        ]
                        a_edges = [
                            Edge(
                                source=e["source"], target=e["target"],
                                label=(e["type"] if show_edge_labels else ""),
                                title=e["type"], arrows="to", smooth=True, font={"size": 10}
                            )
                            for _, e in edges_df.iterrows()
                        ]
                        config = Config(
                            height=650, width=1200, directed=True, physics=physics_on, hierarchical=False,
                            options={
                                "interaction": {"hover": True, "navigationButtons": True, "keyboard": True, "multiselect": True, "tooltipDelay": 120},
                                "nodes": {"shadow": True, "font": {"size": 14}},
                                "edges": {"arrows": {"to": {"enabled": True}}, "smooth": {"enabled": True, "type": "dynamic"}, "color": {"opacity": 0.7}, "width": 1.2},
                                "physics": {"enabled": physics_on, "stabilization": {"enabled": True, "iterations": 250}, "solver": "forceAtlas2Based",
                                            "forceAtlas2Based": {"gravitationalConstant": -45, "springLength": 90, "springConstant": 0.08}}
                            }
                        )
                        agraph(nodes=a_nodes, edges=a_edges, config=config)
                        st.caption("Rendered with agraph (pretty mode).")

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
