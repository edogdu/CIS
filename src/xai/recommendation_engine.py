# recomendation_engine.py
# map target class to MITRE ATT&CK technique

import requests
import re
from bs4 import BeautifulSoup
from functools import lru_cache

@lru_cache(maxsize=128)
def fetch_mitre_page(tid):
    url = f"https://attack.mitre.org/techniques/{tid}/"
    return requests.get(url, timeout=10).text


class RecommendationEngine:

    CLASS_TO_MITRE = {
        0: ["T0809"],   # normal
        1: ["T0830"],   # MITM
        2: ["T0826"],   # Physical Fault
        3: ["T0814"],   # DoS
        4: ["T0846"],   # Scan
    }

    def get_recommendations(self, anomaly_class, data, root_cause_chain, causal_paths):

        technique_ids = self.CLASS_TO_MITRE.get(anomaly_class, [])
        results = []

        for tid in technique_ids:

            # fetch per technique
            html = fetch_mitre_page(tid)
            soup = BeautifulSoup(html, "html.parser")

            mitigations = self._extract_section(soup, "Mitigations")
            detections = self._extract_section(soup, "Detection Strategy")

            contextual = []

            for hop in root_cause_chain:
                ntype = hop.get("node_type")
                oid = hop.get("original_id")

                if oid is None:
                    continue

                ctx = data.context_lookup.get(ntype, {}).get(oid, {})
                ctx = self._normalize_context(ntype, ctx)

                contextual.append({
                    "node": f"{ntype}:{oid}",
                    "context": ctx,
                    "action": self._recommend_action(ntype, ctx, tid)
                })

            results.append({
                "technique_id": tid,
                "mitigations": mitigations,
                "detections": detections,
                "contextual_recommendations": contextual
            })

        return results

    def _extract_section(self, soup, header_text):
        header = soup.find("h2", string=re.compile(header_text, re.I))
        if not header:
            return []

        table = header.find_next("table")
        if not table:
            return []

        rows = table.find_all("tr")

        results = []

        for row in rows[1:]:  # skip header row
            cols = row.find_all("td")

            if len(cols) < 2:
                continue

            item_id = cols[0].get_text(strip=True)
            name = cols[1].get_text(strip=True)
            desc = cols[2].get_text(" ", strip=True) if len(cols) > 2 else ""

            results.append({
                "id": item_id,
                "name": name,
                "description": desc
            })

        return results

    def _recommend_action(self, ntype, ctx, tid):

        sensor = (
            ctx.get("sensor_tag")
            or ctx.get("sensor_id")
            or ctx.get("asset_id")
            or "unknown sensor"
        )

        if ntype == "Connection":
            return (
                f"Inspect network flow for anomalies related to {tid}. "
                f"Review source {ctx.get('source')} and destination {ctx.get('destination')}."
            )

        if ntype == "Measurement":
            return (
                f"Validate sensor {sensor} integrity and check for spoofing attempts associated with {tid}."
            )

        if ntype == "FlowSensors":
            return (
                f"Check flow sensor {sensor} for manipulation or unexpected spikes."
            )

        if ntype == "Valves":
            return (
                f"Review valve {ctx.get('valve_id', 'unknown valve')} command history for unauthorized changes."
            )

        return f"Review node for anomalies related to {tid}."


    # additional preprocessing
    def _normalize_context(self, ntype, ctx):
        if ntype == "Measurement":
            ctx["sensor_tag"] = ctx.get("asset_id")
        elif ntype == "FlowSensors":
            ctx["sensor_tag"] = ctx.get("asset_id") or ctx.get("flow_id")
        return ctx
