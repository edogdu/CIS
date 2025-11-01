from typing import List, Dict, Any
from typing import Tuple
from factories.data import DataFactory

class AnomalyRepository:
    async def save_anomalies(self, system_id, model_type, duration, anomalies) -> None:
        """Save detected anomalies to the database."""
        # Implementation to save anomalies to the database
        pool = await DataFactory.get_pg_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                for a in anomalies:
                    # bucket is last part of snapshot_id after last underscore
                    bucket = a['snapshot_id'].split('_')[-1]
                    await cur.execute("""
                        INSERT INTO anomaly_alerts(bucket, duration, system_id, snapshot_id, model_type, anomaly_score, src_graph_id, dst_graph_id)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (bucket, duration, system_id, a['snapshot_id'], model_type, a['anomaly_score'], a['src_graph_id'], a['dst_graph_id']))

    async def get_anomaly_results(self, system_id, duration, start_time, end_time):
        """Retrieve anomaly results from the database."""
        # Implementation to retrieve anomalies from the database
        if duration not in [30, 16, 10]:
            raise ValueError("Unsupported duration. Supported durations are 30, 16, and 10 seconds.")
        pool = await DataFactory.get_pg_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                table_name = f"scada_agg_{duration}s"
                await cur.execute(f"""
                                  WITH cte AS (
    SELECT DISTINCT bucket, system_id
    FROM {table_name}
    WHERE system_id = %s
      AND bucket >= %s
        AND bucket <= %s
)
SELECT
    case when SUM(sa.num_attacks) = 0 and aa.snapshot_id IS NULL THEN 1 ELSE 0 END AS is_true_negative,
    case when SUM(sa.num_attacks) > 0 and aa.snapshot_id IS NOT NULL THEN 1 ELSE 0 END AS is_true_positive,
    case when SUM(sa.num_attacks) = 0 and aa.snapshot_id IS NOT NULL THEN 1 ELSE 0 END AS is_false_positive,
    case when SUM(sa.num_attacks) > 0 and aa.snapshot_id IS NULL THEN 1 ELSE 0 END AS is_false_negative,
    cte.bucket,
    cte.system_id,
    30 AS duration,
    SUM(sa.num_attacks) AS num_attacks,
    aa.snapshot_id, AVG(aa.anomaly_score) AS anomaly_score
FROM cte
INNER JOIN scada_agg_30s sa
    ON cte.bucket = sa.bucket
    AND cte.system_id = sa.system_id
LEFT JOIN anomaly_alerts aa
    ON cte.bucket = aa.bucket
    AND cte.system_id = aa.system_id
    AND aa.duration = 30
GROUP BY cte.bucket, cte.system_id, aa.snapshot_id
ORDER BY cte.bucket;
                                  """, (system_id, start_time, end_time))
                results = await cur.fetchall()
                return results
