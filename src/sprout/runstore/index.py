"""Ordered tool-prefix DAG and occurrence-specific snapshot/native pairs."""

from __future__ import annotations

from uuid import uuid4

from sprout.harness.agents.mini_swe.history import NativePoint, valid_prefix
from sprout.runstore.specs import canonical, digest
from sprout.runstore.store import Conflict, Store


def normalized_call(name: str, arguments: dict) -> dict:
    if not isinstance(name, str) or not isinstance(arguments, dict):
        raise ValueError("Tool prefix requires tool name and JSON object arguments")
    return {"name": name, "arguments": arguments}


class Index:
    def __init__(self, store: Store, snapshot_valid=None) -> None:
        self.store = store
        self.snapshot_valid = snapshot_valid

    def project(self, job_id: str, token: str, scope: dict, events: list[dict],
                points: list[NativePoint] = ()) -> None:
        calls = [event for event in events if event.get("type") == "tool.started"]
        results = {}
        for event in events:
            if event.get("type") == "tool.finished":
                results.setdefault(event.get("call_id"), event)
        scope_hash = digest(scope)
        node = scope_hash
        nodes = {}
        with self.store.transaction() as cursor:
            job = self.store._fence(cursor, job_id, token)
            for depth, event in enumerate(calls, 1):
                call = normalized_call(event["name"], event.get("args", {}))
                parent, node = node, digest([node, call])
                nodes[depth] = node
                cursor.execute("""INSERT INTO rs_prefix_nodes(node,scope,parent,depth,call)
                    VALUES(%s,%s,%s,%s,%s::jsonb) ON CONFLICT DO NOTHING""",
                               (node, scope_hash, parent, depth, canonical(call)))
                response = results.get(event["call_id"])
                cursor.execute("""INSERT INTO rs_tools(attempt_id,depth,call_id,call,response,prefix_node)
                    VALUES(%s,%s,%s,%s::jsonb,%s::jsonb,%s)
                    ON CONFLICT(attempt_id,depth) DO UPDATE SET response=COALESCE(rs_tools.response,EXCLUDED.response)
                    WHERE rs_tools.call_id=EXCLUDED.call_id AND rs_tools.call=EXCLUDED.call
                    AND rs_tools.prefix_node=EXCLUDED.prefix_node""",
                               (job["active_attempt"], depth, event["call_id"], canonical(call),
                                canonical(response) if response is not None else None, node))
                if cursor.rowcount != 1:
                    raise Conflict("Indexed tool prefix changed")
            for point in points:
                if point.tool_depth not in nodes:
                    raise ValueError("Recovery point extends beyond recorded tools")
                cursor.execute("""INSERT INTO rs_recoveries
                    (id,attempt_id,message_step,tool_depth,prefix_node,snapshot_id,native)
                    VALUES(%s,%s,%s,%s,%s,%s,%s::jsonb)
                    ON CONFLICT(attempt_id,message_step) DO UPDATE SET valid=true,invalid_reason=NULL
                    WHERE rs_recoveries.native=EXCLUDED.native
                    AND rs_recoveries.snapshot_id=EXCLUDED.snapshot_id
                    AND rs_recoveries.prefix_node=EXCLUDED.prefix_node""",
                               (uuid4().hex, job["active_attempt"], point.message_step,
                                point.tool_depth, nodes[point.tool_depth], point.snapshot_id,
                                canonical(point.native)))
                if cursor.rowcount != 1:
                    raise Conflict("Recovery point changed after publication")
            cursor.execute("""UPDATE rs_recoveries SET valid=false,invalid_reason='boundary_no_longer_proven'
                WHERE attempt_id=%s AND NOT(message_step=ANY(%s))""",
                           (job["active_attempt"], [point.message_step for point in points]))

    def tools(self, attempt_id: str, after: int = 0, limit: int = 1000) -> list[dict]:
        with self.store.transaction() as cursor:
            cursor.execute("""SELECT * FROM rs_tools WHERE attempt_id=%s AND depth>%s
                ORDER BY depth LIMIT %s""", (attempt_id, after, min(10000, max(1, limit))))
            return [dict(row) for row in cursor.fetchall()]

    def points(self, attempt_id: str) -> list[dict]:
        with self.store.transaction() as cursor:
            cursor.execute("SELECT * FROM rs_recoveries WHERE attempt_id=%s ORDER BY message_step", (attempt_id,))
            return [dict(row) for row in cursor.fetchall()]

    def get_point(self, point_id: str) -> dict:
        with self.store.transaction() as cursor:
            cursor.execute("""SELECT point.*,attempt.job_id FROM rs_recoveries point
                JOIN rs_attempts attempt ON point.attempt_id=attempt.id WHERE point.id=%s""", (point_id,))
            row = cursor.fetchone()
            if row is None:
                raise KeyError(point_id)
            return dict(row)

    def valid(self, point: dict) -> bool:
        return (point["valid"] and valid_prefix(point["native"])
                and self.snapshot_valid is not None and self.snapshot_valid(point))
