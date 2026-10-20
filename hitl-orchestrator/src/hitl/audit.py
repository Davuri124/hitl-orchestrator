"""Audit trail: every state transition + the policy's action distribution (with ensemble interval) goes to SQLite;
optionally mirrored to a Kafka topic (pip install kafka-python, pass kafka=("host:9092","hitl-audit"))."""
import json, sqlite3, time


class AuditLog:
    def __init__(self, path, run_id, kafka=None):
        self.db = sqlite3.connect(path); self.run_id = run_id; self.prod = None
        self.db.execute("""CREATE TABLE IF NOT EXISTS transitions(
            ts REAL, run_id TEXT, qid TEXT, step INT, stage TEXT, action TEXT,
            p_human REAL, p_human_lo REAL, p_human_hi REAL, budget_used INT, budget_cap INT,
            reward REAL, cost REAL, state TEXT)""")
        if kafka:
            from kafka import KafkaProducer
            self.prod = (KafkaProducer(bootstrap_servers=kafka[0], value_serializer=lambda v: json.dumps(v).encode()), kafka[1])

    def log(self, **r):
        r = {"ts": time.time(), "run_id": self.run_id, **r}
        self.db.execute("INSERT INTO transitions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (r["ts"], r["run_id"], r["qid"], r["step"], r["stage"], r["action"], r.get("p_human"),
                         r.get("p_human_lo"), r.get("p_human_hi"), r["budget_used"], r["budget_cap"],
                         r["reward"], r["cost"], json.dumps(r["state"])))
        if self.prod: self.prod[0].send(self.prod[1], r)

    def close(self):
        self.db.commit(); self.db.close()
        if self.prod: self.prod[0].flush()
