#!/usr/bin/env python3
"""Export quiz sessions + recordings from Supabase for admin / AI supervision.

Reads directly from Postgres (service-side), never via the anon key.

Usage:
    export AIQ_DATABASE_URL='postgresql://postgres:...@db.<ref>.supabase.co:5432/postgres?sslmode=require'
    python tools/export_recordings.py --out recordings.jsonl
    python tools/export_recordings.py --user alice --since 2026-09-05 --out alice.jsonl

Each output line is one session:
    {session_id, username, pack, score_correct, score_total, started_at,
     last_seen, meta, events: [...]}
Events from all chunks are merged in seq order, so the file is directly
loadable by the quiz replay player (wrap as {schema_version:1, meta, events}
if a standalone .json replay file is needed — see --replay-files).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="-", help="output JSONL path ('-' = stdout)")
    parser.add_argument("--user", help="filter by username")
    parser.add_argument("--since", help="only sessions with last_seen >= this ISO date")
    parser.add_argument(
        "--replay-files",
        metavar="DIR",
        help="also write one replayable .json per session into DIR",
    )
    args = parser.parse_args()

    dsn = os.environ.get("AIQ_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not dsn:
        print("error: set AIQ_DATABASE_URL (or DATABASE_URL)", file=sys.stderr)
        return 2

    try:
        import psycopg
    except ImportError:
        print("error: pip install 'psycopg[binary]'", file=sys.stderr)
        return 2

    conn = psycopg.connect(dsn)

    where = []
    params: list = []
    if args.user:
        where.append("u.username = %s")
        params.append(args.user)
    if args.since:
        where.append("s.last_seen >= %s")
        params.append(args.since)
    where_sql = ("where " + " and ".join(where)) if where else ""

    sessions = conn.execute(
        f"""
        select s.session_id, u.username, s.pack, s.score_correct, s.score_total,
               s.started_at, s.last_seen, s.meta
        from quiz_sessions s
        join quiz_users u on u.id = s.user_id
        {where_sql}
        order by s.started_at
        """,
        params,
    ).fetchall()

    out = open(args.out, "w") if args.out != "-" else sys.stdout
    if args.replay_files:
        os.makedirs(args.replay_files, exist_ok=True)

    n = 0
    for session_id, username, pack, sc, st, started, last_seen, meta in sessions:
        chunks = conn.execute(
            "select events from recording_chunks where session_id=%s order by seq",
            (session_id,),
        ).fetchall()
        events = [ev for (chunk,) in chunks for ev in chunk]
        row = {
            "session_id": session_id,
            "username": username,
            "pack": pack,
            "score_correct": sc,
            "score_total": st,
            "started_at": started.astimezone(timezone.utc).isoformat() if started else None,
            "last_seen": last_seen.astimezone(timezone.utc).isoformat() if last_seen else None,
            "meta": meta,
            "events": events,
        }
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        if args.replay_files:
            replay = {
                "schema_version": 1,
                "meta": {
                    "session_id": session_id,
                    "username": username,
                    "pack": pack,
                    "started_at": row["started_at"],
                    **(meta if isinstance(meta, dict) else {}),
                },
                "events": events,
            }
            path = os.path.join(args.replay_files, f"{username}_{session_id}.json")
            with open(path, "w") as fh:
                json.dump(replay, fh, ensure_ascii=False)
        n += 1

    if args.out != "-":
        out.close()
    print(f"exported {n} session(s)", file=sys.stderr)
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
