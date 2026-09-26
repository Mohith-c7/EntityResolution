"""Materialized corpus-local frequencies for bounded B-tree term lookup."""

import json
import os
import sqlite3
from pathlib import Path


def frequency_path(base_path):
    return Path(str(base_path)+".frequencies.sqlite")


def ensure_frequency_index(base_path):
    base_path=Path(base_path).resolve()
    output=frequency_path(base_path)
    with sqlite3.connect(f"{base_path.as_uri()}?mode=ro",uri=True) as base:
        metadata=json.loads(base.execute("SELECT value FROM metadata WHERE key='build'").fetchone()[0])
        expected=metadata["fingerprint"]
        if output.exists():
            with sqlite3.connect(f"{output.as_uri()}?mode=ro",uri=True) as saved:
                actual=json.loads(saved.execute("SELECT value FROM metadata").fetchone()[0])
            if expected!=actual:
                raise ValueError(f"Frequency input fingerprint differs: {output}")
            return output
        temporary=output.with_name(output.name+f".building-{os.getpid()}")
        if temporary.exists():
            raise FileExistsError(temporary)
        connection=sqlite3.connect(temporary)
        try:
            connection.executescript("""
                PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
                CREATE TABLE frequencies(term TEXT,col TEXT,doc INTEGER,PRIMARY KEY(term,col)) WITHOUT ROWID;
                CREATE TABLE metadata(value TEXT);
            """)
            cursor=base.execute("SELECT term,col,doc FROM word_vocab")
            while rows:=cursor.fetchmany(20_000):
                connection.executemany("INSERT INTO frequencies VALUES(?,?,?)",rows)
            connection.execute("INSERT INTO metadata VALUES(?)",(json.dumps(expected),))
            connection.commit()
            connection.close()
            temporary.replace(output)
        except BaseException:
            connection.close()
            raise
    return output
