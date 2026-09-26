"""Same FTS5 MATCH and BM25 arithmetic with already-materialized frequencies."""

import re
from pathlib import Path

from ..preprocessing.normalize import accent_fold

LEXER = re.compile(r'"((?:[^"]|"")*)"|([a-z_]+)\s*:|(\()|(\))')


def phrase_counts(expression, index):
    column, pending, stack, counts = None, None, [], []
    for match in LEXER.finditer(expression):
        quoted, prefix, opening, closing = match.groups()
        if prefix:
            pending = prefix
        elif opening:
            stack.append(column)
            column = pending or column
            pending = None
        elif closing:
            column = stack.pop() if stack else None
        elif quoted is not None:
            term = accent_fold(quoted.replace('""', '"'))
            # Multi-token and non-Latin phrases retain SQLite's exact phrase
            # counting. All-column counts cannot be obtained by summing fields.
            count = index.df(column, term) if column in ("name", "address", "digits") and re.fullmatch("[a-z0-9]+", term) else -1
            counts.append(count)
    return counts


class FastRankSearch:
    def __init__(self, index, extension):
        self.index = index
        index.connection.enable_load_extension(True)
        try:
            index.connection.load_extension(str(Path(extension).resolve()), entrypoint="sqlite3_erbm25_init")
        finally:
            index.connection.enable_load_extension(False)

    def __call__(self, expression, *, character=False, country=None):
        if character:
            return self.original(expression, character=True, country=country)
        counts = phrase_counts(expression, self.index)
        if not counts:
            return self.original(expression, country=country)
        ranking = "er_bm25(" + ",".join(map(str, counts)) + ")"
        connection, limit = self.index.connection, self.index.config.path_top_k
        if country:
            return [row[0] for row in connection.execute(
                "SELECT words.rowid FROM words JOIN records ON records.rowid=words.rowid "
                "WHERE words MATCH ? AND (records.country=? OR records.country='') "
                "AND words.rank MATCH ? ORDER BY words.rank LIMIT ?",
                (expression, country, ranking, limit))]
        return [row[0] for row in connection.execute(
            "SELECT rowid FROM words WHERE words MATCH ? AND rank MATCH ? ORDER BY rank LIMIT ?",
            (expression, ranking, limit))]


def install_fast_rank(index, extension):
    search = FastRankSearch(index, extension)
    search.original = index.search
    index.search = search
    return search
