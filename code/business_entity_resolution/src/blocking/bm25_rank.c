/* Corpus-frequency acceleration for SQLite FTS5 BM25.
 * The scoring arithmetic follows SQLite 3.50.4's public-domain fts5_aux.c.
 * https://github.com/sqlite/sqlite/blob/version-3.50.4/ext/fts5/fts5_aux.c
 * Remaining code is provided under the project MIT license.
 * Arguments are exact per-phrase document counts; -1 uses SQLite's own count.
 * This changes no MATCH expression, scoring formula, column weight, or limit.
 */
#include <sqlite3ext.h>
SQLITE_EXTENSION_INIT1
#include <math.h>
#include <string.h>

typedef struct RankData {
  int nPhrase;
  double avgdl;
  double *idf;
  double *freq;
} RankData;

static int count_cb(const Fts5ExtensionApi *api, Fts5Context *fts, void *data) {
  (void)api; (void)fts;
  (*(sqlite3_int64*)data)++;
  return SQLITE_OK;
}

static void rank_function(const Fts5ExtensionApi *api, Fts5Context *fts,
    sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  const double k1 = 1.2, b = 0.75;
  int rc = SQLITE_OK, i, instances = 0, tokens = 0;
  double score = 0.0, D;
  RankData *data = (RankData*)api->xGetAuxdata(fts, 0);
  if (!data) {
    int phrases = api->xPhraseCount(fts);
    sqlite3_int64 rows = 0, total = 0;
    if (argc != phrases) {
      sqlite3_result_error(ctx, "er_bm25 phrase/count mismatch", -1);
      return;
    }
    data = sqlite3_malloc64(sizeof(RankData) + 2 * phrases * sizeof(double));
    if (!data) { sqlite3_result_error_nomem(ctx); return; }
    memset(data, 0, sizeof(RankData) + 2 * phrases * sizeof(double));
    data->nPhrase = phrases;
    data->idf = (double*)(data + 1);
    data->freq = data->idf + phrases;
    rc = api->xRowCount(fts, &rows);
    if (rc == SQLITE_OK) rc = api->xColumnTotalSize(fts, -1, &total);
    if (rc == SQLITE_OK) data->avgdl = (double)total / (double)rows;
    for (i = 0; i < phrases && rc == SQLITE_OK; ++i) {
      sqlite3_int64 hits = sqlite3_value_int64(argv[i]);
      if (hits < 0) {
        hits = 0;
        rc = api->xQueryPhrase(fts, i, &hits, count_cb);
      }
      if (rc == SQLITE_OK) {
        double idf = log((rows - hits + 0.5) / (hits + 0.5));
        data->idf[i] = idf <= 0.0 ? 1e-6 : idf;
      }
    }
    if (rc == SQLITE_OK) rc = api->xSetAuxdata(fts, data, sqlite3_free);
    else sqlite3_free(data);
    if (rc != SQLITE_OK) { sqlite3_result_error_code(ctx, rc); return; }
  }
  memset(data->freq, 0, sizeof(double) * data->nPhrase);
  rc = api->xInstCount(fts, &instances);
  for (i = 0; i < instances && rc == SQLITE_OK; ++i) {
    int phrase, column, offset;
    rc = api->xInst(fts, i, &phrase, &column, &offset);
    if (rc == SQLITE_OK) data->freq[phrase] += 1.0;
  }
  if (rc == SQLITE_OK) rc = api->xColumnSize(fts, -1, &tokens);
  D = (double)tokens;
  if (rc == SQLITE_OK) {
    for (i = 0; i < data->nPhrase; ++i) {
      score += data->idf[i] * ((data->freq[i] * (k1 + 1.0)) /
        (data->freq[i] + k1 * (1 - b + b * D / data->avgdl)));
    }
    sqlite3_result_double(ctx, -1.0 * score);
  } else sqlite3_result_error_code(ctx, rc);
}

#ifdef _WIN32
__declspec(dllexport)
#endif
int sqlite3_erbm25_init(sqlite3 *db, char **error, const sqlite3_api_routines *routines) {
  fts5_api *api = 0;
  sqlite3_stmt *statement = 0;
  int rc;
  (void)error;
  SQLITE_EXTENSION_INIT2(routines);
  rc = sqlite3_prepare_v2(db, "SELECT fts5(?1)", -1, &statement, 0);
  if (rc == SQLITE_OK) rc = sqlite3_bind_pointer(statement, 1, &api, "fts5_api_ptr", 0);
  if (rc == SQLITE_OK) { sqlite3_step(statement); rc = api ? SQLITE_OK : SQLITE_ERROR; }
  sqlite3_finalize(statement);
  if (rc != SQLITE_OK) return rc;
  return api->xCreateFunction(api, "er_bm25", 0, rank_function, 0);
}
