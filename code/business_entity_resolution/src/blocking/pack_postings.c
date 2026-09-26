/* MIT license. Compact rowid transfer for bounded SQLite postings queries. */
#include <sqlite3ext.h>
SQLITE_EXTENSION_INIT1
#include <stdint.h>
#include <string.h>

typedef struct { uint32_t *items; sqlite3_uint64 count, capacity; int failed; } Buffer;

static void step(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    (void)argc;
    Buffer *b = sqlite3_aggregate_context(ctx, sizeof(Buffer));
    if (!b) { sqlite3_result_error_nomem(ctx); return; }
    if (b->failed) return;
    sqlite3_int64 value = sqlite3_value_int64(argv[0]);
    if (sqlite3_value_type(argv[0]) != SQLITE_INTEGER || value <= 0 || value > INT32_MAX) {
        b->failed = 1;
        sqlite3_result_error(ctx, "Posting rowid outside positive int32 range", -1);
        return;
    }
    if (b->count == b->capacity) {
        sqlite3_uint64 capacity = b->capacity ? b->capacity * 2 : 256;
        uint32_t *next = sqlite3_realloc64(b->items, capacity * sizeof(uint32_t));
        if (!next) { b->failed = 1; sqlite3_result_error_nomem(ctx); return; }
        b->items = next; b->capacity = capacity;
    }
    b->items[b->count++] = (uint32_t)value;
}

static void finish(sqlite3_context *ctx) {
    Buffer *b = sqlite3_aggregate_context(ctx, 0);
    if (!b) { sqlite3_result_zeroblob(ctx, 0); return; }
    if (b->failed) { sqlite3_free(b->items); b->items = 0; return; }
    sqlite3_result_blob64(ctx, b->items, b->count * sizeof(uint32_t), sqlite3_free);
    b->items = 0;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
int sqlite3_erpostings_init(sqlite3 *db, char **error, const sqlite3_api_routines *api) {
    (void)error;
    SQLITE_EXTENSION_INIT2(api);
    return sqlite3_create_function(db, "er_pack", 1, SQLITE_UTF8, 0, 0, step, finish);
}

/* Sum bounded posting lists with a compact per-query hash table. */
int64_t er_reduce(const int32_t **lists, const int64_t *lengths,
    const int32_t *fields, const double *weights, int64_t nlists,
    int32_t *ids, double *values, uint32_t *slots, uint64_t capacity, int64_t records) {
    int64_t count = 0;
    if (!capacity || (capacity & (capacity - 1))) return -1;
    for (int64_t j = 0; j < nlists; j++) {
        if (fields[j] < 0 || fields[j] >= 4) return -1;
        for (int64_t k = 0; k < lengths[j]; k++) {
            int32_t id = lists[j][k];
            if (id <= 0 || id > records) return -1;
            uint64_t slot = ((uint64_t)(uint32_t)id * 2654435761u) & (capacity - 1);
            while (slots[slot] && ids[slots[slot] - 1] != id) slot = (slot + 1) & (capacity - 1);
            if (!slots[slot]) {
                ids[count] = id;
                memset(values + count * 4, 0, 4 * sizeof(double));
                slots[slot] = (uint32_t)++count;
            }
            values[((int64_t)slots[slot] - 1) * 4 + fields[j]] += weights[j];
        }
    }
    return count;
}
