#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ACTIONS 3
#define LOOKBACK 12
#define MAX_ROWS 720
#define FEE 0.001
#define GAMMA 1.0
#define LINE_CAP 1024

typedef struct {
    char timestamp[32];
    double open;
    double close;
    double high;
    double low;
    double volume;
} Candle;

typedef struct {
    Candle *items;
    int count;
    int capacity;
} CandleList;

static void executable_dir(const char *argv0, char *out, size_t out_size) {
    const char *slash = strrchr(argv0, '/');
    size_t len;

    if (!slash) {
        snprintf(out, out_size, ".");
        return;
    }

    len = (size_t)(slash - argv0);
    if (len >= out_size) len = out_size - 1;
    memcpy(out, argv0, len);
    out[len] = '\0';
}

static void default_path(char *out, size_t out_size, const char *exe_dir, const char *relative) {
    snprintf(out, out_size, "%s/../../%s", exe_dir, relative);
}

static int ieq(const char *a, const char *b) {
    while (*a && *b) {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) return 0;
        a++;
        b++;
    }
    return *a == '\0' && *b == '\0';
}

static char *trim(char *s) {
    char *end;
    while (*s && isspace((unsigned char)*s)) s++;
    end = s + strlen(s);
    while (end > s && isspace((unsigned char)end[-1])) end--;
    *end = '\0';
    return s;
}

static void push_candle(CandleList *list, Candle candle) {
    if (list->count == list->capacity) {
        int next_capacity = list->capacity ? list->capacity * 2 : 1024;
        Candle *next = (Candle *)realloc(list->items, (size_t)next_capacity * sizeof(Candle));
        if (!next) {
            fprintf(stderr, "out of memory\n");
            exit(1);
        }
        list->items = next;
        list->capacity = next_capacity;
    }
    list->items[list->count++] = candle;
}

static int split_csv(char *line, char **cols, int max_cols) {
    int n = 0;
    char *p = line;
    while (n < max_cols) {
        cols[n++] = trim(p);
        p = strchr(p, ',');
        if (!p) break;
        *p = '\0';
        p++;
    }
    return n;
}

static int find_column(char **cols, int n, const char *name) {
    int i;
    for (i = 0; i < n; i++) {
        if (ieq(trim(cols[i]), name)) return i;
    }
    return -1;
}

static int read_csv(const char *path, CandleList *list) {
    FILE *f = fopen(path, "r");
    char line[LINE_CAP];
    char *cols[16];
    int ncols;
    int ts;
    int open;
    int close;
    int high;
    int low;
    int volume;

    if (!f) {
        fprintf(stderr, "failed to open %s\n", path);
        return 0;
    }

    if (!fgets(line, sizeof(line), f)) {
        fclose(f);
        return 0;
    }

    ncols = split_csv(line, cols, 16);
    ts = find_column(cols, ncols, "Timestamp");
    open = find_column(cols, ncols, "Open");
    close = find_column(cols, ncols, "Close");
    high = find_column(cols, ncols, "High");
    low = find_column(cols, ncols, "Low");
    volume = find_column(cols, ncols, "Volume");

    if (open < 0 || close < 0 || high < 0 || low < 0 || volume < 0) {
        fprintf(stderr, "CSV needs Timestamp, Open, Close, High, Low, Volume columns\n");
        fclose(f);
        return 0;
    }

    while (fgets(line, sizeof(line), f)) {
        Candle c;
        char copy[LINE_CAP];
        strncpy(copy, line, sizeof(copy) - 1);
        copy[sizeof(copy) - 1] = '\0';
        ncols = split_csv(copy, cols, 16);
        if (ncols <= volume) continue;

        memset(&c, 0, sizeof(c));
        if (ts >= 0 && ts < ncols) {
            strncpy(c.timestamp, cols[ts], sizeof(c.timestamp) - 1);
        }
        c.open = atof(cols[open]);
        c.close = atof(cols[close]);
        c.high = atof(cols[high]);
        c.low = atof(cols[low]);
        c.volume = atof(cols[volume]);
        if (c.close > 0.0) push_candle(list, c);
    }

    fclose(f);
    return list->count > LOOKBACK + 1;
}

static Candle *latest_window(CandleList *list, int max_rows, int *out_count) {
    int count = list->count < max_rows ? list->count : max_rows;
    *out_count = count;
    return list->items + (list->count - count);
}

static int state_index(int time_index, int position) {
    return time_index * 2 + position;
}

static int next_position(int position, int action) {
    if (action == 1) return 1;
    if (action == 2) return 0;
    return position;
}

static void train(Candle *candles, int n, double *weights) {
    double *v = (double *)calloc((size_t)n * 2, sizeof(double));
    int i;
    int position;
    int action;

    if (!v) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }

    for (i = n - 2; i >= LOOKBACK; i--) {
        double price_return = candles[i + 1].close / candles[i].close - 1.0;
        for (position = 0; position < 2; position++) {
            double best = -1e30;
            for (action = 0; action < ACTIONS; action++) {
                int next_pos = next_position(position, action);
                double cost = next_pos == position ? 0.0 : FEE;
                double reward = next_pos * price_return - cost;
                double q = reward + GAMMA * v[state_index(i + 1, next_pos)];
                weights[action * n * 2 + state_index(i, position)] = q;
                if (q > best) best = q;
            }
            v[state_index(i, position)] = best;
        }
    }

    free(v);
}

static void write_weights(FILE *f, double *weights, int state_size) {
    int action;
    int s;
    fputs("[", f);
    for (action = 0; action < ACTIONS; action++) {
        if (action > 0) fputs(",", f);
        fputs("[", f);
        for (s = 0; s < state_size; s++) {
            if (s > 0) fputs(",", f);
            fprintf(f, "%.8f", weights[action * state_size + s]);
        }
        fputs("]", f);
    }
    fputs("]", f);
}

static void write_closes(FILE *f, Candle *candles, int n) {
    int i;
    fputs("[", f);
    for (i = 0; i < n; i++) {
        if (i > 0) fputs(",", f);
        fprintf(f, "%.2f", candles[i].close);
    }
    fputs("]", f);
}

static int write_js(const char *path, Candle *candles, int n, double *weights) {
    FILE *f = fopen(path, "w");
    if (!f) return 0;
    fputs("window.tinyfinBTCTradingModel = {\n", f);
    fputs("  format: \"tinyfin-js-btc-oracle-v1\",\n", f);
    fprintf(f, "  rows: %d,\n  lookback: %d,\n  stateSize: %d,\n  actionSize: %d,\n", n, LOOKBACK, n * 2, ACTIONS);
    fprintf(f, "  fee: %.6f,\n  gamma: %.4f,\n", FEE, GAMMA);
    fputs("  closes: ", f);
    write_closes(f, candles, n);
    fputs(",\n  weights: ", f);
    write_weights(f, weights, n * 2);
    fputs("\n};\n", f);
    fclose(f);
    return 1;
}

static int write_json(const char *path, Candle *candles, int n, double *weights) {
    FILE *f = fopen(path, "w");
    if (!f) return 0;
    fprintf(f, "{\n  \"format\": \"tinyfin-js-btc-oracle-v1\",\n");
    fprintf(f, "  \"rows\": %d,\n  \"lookback\": %d,\n  \"stateSize\": %d,\n  \"actionSize\": %d,\n", n, LOOKBACK, n * 2, ACTIONS);
    fprintf(f, "  \"fee\": %.6f,\n  \"gamma\": %.4f,\n", FEE, GAMMA);
    fputs("  \"closes\": ", f);
    write_closes(f, candles, n);
    fputs(",\n  \"weights\": ", f);
    write_weights(f, weights, n * 2);
    fputs("\n}\n", f);
    fclose(f);
    return 1;
}

static void usage(const char *argv0) {
    fprintf(stderr, "Usage: %s [--csv path] [--js path] [--json path] [--max-rows n]\n", argv0);
}

int main(int argc, char **argv) {
    char exe_dir[1024];
    char default_csv[1200];
    char default_js[1200];
    char default_json[1200];
    const char *csv_path;
    const char *js_path;
    const char *json_path;
    int max_rows = MAX_ROWS;
    CandleList list;
    Candle *candles;
    int n;
    double *weights;
    int i;

    executable_dir(argv[0], exe_dir, sizeof(exe_dir));
    default_path(default_csv, sizeof(default_csv), exe_dir, "dataset/btc_4h.csv");
    default_path(default_js, sizeof(default_js), exe_dir, "examples/btc-trading-model.js");
    default_path(default_json, sizeof(default_json), exe_dir, "examples/btc-trading-model.json");
    csv_path = default_csv;
    js_path = default_js;
    json_path = default_json;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (strcmp(argv[i], "--js") == 0 && i + 1 < argc) {
            js_path = argv[++i];
        } else if (strcmp(argv[i], "--json") == 0 && i + 1 < argc) {
            json_path = argv[++i];
        } else if (strcmp(argv[i], "--max-rows") == 0 && i + 1 < argc) {
            max_rows = atoi(argv[++i]);
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (max_rows <= LOOKBACK + 1) {
        usage(argv[0]);
        return 2;
    }

    memset(&list, 0, sizeof(list));
    if (!read_csv(csv_path, &list)) {
        free(list.items);
        return 1;
    }

    candles = latest_window(&list, max_rows, &n);
    weights = (double *)calloc((size_t)n * 2 * ACTIONS, sizeof(double));
    if (!weights) {
        fprintf(stderr, "out of memory\n");
        free(list.items);
        return 1;
    }

    train(candles, n, weights);

    if (!write_js(js_path, candles, n, weights)) {
        fprintf(stderr, "failed to write %s\n", js_path);
        free(weights);
        free(list.items);
        return 1;
    }
    if (!write_json(json_path, candles, n, weights)) {
        fprintf(stderr, "failed to write %s\n", json_path);
        free(weights);
        free(list.items);
        return 1;
    }

    printf("trained on %d rows from %s\n", n, csv_path);
    printf("wrote %s and %s\n", js_path, json_path);
    free(weights);
    free(list.items);
    return 0;
}
