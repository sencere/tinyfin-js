#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH 12
#define HEIGHT 12
#define STATES (WIDTH * HEIGHT)
#define ACTIONS 4

static const char *MAZE[HEIGHT] = {
    "############",
    "#S..#.....G#",
    "#.#.#.###..#",
    "#.#...#....#",
    "#.#####.##.#",
    "#.....#..#.#",
    "###.#.##.#.#",
    "#...#....#.#",
    "#.#####.##.#",
    "#.....#....#",
    "#..H..#..H.#",
    "############"
};

typedef struct {
    int x;
    int y;
    int hit_wall;
} Move;

typedef struct {
    double reward;
    int done;
} Outcome;

static int state_index(int x, int y) {
    return y * WIDTH + x;
}

static int is_wall(int x, int y) {
    return MAZE[y][x] == '#';
}

static Move move_from(int x, int y, int action) {
    Move m;
    m.x = x;
    m.y = y;
    m.hit_wall = 0;

    if (action == 0) m.y -= 1;
    if (action == 1) m.x += 1;
    if (action == 2) m.y += 1;
    if (action == 3) m.x -= 1;

    if (is_wall(m.x, m.y)) {
        m.x = x;
        m.y = y;
        m.hit_wall = 1;
    }
    return m;
}

static Outcome outcome_for(Move m) {
    Outcome o;
    o.reward = -0.015;
    o.done = 0;

    if (m.hit_wall) {
        o.reward = -0.08;
        return o;
    }
    if (MAZE[m.y][m.x] == 'G') {
        o.reward = 1.0;
        o.done = 1;
        return o;
    }
    if (MAZE[m.y][m.x] == 'H') {
        o.reward = -0.7;
        o.done = 1;
        return o;
    }
    return o;
}

static double max4(const double *q) {
    double best = q[0];
    int i;
    for (i = 1; i < ACTIONS; i++) {
        if (q[i] > best) best = q[i];
    }
    return best;
}

static void train(double q[ACTIONS][STATES], double gamma, int iterations) {
    double v[STATES];
    double next_v[STATES];
    int i;
    int it;
    int x;
    int y;
    int a;

    for (i = 0; i < STATES; i++) {
        v[i] = 0.0;
        next_v[i] = 0.0;
    }

    for (it = 0; it < iterations; it++) {
        for (y = 0; y < HEIGHT; y++) {
            for (x = 0; x < WIDTH; x++) {
                int s = state_index(x, y);
                char c = MAZE[y][x];
                double local_q[ACTIONS];

                if (c == '#') {
                    next_v[s] = -1.0;
                    continue;
                }
                if (c == 'G' || c == 'H') {
                    next_v[s] = 0.0;
                    continue;
                }

                for (a = 0; a < ACTIONS; a++) {
                    Move m = move_from(x, y, a);
                    Outcome o = outcome_for(m);
                    local_q[a] = o.reward + (o.done ? 0.0 : gamma * v[state_index(m.x, m.y)]);
                }
                next_v[s] = max4(local_q);
            }
        }

        for (i = 0; i < STATES; i++) v[i] = next_v[i];
    }

    for (y = 0; y < HEIGHT; y++) {
        for (x = 0; x < WIDTH; x++) {
            int s = state_index(x, y);
            for (a = 0; a < ACTIONS; a++) {
                if (MAZE[y][x] == '#') {
                    q[a][s] = -1.0;
                } else {
                    Move m = move_from(x, y, a);
                    Outcome o = outcome_for(m);
                    q[a][s] = o.reward + (o.done ? 0.0 : gamma * v[state_index(m.x, m.y)]);
                }
            }
        }
    }
}

static void write_array(FILE *f, double q[ACTIONS][STATES]) {
    int a;
    int s;
    fputs("[", f);
    for (a = 0; a < ACTIONS; a++) {
        if (a > 0) fputs(",", f);
        fputs("[", f);
        for (s = 0; s < STATES; s++) {
            if (s > 0) fputs(",", f);
            fprintf(f, "%.4f", q[a][s]);
        }
        fputs("]", f);
    }
    fputs("]", f);
}

static void write_map(FILE *f) {
    int y;
    fputs("[", f);
    for (y = 0; y < HEIGHT; y++) {
        if (y > 0) fputs(",", f);
        fprintf(f, "\"%s\"", MAZE[y]);
    }
    fputs("]", f);
}

static int write_json(const char *path, double q[ACTIONS][STATES], double gamma, int iterations) {
    FILE *f = fopen(path, "w");
    if (!f) return 0;
    fprintf(f, "{\n  \"format\": \"tinyfin-js-q-table-v1\",\n");
    fprintf(f, "  \"stateSize\": %d,\n  \"actionSize\": %d,\n", STATES, ACTIONS);
    fprintf(f, "  \"width\": %d,\n  \"height\": %d,\n", WIDTH, HEIGHT);
    fprintf(f, "  \"gamma\": %.4f,\n  \"iterations\": %d,\n", gamma, iterations);
    fputs("  \"map\": ", f);
    write_map(f);
    fputs(",\n  \"weights\": ", f);
    write_array(f, q);
    fputs("\n}\n", f);
    fclose(f);
    return 1;
}

static int write_js(const char *path, double q[ACTIONS][STATES], double gamma, int iterations) {
    FILE *f = fopen(path, "w");
    if (!f) return 0;
    fputs("window.tinyfinAdvancedMazeModel = {\n", f);
    fputs("  format: \"tinyfin-js-q-table-v1\",\n", f);
    fprintf(f, "  stateSize: %d,\n  actionSize: %d,\n", STATES, ACTIONS);
    fprintf(f, "  width: %d,\n  height: %d,\n", WIDTH, HEIGHT);
    fprintf(f, "  gamma: %.4f,\n  iterations: %d,\n", gamma, iterations);
    fputs("  map: ", f);
    write_map(f);
    fputs(",\n  weights: ", f);
    write_array(f, q);
    fputs("\n};\n", f);
    fclose(f);
    return 1;
}

static void usage(const char *argv0) {
    fprintf(stderr, "Usage: %s [--js path] [--json path] [--iterations n] [--gamma g]\n", argv0);
}

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

static void default_output_path(char *out, size_t out_size, const char *exe_dir, const char *name) {
    snprintf(out, out_size, "%s/../../examples/%s", exe_dir, name);
}

int main(int argc, char **argv) {
    char exe_dir[1024];
    char default_js_path[1200];
    char default_json_path[1200];
    const char *js_path;
    const char *json_path;
    int iterations = 500;
    double gamma = 0.96;
    double q[ACTIONS][STATES];
    int i;

    executable_dir(argv[0], exe_dir, sizeof(exe_dir));
    default_output_path(default_js_path, sizeof(default_js_path), exe_dir, "advanced-maze-model.js");
    default_output_path(default_json_path, sizeof(default_json_path), exe_dir, "advanced-maze-model.json");
    js_path = default_js_path;
    json_path = default_json_path;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--js") == 0 && i + 1 < argc) {
            js_path = argv[++i];
        } else if (strcmp(argv[i], "--json") == 0 && i + 1 < argc) {
            json_path = argv[++i];
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--gamma") == 0 && i + 1 < argc) {
            gamma = atof(argv[++i]);
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (iterations <= 0 || gamma < 0.0 || gamma > 1.0) {
        usage(argv[0]);
        return 2;
    }

    train(q, gamma, iterations);

    if (!write_js(js_path, q, gamma, iterations)) {
        fprintf(stderr, "failed to write %s\n", js_path);
        return 1;
    }
    if (!write_json(json_path, q, gamma, iterations)) {
        fprintf(stderr, "failed to write %s\n", json_path);
        return 1;
    }

    printf("wrote %s and %s\n", js_path, json_path);
    return 0;
}
