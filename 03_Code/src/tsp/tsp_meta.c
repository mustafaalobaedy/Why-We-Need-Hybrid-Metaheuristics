/* ACS, ILS (2-opt + double bridge) and their hybrids for the symmetric TSP.
 * Budget: CPU time of the calling process (clock()).
 * alg: 0 = ACS, 1 = ILS, 2 = ACS+2-opt (memetic, LRH), 3 = ACS->ILS (relay, HRH)
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ------------------------------------------------------------------ random numbers (splitmix64) */
static unsigned long long rs;
static inline unsigned long long rnext(void) {
    unsigned long long z = (rs += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static inline double urand(void) { return (rnext() >> 11) * (1.0 / 9007199254740992.0); }
static inline int irand(int k) { return (int)(urand() * k); }

/* ------------------------------------------------------------------ instance and budget */
static int n, nn;
static const int *D, *NN;
#define DIST(a, b) D[(a) * n + (b)]

static double t0, tlimit;
static const double *cpf;
static int ncp, cp;
static long long best_len, *trace;
static int *best_tour;

static double elapsed(void) { return (double)clock() / CLOCKS_PER_SEC - t0; }
static int time_up(void) { return elapsed() >= tlimit; }

static void flush_checkpoints(void) {
    double el = elapsed();
    while (cp < ncp && el >= cpf[cp] * tlimit) trace[cp++] = best_len;
}

static void offer(const int *tour, long long len) {
    flush_checkpoints();
    if (len < best_len) {
        best_len = len;
        memcpy(best_tour, tour, n * sizeof(int));
    }
}

static long long tour_length(const int *t) {
    long long s = 0;
    for (int i = 0; i < n - 1; i++) s += DIST(t[i], t[i + 1]);
    return s + DIST(t[n - 1], t[0]);
}

/* ------------------------------------------------------------------ 2-opt with neighbor lists and don't-look bits */
static int *pos;
static unsigned char *dlb;

static void reverse_path(int *t, int i, int j) { /* reverse positions i..j (cyclic, forward from i) */
    int len = (j - i + n) % n + 1;
    for (int k = 0; k < len / 2; k++) {
        int a = (i + k) % n, b = (j - k + n) % n;
        int ca = t[a], cb = t[b];
        t[a] = cb; pos[cb] = a;
        t[b] = ca; pos[ca] = b;
    }
}

static void reverse_shorter(int *t, int from_city, int to_city, int comp_from, int comp_to) {
    int i = pos[from_city], j = pos[to_city];
    int len = (j - i + n) % n + 1;
    if (2 * len <= n) reverse_path(t, i, j);
    else reverse_path(t, pos[comp_from], pos[comp_to]);
}

/* first-improvement 2-opt; cities whose don't-look bit is 0 are processed */
static long long two_opt(int *t, long long len) {
    for (int i = 0; i < n; i++) pos[t[i]] = i;
    int improved = 1;
    while (improved) {
        improved = 0;
        for (int r = 0; r < n; r++) {
            int c1 = t[r];
            if (dlb[c1]) continue;
            int found = 0;
            /* successor direction: remove (c1,s1),(c2,s2); add (c1,c2),(s1,s2) */
            int s1 = t[(pos[c1] + 1) % n];
            int rad = DIST(c1, s1);
            for (int k = 0; k < nn && !found; k++) {
                int c2 = NN[c1 * nn + k];
                int g = DIST(c1, c2);
                if (g >= rad) break;
                int s2 = t[(pos[c2] + 1) % n];
                if (c2 == s1 || s2 == c1) continue;
                int delta = g + DIST(s1, s2) - rad - DIST(c2, s2);
                if (delta < 0) {
                    reverse_shorter(t, s1, c2, s2, c1);
                    len += delta;
                    dlb[c1] = dlb[s1] = dlb[c2] = dlb[s2] = 0;
                    found = 1;
                }
            }
            /* predecessor direction: remove (p1,c1),(p2,c2); add (c1,c2),(p1,p2) */
            if (!found) {
                int p1 = t[(pos[c1] - 1 + n) % n];
                rad = DIST(p1, c1);
                for (int k = 0; k < nn && !found; k++) {
                    int c2 = NN[c1 * nn + k];
                    int g = DIST(c1, c2);
                    if (g >= rad) break;
                    int p2 = t[(pos[c2] - 1 + n) % n];
                    if (c2 == p1 || p2 == c1) continue;
                    int delta = g + DIST(p1, p2) - rad - DIST(p2, c2);
                    if (delta < 0) {
                        reverse_shorter(t, c1, p2, c2, p1);
                        len += delta;
                        dlb[c1] = dlb[p1] = dlb[c2] = dlb[p2] = 0;
                        found = 1;
                    }
                }
            }
            if (found) improved = 1;
            else dlb[c1] = 1;
        }
    }
    return len;
}

static long long full_two_opt(int *t) {
    memset(dlb, 0, n);
    return two_opt(t, tour_length(t));
}

/* ------------------------------------------------------------------ ILS (Lourenço et al.): double bridge + 2-opt, better acceptance */
static void double_bridge(const int *src, int *dst) {
    int p[3];
    for (int k = 0; k < 3; k++) p[k] = 1 + irand(n - 1);
    for (int a = 0; a < 2; a++)          /* sort the three cut points */
        for (int b = 0; b < 2 - a; b++)
            if (p[b] > p[b + 1]) { int x = p[b]; p[b] = p[b + 1]; p[b + 1] = x; }
    int i = p[0], j = p[1], k = p[2], m = 0;
    /* A=[0,i) B=[i,j) C=[j,k) D=[k,n) -> A C B D */
    for (int q = 0; q < i; q++) dst[m++] = src[q];
    for (int q = j; q < k; q++) dst[m++] = src[q];
    for (int q = i; q < j; q++) dst[m++] = src[q];
    for (int q = k; q < n; q++) dst[m++] = src[q];
}

static void ils(const int *start) {
    int *cur = malloc(n * sizeof(int)), *cand = malloc(n * sizeof(int));
    memcpy(cur, start, n * sizeof(int));
    long long cur_len = full_two_opt(cur);  /* 2-opt never lengthens the start tour */
    offer(cur, cur_len);
    while (!time_up()) {
        double_bridge(cur, cand);
        memset(dlb, 1, n);
        /* only the endpoints of the four changed edges are re-examined */
        for (int q = 0; q < n; q++) {
            int a = cand[q], b = cand[(q + 1) % n];
            /* an edge is new if it is not an edge of cur */
            int pa = pos[a];  /* pos still refers to cur */
            if (cur[(pa + 1) % n] != b && cur[(pa - 1 + n) % n] != b) dlb[a] = dlb[b] = 0;
        }
        long long l = two_opt(cand, tour_length(cand));
        if (l < cur_len) {
            int *x = cur; cur = cand; cand = x;
            cur_len = l;
            offer(cur, cur_len);
        } else {
            for (int q = 0; q < n; q++) pos[cur[q]] = q;  /* restore positions of the incumbent */
            flush_checkpoints();
        }
    }
    free(cur); free(cand);
}

/* ------------------------------------------------------------------ ACS (Dorigo & Gambardella, 1997) */
static void acs(int local_search, double phase_share) {
    const int m = 10;
    const double beta = 2.0, rho = 0.1, xi = 0.1, q0 = 0.9;
    double phase_end = tlimit * (1.0 - phase_share);
    double *tau = malloc((size_t)n * n * sizeof(double)), *heur = malloc((size_t)n * n * sizeof(double));
    double *tot = malloc((size_t)n * n * sizeof(double)), *prob = malloc(nn * sizeof(double));
    int *tour = malloc(n * sizeof(int));
    unsigned char *vis = malloc(n);

    /* nearest-neighbour tour for tau0 */
    memset(vis, 0, n);
    int c = irand(n); tour[0] = c; vis[c] = 1;
    for (int s = 1; s < n; s++) {
        int bj = -1;
        for (int j = 0; j < n; j++) if (!vis[j] && (bj < 0 || DIST(c, j) < DIST(c, bj))) bj = j;
        tour[s] = bj; vis[bj] = 1; c = bj;
    }
    long long lnn = tour_length(tour);
    offer(tour, lnn);
    double tau0 = 1.0 / ((double)n * lnn);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            tau[i * n + j] = tau0;
            heur[i * n + j] = (i == j) ? 0.0 : pow(1.0 / (DIST(i, j) > 0 ? DIST(i, j) : 0.1), beta);
            tot[i * n + j] = tau0 * heur[i * n + j];
        }
    long long gb_len = lnn;
    int *gb = malloc(n * sizeof(int));
    memcpy(gb, tour, n * sizeof(int));

    while (elapsed() < phase_end) {
        for (int k = 0; k < m; k++) {
            memset(vis, 0, n);
            c = irand(n); tour[0] = c; vis[c] = 1;
            for (int s = 1; s < n; s++) {
                int next = -1;
                if (urand() < q0) {
                    double bv = -1.0;
                    for (int q = 0; q < nn; q++) {
                        int j = NN[c * nn + q];
                        if (!vis[j] && tot[c * n + j] > bv) { bv = tot[c * n + j]; next = j; }
                    }
                } else {
                    double sum = 0.0;
                    for (int q = 0; q < nn; q++) {
                        int j = NN[c * nn + q];
                        prob[q] = vis[j] ? 0.0 : tot[c * n + j];
                        sum += prob[q];
                    }
                    if (sum > 0.0) {
                        double r = urand() * sum, acc = 0.0;
                        for (int q = 0; q < nn; q++) {
                            acc += prob[q];
                            if (prob[q] > 0.0 && acc >= r) { next = NN[c * nn + q]; break; }
                        }
                        if (next < 0)
                            for (int q = nn - 1; q >= 0; q--) if (prob[q] > 0.0) { next = NN[c * nn + q]; break; }
                    }
                }
                if (next < 0) { /* all candidates visited: best of the remaining cities */
                    double bv = -1.0;
                    for (int j = 0; j < n; j++) if (!vis[j] && tot[c * n + j] > bv) { bv = tot[c * n + j]; next = j; }
                }
                /* local pheromone update */
                double v = (1.0 - xi) * tau[c * n + next] + xi * tau0;
                tau[c * n + next] = tau[next * n + c] = v;
                tot[c * n + next] = tot[next * n + c] = v * heur[c * n + next];
                tour[s] = next; vis[next] = 1; c = next;
            }
            double v = (1.0 - xi) * tau[c * n + tour[0]] + xi * tau0;
            tau[c * n + tour[0]] = tau[tour[0] * n + c] = v;
            tot[c * n + tour[0]] = tot[tour[0] * n + c] = v * heur[c * n + tour[0]];
            long long len = local_search ? full_two_opt(tour) : tour_length(tour);
            if (len < gb_len) { gb_len = len; memcpy(gb, tour, n * sizeof(int)); }
            offer(tour, len);
            if (elapsed() >= phase_end) break;
        }
        /* global pheromone update on the best-so-far tour */
        for (int s = 0; s < n; s++) {
            int a = gb[s], b = gb[(s + 1) % n];
            double v = (1.0 - rho) * tau[a * n + b] + rho / (double)gb_len;
            tau[a * n + b] = tau[b * n + a] = v;
            tot[a * n + b] = tot[b * n + a] = v * heur[a * n + b];
        }
    }
    if (phase_share > 0.0) ils(gb);
    free(tau); free(heur); free(tot); free(prob); free(tour); free(vis); free(gb);
}

/* ------------------------------------------------------------------ entry point */
long long run_tsp(int n_, int nn_, const int *D_, const int *NN_, int alg, double time_limit,
                  unsigned long long seed, double share, const double *cpf_, int ncp_,
                  long long *trace_, int *best_tour_) {
    n = n_; nn = nn_; D = D_; NN = NN_;
    rs = seed; tlimit = time_limit; cpf = cpf_; ncp = ncp_; cp = 0; trace = trace_;
    best_tour = best_tour_; best_len = (long long)1 << 62;
    pos = malloc(n * sizeof(int));
    dlb = malloc(n);
    t0 = (double)clock() / CLOCKS_PER_SEC;
    if (alg == 0) acs(0, 0.0);
    else if (alg == 2) acs(1, 0.0);
    else if (alg == 3) acs(0, share);
    else {
        int *t = malloc(n * sizeof(int));
        for (int i = 0; i < n; i++) t[i] = i;
        for (int i = n - 1; i > 0; i--) { int j = irand(i + 1); int x = t[i]; t[i] = t[j]; t[j] = x; }
        ils(t);
        free(t);
    }
    while (cp < ncp) trace[cp++] = best_len;
    free(pos); free(dlb);
    return best_len;
}
