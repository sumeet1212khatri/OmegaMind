#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <ctime>
#include <cstdint>
#include <algorithm>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <immintrin.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
  #include <windows.h>
  static double get_ms() {
      LARGE_INTEGER f, c;
      QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
      return (double)c.QuadPart / f.QuadPart * 1000.0;
  }
#else
  #include <sys/time.h>
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <fcntl.h>
  #include <unistd.h>
  static double get_ms() {
      struct timeval tv; gettimeofday(&tv, NULL);
      return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
  }
#endif

static void* mmap_file(const char* filepath, size_t* size_out) {
    int fd = open(filepath, O_RDONLY);
    if (fd < 0) return nullptr;
    struct stat sb;
    if (fstat(fd, &sb) < 0) { close(fd); return nullptr; }
    *size_out = sb.st_size;
    void* ptr = mmap(NULL, sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (ptr == MAP_FAILED) return nullptr;
    return ptr;
}

static thread_local uint64_t tl_rng_state = 0;
static inline float rng_float() {
    if (!tl_rng_state) {
        uint64_t tid = 0;
#ifdef _OPENMP
        tid = (uint64_t)omp_get_thread_num();
#endif
        tl_rng_state = (uint64_t)get_ms() ^ (uint64_t)(uintptr_t)&tl_rng_state ^ (tid * 6364136223846793005ULL + 1);
        if (!tl_rng_state) tl_rng_state = 0xDEADBEEFCAFEBABEULL;
    }
    tl_rng_state ^= tl_rng_state << 13;
    tl_rng_state ^= tl_rng_state >> 7;
    tl_rng_state ^= tl_rng_state << 17;
    return (float)(tl_rng_state >> 11) * (1.0f / (float)(1ULL << 53));
}

struct Config { int n_layer, n_head, n_embd, block_size, vocab_size; };
struct Weights {
    float *wte, *wpe;
    float **ln1_w, **ln1_b, **c_attn_w, **c_attn_b;
    float **c_proj_w, **c_proj_b, **ln2_w, **ln2_b;
    float **fc_w, **fc_b, **mlp_proj_w, **mlp_proj_b;
    float *ln_f_w, *ln_f_b, *lm_head_w;
};

static Config  cfg;
static Weights W;
static float*  g_data = nullptr;

struct Session {
    float*           k_cache  = nullptr;
    float*           v_cache  = nullptr;
    int              pos      = 0;
    double           last_use = 0.0;
    std::vector<int> history;
};

static const int MAX_SESSIONS = 16;
static std::unordered_map<std::string, Session> g_sessions;

static float *g_x, *g_buf, *g_qkv, *g_attn_buf;
static float *g_ff, *g_logits, *g_tmp_out;
static std::pair<float,int>* g_topk_pairs = nullptr;

static float *g_px   = nullptr;
static float *g_pbuf = nullptr;
static float *g_pqkv = nullptr;

static thread_local float* tl_tmp = nullptr;
static thread_local float* tl_ff  = nullptr;

static void ensure_thread_locals() {
    if (!tl_tmp) {
        tl_tmp = (float*)malloc(cfg.n_embd * sizeof(float));
        tl_ff  = (float*)malloc(4 * cfg.n_embd * sizeof(float));
    }
}

static void layer_norm(float* out, const float* x, const float* w, const float* b, int N) {
    __m256 vsum = _mm256_setzero_ps();
    for (int i = 0; i < N; i += 8) vsum = _mm256_add_ps(vsum, _mm256_loadu_ps(x + i));
    float tmp[8]; _mm256_storeu_ps(tmp, vsum);
    float mean = (tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7]) / (float)N;
    __m256 vmean = _mm256_set1_ps(mean);
    __m256 vvar  = _mm256_setzero_ps();
    for (int i = 0; i < N; i += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(x + i), vmean);
        vvar = _mm256_fmadd_ps(d, d, vvar);
    }
    _mm256_storeu_ps(tmp, vvar);
    float var = (tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7]) / (float)N;
    __m256 vsc = _mm256_set1_ps(1.f / sqrtf(var + 1e-5f));
    for (int i = 0; i < N; i += 8) {
        __m256 d      = _mm256_sub_ps(_mm256_loadu_ps(x + i), vmean);
        __m256 norm   = _mm256_mul_ps(d, vsc);
        __m256 result = _mm256_fmadd_ps(norm, _mm256_loadu_ps(w + i), _mm256_loadu_ps(b + i));
        _mm256_storeu_ps(out + i, result);
    }
}

static void matmul_vec(float* __restrict__ out, const float* __restrict__ mat, const float* __restrict__ x, int M, int K) {
#pragma omp parallel for schedule(static) if(M >= 64)
    for (int i = 0; i < M; i++) {
        const float* row = mat + (long long)i * K;
        __m256 acc = _mm256_setzero_ps();
        int j = 0;
        for (; j <= K-8; j += 8) acc = _mm256_fmadd_ps(_mm256_loadu_ps(row+j), _mm256_loadu_ps(x+j), acc);
        float t[8]; _mm256_storeu_ps(t, acc);
        float s = t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];
        for (; j < K; j++) s += row[j] * x[j];
        out[i] = s;
    }
}

static inline void matmul_vec_serial(float* __restrict__ out, const float* __restrict__ mat, const float* __restrict__ x, int M, int K) {
    for (int i = 0; i < M; i++) {
        const float* row = mat + (long long)i * K;
        __m256 acc = _mm256_setzero_ps();
        int j = 0;
        for (; j <= K-8; j += 8) acc = _mm256_fmadd_ps(_mm256_loadu_ps(row+j), _mm256_loadu_ps(x+j), acc);
        float t[8]; _mm256_storeu_ps(t, acc);
        float s = t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];
        for (; j < K; j++) s += row[j] * x[j];
        out[i] = s;
    }
}

static inline float dot_avx2_safe(const float* __restrict__ a, const float* __restrict__ b, int n) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n-8; i += 8) acc = _mm256_fmadd_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i), acc);
    float tmp[8]; _mm256_storeu_ps(tmp, acc);
    float s = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
    for (; i < n; i++) s += a[i]*b[i];
    return s;
}

static inline void weighted_acc_avx2(float* __restrict__ out, const float* __restrict__ v, float w, int n) {
    __m256 wv = _mm256_set1_ps(w);
    int i = 0;
    for (; i <= n-8; i += 8) _mm256_storeu_ps(out+i, _mm256_fmadd_ps(wv, _mm256_loadu_ps(v+i), _mm256_loadu_ps(out+i)));
    for (; i < n; i++) out[i] += w * v[i];
}

static inline void add_bias(float* x, const float* b, int N) { for (int i = 0; i < N; i++) x[i] += b[i]; }
static inline void residual_add(float* x, const float* y, int N) { for (int i = 0; i < N; i++) x[i] += y[i]; }

static void gelu_inplace(float* x, int N) {
    for (int i = 0; i < N; i++) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
}

static void softmax_inplace(float* x, int N) {
    float mx = x[0];
    for (int i = 1; i < N; i++) if (x[i] > mx) mx = x[i];
    float s = 0.f;
    for (int i = 0; i < N; i++) { x[i] = expf(x[i]-mx); s += x[i]; }
    for (int i = 0; i < N; i++) x[i] /= s;
}

static void forward(int token_id, int pos, float* k_cache, float* v_cache) {
    const int C = cfg.n_embd, H = cfg.n_head, hs = C / H;
    float* te = W.wte + (long long)token_id * C;
    float* pe = W.wpe + (long long)pos * C;
#pragma omp parallel for
    for (int i = 0; i < C; i++) g_x[i] = te[i] + pe[i];

    for (int l = 0; l < cfg.n_layer; l++) {
        layer_norm(g_buf, g_x, W.ln1_w[l], W.ln1_b[l], C);
        matmul_vec(g_qkv, W.c_attn_w[l], g_buf, 3*C, C);
        add_bias(g_qkv, W.c_attn_b[l], 3*C);

        float* q = g_qkv;
        float* k = g_qkv + C;
        float* v = g_qkv + 2*C;
        float* kc = k_cache + (long long)l * cfg.block_size * C;
        float* vc = v_cache + (long long)l * cfg.block_size * C;
        memcpy(kc + (long long)pos*C, k, C*sizeof(float));
        memcpy(vc + (long long)pos*C, v, C*sizeof(float));

#pragma omp parallel for schedule(static)
        for (int h = 0; h < H; h++) {
            float* qh = q + h*hs;
            float scale = 1.f / sqrtf((float)hs);
            float* attn = g_attn_buf + h*cfg.block_size;
            for (int t = 0; t <= pos; t++) {
                float* kh = kc + (long long)t*C + h*hs;
                attn[t] = dot_avx2_safe(qh, kh, hs) * scale;
            }
            softmax_inplace(attn, pos+1);
            float* oh = g_buf + h*hs;
            memset(oh, 0, hs*sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float* vh = vc + (long long)t*C + h*hs;
                weighted_acc_avx2(oh, vh, attn[t], hs);
            }
        }
        matmul_vec(g_tmp_out, W.c_proj_w[l], g_buf, C, C);
        add_bias(g_tmp_out, W.c_proj_b[l], C);
        residual_add(g_x, g_tmp_out, C);

        layer_norm(g_buf, g_x, W.ln2_w[l], W.ln2_b[l], C);
        matmul_vec(g_ff, W.fc_w[l], g_buf, 4*C, C);
        add_bias(g_ff, W.fc_b[l], 4*C);
        gelu_inplace(g_ff, 4*C);
        matmul_vec(g_tmp_out, W.mlp_proj_w[l], g_ff, C, 4*C);
        add_bias(g_tmp_out, W.mlp_proj_b[l], C);
        residual_add(g_x, g_tmp_out, C);
    }
    layer_norm(g_buf, g_x, W.ln_f_w, W.ln_f_b, C);
    matmul_vec(g_logits, W.lm_head_w, g_buf, cfg.vocab_size, C);
}

static void forward_prefill_batch(const int* tokens, int T, int start_pos, float* k_cache, float* v_cache) {
    const int C = cfg.n_embd, H = cfg.n_head, hs = C / H;
    for (int t = 0; t < T; t++) {
        float* dst = g_px + (long long)t * C;
        float* te = W.wte + (long long)tokens[t] * C;
        float* pe = W.wpe + (long long)(start_pos + t) * C;
        for (int i = 0; i < C; i++) dst[i] = te[i] + pe[i];
    }

    for (int l = 0; l < cfg.n_layer; l++) {
        float* kc = k_cache + (long long)l * cfg.block_size * C;
        float* vc = v_cache + (long long)l * cfg.block_size * C;

        for (int t = 0; t < T; t++)
            layer_norm(g_pbuf + (long long)t*C, g_px + (long long)t*C, W.ln1_w[l], W.ln1_b[l], C);

#pragma omp parallel for schedule(static)
        for (int t = 0; t < T; t++) {
            float* qkv = g_pqkv + (long long)t * 3 * C;
            matmul_vec_serial(qkv, W.c_attn_w[l], g_pbuf + (long long)t*C, 3*C, C);
            const float* bias = W.c_attn_b[l];
            for (int i = 0; i < 3*C; i++) qkv[i] += bias[i];
            memcpy(kc + (long long)(start_pos + t)*C, qkv + C, C*sizeof(float));
            memcpy(vc + (long long)(start_pos + t)*C, qkv + 2*C, C*sizeof(float));
        }

        memset(g_pbuf, 0, (long long)T * C * sizeof(float));
        const float scale = 1.f / sqrtf((float)hs);

#pragma omp parallel for schedule(static)
        for (int h = 0; h < H; h++) {
            std::vector<float> attn_scores_vec(cfg.block_size);
            float* attn_scores = attn_scores_vec.data();
            for (int t = 0; t < T; t++) {
                float* qh = g_pqkv + (long long)t*3*C + h*hs;
                int seq_len = start_pos + t + 1;
                for (int s = 0; s < seq_len; s++)
                    attn_scores[s] = dot_avx2_safe(qh, kc + (long long)s*C + h*hs, hs) * scale;
                softmax_inplace(attn_scores, seq_len);
                float* oh = g_pbuf + (long long)t*C + h*hs;
                for (int s = 0; s < seq_len; s++)
                    weighted_acc_avx2(oh, vc + (long long)s*C + h*hs, attn_scores[s], hs);
            }
        }

#pragma omp parallel for schedule(static)
        for (int t = 0; t < T; t++) {
            ensure_thread_locals();
            float* xt = g_px + (long long)t * C;
            float* bt = g_pbuf + (long long)t * C;
            matmul_vec_serial(tl_tmp, W.c_proj_w[l], bt, C, C);
            const float* bias = W.c_proj_b[l];
            for (int i = 0; i < C; i++) xt[i] += tl_tmp[i] + bias[i];
        }

        for (int t = 0; t < T; t++)
            layer_norm(g_pbuf + (long long)t*C, g_px + (long long)t*C, W.ln2_w[l], W.ln2_b[l], C);

#pragma omp parallel for schedule(static)
        for (int t = 0; t < T; t++) {
            ensure_thread_locals();
            float* xt = g_px + (long long)t * C;
            float* bt = g_pbuf + (long long)t * C;
            matmul_vec_serial(tl_ff, W.fc_w[l], bt, 4*C, C);
            const float* bias_fc = W.fc_b[l];
            for (int i = 0; i < 4*C; i++) tl_ff[i] += bias_fc[i];
            gelu_inplace(tl_ff, 4*C);
            matmul_vec_serial(tl_tmp, W.mlp_proj_w[l], tl_ff, C, 4*C);
            const float* bias_p = W.mlp_proj_b[l];
            for (int i = 0; i < C; i++) xt[i] += tl_tmp[i] + bias_p[i];
        }
    }
    layer_norm(g_buf, g_px + (long long)(T-1)*C, W.ln_f_w, W.ln_f_b, C);
    matmul_vec(g_logits, W.lm_head_w, g_buf, cfg.vocab_size, C);
}

static void map_weights(float* data, long long wbytes) {
    float* p = data;
    float* end = data + wbytes / sizeof(float);
    const int C = cfg.n_embd, L = cfg.n_layer;

    W.wte = p; p += (long long)cfg.vocab_size * C;
    W.wpe = p; p += (long long)cfg.block_size * C;

    #define ARR(f) W.f = (float**)malloc(L*sizeof(float*))
    ARR(ln1_w); ARR(ln1_b); ARR(c_attn_w); ARR(c_attn_b);
    ARR(c_proj_w); ARR(c_proj_b); ARR(ln2_w); ARR(ln2_b);
    ARR(fc_w); ARR(fc_b); ARR(mlp_proj_w); ARR(mlp_proj_b);
    #undef ARR

    for (int l = 0; l < L; l++) {
        W.ln1_w[l]      = p; p += C;
        W.ln1_b[l]      = p; p += C;
        W.c_attn_w[l]   = p; p += 3LL*C*C;
        W.c_attn_b[l]   = p; p += 3LL*C;
        W.c_proj_w[l]   = p; p += 1LL*C*C;
        W.c_proj_b[l]   = p; p += C;
        W.ln2_w[l]      = p; p += C;
        W.ln2_b[l]      = p; p += C;
        W.fc_w[l]       = p; p += 4LL*C*C;
        W.fc_b[l]       = p; p += 4LL*C;
        W.mlp_proj_w[l] = p; p += 1LL*C*4*C;
        W.mlp_proj_b[l] = p; p += C;
    }
    W.ln_f_w    = p; p += C;
    W.ln_f_b    = p; p += C;
    W.lm_head_w = p; p += (long long)cfg.vocab_size * C;
}

static int sample_token(float temperature, int top_k, float top_p, float rep_penalty, const std::vector<int>& history) {
    if (rep_penalty > 1.0f && !history.empty()) {
        int start_idx = std::max(0, (int)history.size() - 64);
        std::unordered_set<int> recent(history.begin() + start_idx, history.end());
        for (int tok_id : recent) {
            if (tok_id >= 0 && tok_id < cfg.vocab_size) {
                if (g_logits[tok_id] > 0.0f) g_logits[tok_id] /= rep_penalty;
                else                         g_logits[tok_id] *= rep_penalty;
            }
        }
    }

    float inv_temp = 1.0f / std::max(temperature, 1e-4f);
    for (int v = 0; v < cfg.vocab_size; v++) g_topk_pairs[v] = {g_logits[v] * inv_temp, v};

    int K = std::min(std::max(top_k, 1), cfg.vocab_size);
    std::partial_sort(g_topk_pairs, g_topk_pairs + K, g_topk_pairs + cfg.vocab_size,
        [](const auto& a, const auto& b){ return a.first > b.first; });

    float max_logit = g_topk_pairs[0].first;
    float sum = 0.f;
    for (int j = 0; j < K; j++) {
        g_topk_pairs[j].first = expf(g_topk_pairs[j].first - max_logit);
        sum += g_topk_pairs[j].first;
    }
    for (int j = 0; j < K; j++) g_topk_pairs[j].first /= sum;

    int active_k = K;
    if (top_p > 0.0f && top_p < 1.0f) {
        float cum_prob = 0.0f;
        for (int j = 0; j < K; j++) {
            cum_prob += g_topk_pairs[j].first;
            if (cum_prob >= top_p) { active_k = j + 1; break; }
        }
        float p_sum = 0.0f;
        for (int j = 0; j < active_k; j++) p_sum += g_topk_pairs[j].first;
        for (int j = 0; j < active_k; j++) g_topk_pairs[j].first /= p_sum;
    }

    float r = rng_float(), cum = 0.f;
    int best = g_topk_pairs[0].second;
    for (int j = 0; j < active_k; j++) {
        cum += g_topk_pairs[j].first;
        if (r < cum) { best = g_topk_pairs[j].second; break; }
    }
    return best;
}

static void handle_request(const std::string& line) {
    auto parts = [line](){
        std::vector<std::string> out; std::string cur;
        for (char c : line) { if (c=='|'){out.push_back(cur);cur.clear();}else cur+=c; }
        out.push_back(cur); return out;
    }();
    if (parts.size() < 7) { printf("ERROR bad_format\n"); fflush(stdout); return; }

    std::string sess_id = parts[1];
    auto parse_ints = [](const std::string& s){
        std::vector<int> out; std::string cur;
        for (char c : s) { if (c==','){if(!cur.empty()){out.push_back(atoi(cur.c_str()));cur.clear();}}else cur+=c; }
        if(!cur.empty()) out.push_back(atoi(cur.c_str()));
        return out;
    };
    auto new_tokens = parse_ints(parts[2]);
    int max_new = atoi(parts[3].c_str());
    float temp = (float)atof(parts[4].c_str());
    int top_k = atoi(parts[5].c_str());
    auto stop_list = parse_ints(parts[6]);
    float top_p = (parts.size() >= 8) ? (float)atof(parts[7].c_str()) : 0.85f;
    float rep_penalty = (parts.size() >= 9) ? (float)atof(parts[8].c_str()) : 1.18f;

    std::unordered_set<int> stop_ids(stop_list.begin(), stop_list.end());
    stop_ids.insert(50256);

    auto it = g_sessions.find(sess_id);
    if (it == g_sessions.end()) {
        long long nb = (long long)cfg.n_layer * cfg.block_size * cfg.n_embd * sizeof(float);
        float* kc = (float*)calloc(nb, 1);
        float* vc = (float*)calloc(nb, 1);
        g_sessions[sess_id] = Session{kc, vc, 0, get_ms(), {}};
        it = g_sessions.find(sess_id);
    }
    Session& sess = it->second;

    if (sess.pos + (int)new_tokens.size() >= cfg.block_size) { sess.pos = 0; sess.history.clear(); }
    for (int t : new_tokens) sess.history.push_back(t);

    double t0 = get_ms();
    forward_prefill_batch(new_tokens.data(), (int)new_tokens.size(), sess.pos, sess.k_cache, sess.v_cache);
    sess.pos += (int)new_tokens.size();

    int gen = 0;
    for (int i = 0; i < max_new; i++) {
        if (sess.pos >= cfg.block_size) break;
        int next = sample_token(temp, top_k, top_p, rep_penalty, sess.history);
        sess.history.push_back(next);
        printf("TOKEN %d %.2f %d\n", next, get_ms() - t0, sess.pos);
        fflush(stdout);
        gen++;
        if (stop_ids.count(next)) break;
        forward(next, sess.pos, sess.k_cache, sess.v_cache);
        sess.pos++;
    }
    printf("DONE %d %.2f %d\n", gen, get_ms() - t0, sess.pos);
    fflush(stdout);
}

int main() {
    size_t mapped_size = 0;
    int fd = open("model.bin", O_RDONLY);
    struct stat sb; fstat(fd, &sb);
    mapped_size = sb.st_size;
    void* file_data = mmap(NULL, mapped_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);

    memcpy(&cfg, (char*)file_data, 5 * sizeof(int));
    map_weights((float*)((char*)file_data + 20), mapped_size - 20);

    const int C = cfg.n_embd;
    g_x = (float*)malloc(C * sizeof(float));
    g_buf = (float*)malloc(C * sizeof(float));
    g_qkv = (float*)malloc(3*C * sizeof(float));
    g_attn_buf = (float*)malloc((long long)cfg.n_head * cfg.block_size * sizeof(float));
    g_ff = (float*)malloc(4*C * sizeof(float));
    g_logits = (float*)malloc((long long)cfg.vocab_size * sizeof(float));
    g_tmp_out = (float*)malloc(C * sizeof(float));
    g_topk_pairs = (std::pair<float,int>*)malloc((long long)cfg.vocab_size * sizeof(std::pair<float,int>));

    g_px = (float*)malloc((long long)cfg.block_size * C * sizeof(float));
    g_pbuf = (float*)malloc((long long)cfg.block_size * C * sizeof(float));
    g_pqkv = (float*)malloc((long long)cfg.block_size * 3 * C * sizeof(float));

    printf("READY\n"); fflush(stdout);

    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back()=='\r') line.pop_back();
        if (line.empty()) continue;
        if (line == "QUIT") break;
        else if (line.rfind("RESET|", 0)==0) { g_sessions.clear(); printf("RESET_OK\n"); fflush(stdout); }
        else if (line.rfind("REQUEST|", 0)==0) handle_request(line);
    }
    return 0;
}