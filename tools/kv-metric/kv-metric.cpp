// Fold the attention metric into a model, so a compressed KV cache damages q.k less.
//
// A key's quantization error e costs (q.e)^2 in the attention score. Scaling key dimension i by D_i
// and query dimension i by 1/D_i leaves every q.k unchanged, and with D_i^2 = sqrt(a_i / b_i), where
// a and b are the query and key energy of that dimension, the error the cache adds to q.k is
// smallest: it goes as (sum D^2 b)(sum a / D^2). Both halves of a RoPE pair share D, so D commutes
// with RoPE. The energies are measured on calibration text at the tensors that enter attention.
//
// D is written into the weights that produce q and k: attn_q_norm / attn_k_norm when the model has
// them (any GGUF, quantized or not), otherwise the rows and biases of attn_q / attn_k, which then have
// to be float (fold the f16 GGUF, quantize after). The output is marked with neuron.kv_metric.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "gguf.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

static const char * KV_METRIC_KEY = "neuron.kv_metric";

// exit codes the caller can act on
enum {
    KVM_ERR_ARGS      = 1,
    KVM_ERR_RUN       = 2,
    KVM_ERR_QUANTIZED = 3, // no q/k norm and the q/k projections are quantized: fold before quantizing
    KVM_ERR_SHARED_KV = 4, // a layer attends with another layer's keys: the fold would not be exact
    KVM_ERR_FOLDED    = 5, // the model already carries the fold
};

struct energy_collector {
    // squared activations per head dimension, summed over heads and tokens
    std::map<int, std::vector<double>> q_sum, k_sum;
    std::map<int, std::vector<double>> q_pend, k_pend; // the current batch: a later tensor of the same name replaces an earlier one
    std::vector<float> buf;

    static bool layer_of(const char * name, const char * prefix, int & il) {
        const size_t n = strlen(prefix);
        if (strncmp(name, prefix, n) != 0 || name[n] != '-') {
            return false;
        }
        char * end = nullptr;
        const long v = strtol(name + n + 1, &end, 10);
        if (end == name + n + 1 || *end != '\0') {
            return false;
        }
        il = (int) v;
        return true;
    }

    bool on_tensor(ggml_tensor * t, bool ask) {
        int il = -1;
        const bool is_q = layer_of(t->name, "Qcur", il);
        const bool is_k = !is_q && layer_of(t->name, "Kcur", il);
        const bool want = (is_q || is_k) && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t);

        if (ask || !want) {
            return ask ? want : true;
        }

        const int64_t d = t->ne[0];
        const int64_t n = ggml_nelements(t) / d;

        buf.resize(ggml_nelements(t));
        ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));

        auto & e = (is_q ? q_pend : k_pend)[il];
        e.assign(d, 0.0);
        for (int64_t r = 0; r < n; ++r) {
            const float * x = buf.data() + r*d;
            for (int64_t i = 0; i < d; ++i) {
                e[i] += (double) x[i]*x[i];
            }
        }
        return true;
    }

    void commit() {
        for (auto * m : { &q_pend, &k_pend }) {
            auto & sum = m == &q_pend ? q_sum : k_sum;
            for (auto & [il, e] : *m) {
                auto & s = sum[il];
                s.resize(e.size(), 0.0);
                for (size_t i = 0; i < e.size(); ++i) {
                    s[i] += e[i];
                }
            }
            m->clear();
        }
    }
};

static energy_collector g_collector;

static bool collect(ggml_tensor * t, bool ask, void * user_data) {
    GGML_UNUSED(user_data);
    return g_collector.on_tensor(t, ask);
}

// D per head dimension, geometric mean 1. gain is the predicted drop of the error in q.k
static std::vector<float> metric_scale(const std::vector<double> & a, const std::vector<double> & b, int n_rot, bool neox, double & gain) {
    const int d = (int) a.size();
    n_rot = std::min(n_rot, d) & ~1;

    auto partner = [&](int i) {
        if (i >= n_rot) {
            return i;
        }
        return neox ? (i < n_rot/2 ? i + n_rot/2 : i - n_rot/2) : (i ^ 1);
    };

    std::vector<float> D(d, 1.0f);
    double sa = 0.0, sb = 0.0, sab = 0.0, log_sum = 0.0;
    for (int i = 0; i < d; ++i) {
        const int p = partner(i);
        const double A = a[i] + (p != i ? a[p] : 0.0);
        const double B = b[i] + (p != i ? b[p] : 0.0);
        if (A > 0.0 && B > 0.0) {
            D[i] = (float) std::pow(A/B, 0.25);
        }
        if (p >= i) { // each pair counted once
            sa += A; sb += B; sab += std::sqrt(A*B);
        }
        log_sum += std::log(D[i]);
    }
    const float norm = (float) std::exp(log_sum/d);
    for (auto & x : D) {
        x /= norm;
    }
    gain = sab > 0.0 ? sa*sb/(sab*sab) : 1.0;
    return D;
}

static bool is_float(ggml_type t) {
    return t == GGML_TYPE_F32 || t == GGML_TYPE_F16 || t == GGML_TYPE_BF16;
}

// multiply element j of a float tensor by per_unit[(j / row) % per_unit.size()], in place in `data`
static void scale_float(ggml_type type, std::vector<uint8_t> & data, int64_t n, int64_t row, const std::vector<float> & per_unit) {
    std::vector<float> v(n);
    switch (type) {
        case GGML_TYPE_F32:  memcpy(v.data(), data.data(), n*sizeof(float)); break;
        case GGML_TYPE_F16:  ggml_fp16_to_fp32_row((const ggml_fp16_t *) data.data(), v.data(), n); break;
        case GGML_TYPE_BF16: ggml_bf16_to_fp32_row((const ggml_bf16_t *) data.data(), v.data(), n); break;
        default: GGML_ABORT("not a float type");
    }
    const int64_t units = (int64_t) per_unit.size();
    for (int64_t j = 0; j < n; ++j) {
        v[j] *= per_unit[(j/row) % units];
    }
    switch (type) {
        case GGML_TYPE_F32:  memcpy(data.data(), v.data(), n*sizeof(float)); break;
        case GGML_TYPE_F16:  ggml_fp32_to_fp16_row(v.data(), (ggml_fp16_t *) data.data(), n); break;
        case GGML_TYPE_BF16: ggml_fp32_to_bf16_row(v.data(), (ggml_bf16_t *) data.data(), n); break;
        default: GGML_ABORT("not a float type");
    }
}

struct fold_patch {
    std::string        name;
    int64_t            row;      // elements per scaled unit (1 for a vector, ne0 for a matrix row)
    std::vector<float> factor;   // per head dimension, tiled over heads
};

// before any calibration: every attention layer needs float q/k norms, or float q/k projections
static int check_foldable(const gguf_context * g, const ggml_context * meta) {
    std::map<int, bool> layers;
    for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
        int il = -1;
        char rest[128];
        if (sscanf(gguf_get_tensor_name(g, i), "blk.%d.%127s", &il, rest) == 2 &&
                (strcmp(rest, "attn_k.weight") == 0 || strcmp(rest, "attn_k_norm.weight") == 0 || strcmp(rest, "attn_qkv.weight") == 0)) {
            layers[il] = true;
        }
    }
    for (const auto & [il, _] : layers) {
        const std::string p = "blk." + std::to_string(il) + ".";
        auto get = [&](const char * n) { return ggml_get_tensor(const_cast<ggml_context *>(meta), (p + n).c_str()); };
        ggml_tensor * qn = get("attn_q_norm.weight");
        ggml_tensor * kn = get("attn_k_norm.weight");
        if (qn && kn && is_float(qn->type) && is_float(kn->type)) {
            continue;
        }
        ggml_tensor * qw = get("attn_q.weight");
        ggml_tensor * kw = get("attn_k.weight");
        if (!qw || !kw) {
            LOG_ERR("%s: layer %d has neither q/k norms nor separate attn_q / attn_k weights\n", __func__, il);
            return KVM_ERR_RUN;
        }
        for (ggml_tensor * t : { qw, kw, get("attn_q.bias"), get("attn_k.bias") }) {
            if (t && !is_float(t->type)) {
                LOG_ERR("%s: layer %d: %s is %s; fold the float GGUF and quantize it afterwards\n", __func__, il, t->name, ggml_type_name(t->type));
                return KVM_ERR_QUANTIZED;
            }
        }
    }
    if (layers.empty()) {
        LOG_ERR("%s: no attention layers found\n", __func__);
        return KVM_ERR_RUN;
    }
    return 0;
}

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s -m model.gguf -f calibration.txt -o model-kvm.gguf [-c 512] [--chunks 32]\n\n", argv[0]);
}

int main(int argc, char ** argv) {
    common_params params;
    params.n_ctx    = 512;
    params.n_chunks = 32;
    params.out_file = "";
    params.escape   = false;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_KV_METRIC, print_usage)) {
        return KVM_ERR_ARGS;
    }
    if (params.prompt.empty() || params.out_file.empty() || params.out_file == params.model.path) {
        LOG_ERR("%s: needs calibration text (-f) and an output file (-o) other than the model\n", __func__);
        return KVM_ERR_ARGS;
    }

    {
        ggml_context * meta0 = nullptr;
        const gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ &meta0 };
        gguf_context * g = gguf_init_from_file(params.model.path.c_str(), gp);
        if (!g) {
            LOG_ERR("%s: cannot read %s\n", __func__, params.model.path.c_str());
            return KVM_ERR_ARGS;
        }
        int code = 0;
        if (gguf_find_key(g, KV_METRIC_KEY) >= 0) {
            LOG_ERR("%s: %s already carries the attention metric\n", __func__, params.model.path.c_str());
            code = KVM_ERR_FOLDED;
        } else {
            code = check_foldable(g, meta0);
        }
        gguf_free(g);
        ggml_free(meta0);
        if (code != 0) {
            return code;
        }
    }

    const int32_t n_ctx = params.n_ctx;
    params.n_batch    = n_ctx;
    params.n_ubatch   = n_ctx;
    params.n_parallel = 1;
    params.cb_eval           = collect;
    params.cb_eval_user_data = nullptr;
    params.warmup            = false;

    llama_backend_init();

    auto init = common_init_from_params(params);
    llama_model   * model = init->model();
    llama_context * ctx   = init->context();
    if (!model || !ctx) {
        LOG_ERR("%s: failed to load the model\n", __func__);
        return KVM_ERR_RUN;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true, params.parse_special);
    int n_chunk = (int) (tokens.size() / n_ctx);
    if (params.n_chunks > 0) {
        n_chunk = std::min(n_chunk, params.n_chunks);
    }
    if (n_chunk < 1) {
        LOG_ERR("%s: the calibration text has %zu tokens, fewer than one chunk of %d\n", __func__, tokens.size(), n_ctx);
        return KVM_ERR_ARGS;
    }

    llama_batch batch = llama_batch_init(n_ctx, 0, 1);
    for (int c = 0; c < n_chunk; ++c) {
        llama_memory_clear(llama_get_memory(ctx), true);
        common_batch_clear(batch);
        for (int j = 0; j < n_ctx; ++j) {
            llama_token t = tokens[c*n_ctx + j];
            if (j == 0 && llama_vocab_get_add_bos(vocab)) {
                t = llama_vocab_bos(vocab);
            }
            common_batch_add(batch, t, j, { 0 }, j == n_ctx - 1);
        }
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("%s: decode failed at chunk %d\n", __func__, c);
            llama_batch_free(batch);
            return KVM_ERR_RUN;
        }
        llama_synchronize(ctx);
        g_collector.commit();
        LOG("kv-metric: chunk %d/%d\n", c + 1, n_chunk);
    }
    llama_batch_free(batch);

    // rotary layout
    char buf[128];
    std::string arch = llama_model_meta_val_str(model, "general.architecture", buf, sizeof(buf)) > 0 ? buf : "";
    int n_rot = -1;
    if (llama_model_meta_val_str(model, (arch + ".rope.dimension_count").c_str(), buf, sizeof(buf)) > 0) {
        n_rot = atoi(buf);
    }
    const enum llama_rope_type rope = llama_model_rope_type(model);
    const bool neox = rope == LLAMA_ROPE_TYPE_NEOX || rope == LLAMA_ROPE_TYPE_MROPE || rope == LLAMA_ROPE_TYPE_IMROPE;

    // the scales, and where they go
    ggml_context * meta = nullptr;
    const gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ &meta };
    gguf_context * g = gguf_init_from_file(params.model.path.c_str(), gp);
    if (!g) {
        LOG_ERR("%s: cannot read %s\n", __func__, params.model.path.c_str());
        return KVM_ERR_RUN;
    }

    std::vector<fold_patch> patches;
    std::vector<double> gains;
    int n_norm = 0, n_proj = 0;
    int ret = 0;

    auto tensor = [&](const std::string & name) -> ggml_tensor * { return ggml_get_tensor(meta, name.c_str()); };

    for (auto & [il, a] : g_collector.q_sum) {
        auto it = g_collector.k_sum.find(il);
        if (it == g_collector.k_sum.end()) {
            LOG_ERR("%s: layer %d attends with another layer's keys; the fold would not be exact there\n", __func__, il);
            ret = KVM_ERR_SHARED_KV;
            break;
        }
        const auto & b = it->second;
        if (a.size() != b.size()) {
            LOG_ERR("%s: layer %d has query dimension %zu and key dimension %zu\n", __func__, il, a.size(), b.size());
            ret = KVM_ERR_RUN;
            break;
        }

        double gain = 1.0;
        const std::vector<float> D = metric_scale(a, b, n_rot > 0 ? n_rot : (int) a.size(), neox, gain);
        std::vector<float> Dinv(D.size());
        for (size_t i = 0; i < D.size(); ++i) {
            Dinv[i] = 1.0f/D[i];
        }
        gains.push_back(gain);

        const std::string p = "blk." + std::to_string(il) + ".";
        const int64_t hd = (int64_t) D.size();

        ggml_tensor * qn = tensor(p + "attn_q_norm.weight");
        ggml_tensor * kn = tensor(p + "attn_k_norm.weight");
        if (qn && kn && is_float(qn->type) && is_float(kn->type) &&
                ggml_nelements(qn) % hd == 0 && ggml_nelements(kn) % hd == 0) {
            patches.push_back({ qn->name, 1, Dinv });
            patches.push_back({ kn->name, 1, D });
            n_norm++;
            continue;
        }

        ggml_tensor * qw = tensor(p + "attn_q.weight");
        ggml_tensor * kw = tensor(p + "attn_k.weight");
        if (!qw || !kw || qw->ne[1] % hd != 0 || kw->ne[1] % hd != 0) {
            LOG_ERR("%s: layer %d has neither per-dimension q/k norms nor separate attn_q / attn_k weights\n", __func__, il);
            ret = KVM_ERR_RUN;
            break;
        }
        if (!is_float(qw->type) || !is_float(kw->type)) {
            LOG_ERR("%s: layer %d: attn_q / attn_k are %s / %s; fold the float GGUF and quantize it afterwards\n",
                    __func__, il, ggml_type_name(qw->type), ggml_type_name(kw->type));
            ret = KVM_ERR_QUANTIZED;
            break;
        }
        patches.push_back({ qw->name, qw->ne[0], Dinv });
        patches.push_back({ kw->name, kw->ne[0], D });
        for (const char * bias : { "attn_q.bias", "attn_k.bias" }) {
            ggml_tensor * t = tensor(p + bias);
            if (t && is_float(t->type)) {
                patches.push_back({ t->name, 1, bias[5] == 'q' ? Dinv : D });
            } else if (t) {
                LOG_ERR("%s: layer %d: %s is %s\n", __func__, il, bias, ggml_type_name(t->type));
                ret = KVM_ERR_QUANTIZED;
            }
        }
        if (ret != 0) {
            break;
        }
        n_proj++;
    }

    if (ret == 0 && gains.empty()) {
        LOG_ERR("%s: no Qcur/Kcur tensors were seen; this architecture does not name its attention inputs that way\n", __func__);
        ret = KVM_ERR_RUN;
    }

    if (ret != 0) {
        gguf_free(g);
        ggml_free(meta);
        return ret;
    }

    // the output: the same header plus the marker, the same data with the patched tensors rewritten
    const size_t data_in = gguf_get_data_offset(g);
    gguf_set_val_bool(g, KV_METRIC_KEY, true);
    const size_t data_out = gguf_get_meta_size(g);

    const std::string tmp = params.out_file + ".tmp";
    bool ok = gguf_write_to_file(g, tmp.c_str(), /*only_meta =*/ true);

    FILE * fi = ok ? fopen(params.model.path.c_str(), "rb") : nullptr;
    FILE * fo = ok ? fopen(tmp.c_str(), "r+b") : nullptr;
    ok = fi && fo;
    if (ok) {
        fseek(fi, (long) data_in, SEEK_SET);
        fseek(fo, (long) data_out, SEEK_SET);
        std::vector<uint8_t> chunk(64u << 20);
        size_t n;
        while (ok && (n = fread(chunk.data(), 1, chunk.size(), fi)) > 0) {
            ok = fwrite(chunk.data(), 1, n, fo) == n;
        }
    }
    for (size_t i = 0; ok && i < patches.size(); ++i) {
        const auto & pt = patches[i];
        const int64_t id  = gguf_find_tensor(g, pt.name.c_str());
        const size_t  off = gguf_get_tensor_offset(g, id);
        const size_t  nb  = gguf_get_tensor_size(g, id);
        ggml_tensor * t = tensor(pt.name);

        std::vector<uint8_t> bytes(nb);
        fseek(fi, (long) (data_in + off), SEEK_SET);
        ok = fread(bytes.data(), 1, nb, fi) == nb;
        if (ok) {
            scale_float(t->type, bytes, ggml_nelements(t), pt.row, pt.factor);
            fseek(fo, (long) (data_out + off), SEEK_SET);
            ok = fwrite(bytes.data(), 1, nb, fo) == nb;
        }
    }
    if (fi) {
        fclose(fi);
    }
    if (fo) {
        ok = fclose(fo) == 0 && ok;
    }
    gguf_free(g);
    ggml_free(meta);

    if (!ok || rename(tmp.c_str(), params.out_file.c_str()) != 0) {
        remove(tmp.c_str());
        LOG_ERR("%s: failed to write %s\n", __func__, params.out_file.c_str());
        return KVM_ERR_RUN;
    }

    double log_gain = 0.0;
    for (double x : gains) {
        log_gain += std::log(x);
    }
    LOG("kv-metric: folded %d layers into q/k norms and %d into q/k projections; predicted error in q.k down %.2fx (geometric mean, %.2fx..%.2fx)\n",
            n_norm, n_proj, std::exp(log_gain/gains.size()),
            *std::min_element(gains.begin(), gains.end()), *std::max_element(gains.begin(), gains.end()));
    LOG("kv-metric: wrote %s\n", params.out_file.c_str());

    return 0;
}
