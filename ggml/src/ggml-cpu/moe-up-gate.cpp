#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "ops.h"
#include "quants.h"

// Forward declaration since this isn't in a header we include
extern const struct ggml_type_traits_cpu * ggml_get_type_traits_cpu(enum ggml_type type);

// Simplified ggml_moe_up_gate compute forward that handles -1 expert indices
// (SER: Smart Expert Reduction) while using standard vec_dot for matmul.
// This enables threshold-based expert skipping without the full IQK kernel.
void ggml_compute_forward_moe_up_gate(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst) {

    const struct ggml_tensor * src0_up   = dst->src[0]; // up weights (or merged gate_up)
    const struct ggml_tensor * src0_gate = dst->src[1]; // gate weights (or NULL if merged)
    const struct ggml_tensor * src1      = dst->src[2]; // input activations
    const struct ggml_tensor * ids       = dst->src[3]; // expert indices [n_expert_used, n_tokens]

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);
    GGML_ASSERT(ids->ne[2] == 1 && ids->ne[3] == 1);

    // Use GGML_TENSOR_BINARY_OP_LOCALS-like setup
    const int64_t ne00 = src0_up->ne[0];
    const int64_t ne01 = src0_up->ne[1];
    const int64_t ne02 = src0_up->ne[2];
    const int64_t ne03 = src0_up->ne[3];
    const int64_t nb00 = src0_up->nb[0];
    const int64_t nb01 = src0_up->nb[1];
    const int64_t nb02 = src0_up->nb[2];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];
    const int64_t nb10 = src1->nb[0];
    const int64_t nb11 = src1->nb[1];
    const int64_t nb12 = src1->nb[2];

    const int64_t nb0 = dst->nb[0];
    const int64_t nb1 = dst->nb[1];
    const int64_t nb2 = dst->nb[2];

    const int ith = params->ith;
    const int nth = params->nth;

    const enum ggml_type type = src0_up->type;
    const struct ggml_type_traits_cpu * traits = ggml_get_type_traits_cpu(type);
    if (!traits || !traits->vec_dot) {
        GGML_ABORT("moe_up_gate: unsupported type");
    }
    ggml_vec_dot_t vec_dot = traits->vec_dot;
    enum ggml_type vec_dot_type = traits->vec_dot_type;

    const int64_t n_ids    = ids->ne[0]; // n_expert_used
    const int64_t n_tokens = ids->ne[1];
    const int64_t n_as     = ne02;       // n_expert

    const bool merged_gate_up = (src0_gate == NULL);
    const int64_t nr0 = merged_gate_up ? ne01 / 2 : ne01; // output rows per expert

    const int unary_op = ggml_get_op_params_i32(dst, 0);

    // Step 1: Handle -1 indices (SER) — zero out output for skipped experts, then skip them
    for (int64_t iid1 = ith; iid1 < n_tokens; iid1 += nth) {
        for (int id = 0; id < n_ids; ++id) {
            const int32_t i02 = ((const int32_t *) ((const char *) ids->data + iid1*ids->nb[1]))[id];
            if (i02 < 0 || i02 >= n_as) {
                memset((char *)dst->data + id*dst->nb[1] + iid1*dst->nb[2], 0, nr0*sizeof(float));
            }
        }
    }

    // Step 2: Quantize src1 to vec_dot_type if needed (using params->wdata)
    const void * wdata = src1->data;
    ggml_from_float_t from_float = ggml_get_type_traits_cpu(vec_dot_type)->from_float;
    if (src1->type != vec_dot_type && from_float) {
        char * wdata_ptr = (char *)params->wdata;
        const size_t nbw0 = ggml_type_size(vec_dot_type);
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);

        GGML_ASSERT(src1->type == GGML_TYPE_F32);
        for (int64_t i12 = 0; i12 < ne12; i12++) {
            for (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                from_float((float *)((char *) src1->data + i12*nb12 + i11*nb11),
                           (void *)         (wdata_ptr + i12*nbw1*ne11 + i11*nbw1),
                           ne10);
            }
        }
        ggml_barrier(params->threadpool);
        wdata = params->wdata;
    }

    // Step 3: For each (token, expert) pair, compute up+gate projections
    const size_t row_size = ggml_row_size(vec_dot_type, ne10);
    float tmp_up[256], tmp_gate[256]; // stack buffers for results

    for (int64_t iid1 = ith; iid1 < n_tokens; iid1 += nth) {
        const int32_t * ids_row = (const int32_t *) ((const char *) ids->data + iid1*ids->nb[1]);
        const char * src1_row = (const char *) wdata + iid1 * row_size;

        for (int id = 0; id < n_ids; ++id) {
            const int32_t expert_id = ids_row[id];
            if (expert_id < 0 || expert_id >= n_as) continue; // SER: skip

            float * dst_row = (float *)((char *)dst->data + id*nb1 + iid1*nb2);

            const char * up_cur, * gate_cur;
            if (merged_gate_up) {
                gate_cur = (const char *) src0_up->data + expert_id * nb02;
                up_cur   = gate_cur + nb02 / 2;
            } else {
                up_cur   = (const char *) src0_up->data   + expert_id * nb02;
                gate_cur = (const char *) src0_gate->data + expert_id * nb02;
            }

            for (int64_t ir0 = 0; ir0 < nr0; ir0++) {
                // Compute up and gate dot products
                vec_dot(ne00, &tmp_up[ir0],   0, up_cur   + ir0 * nb01, 0, src1_row, 0, 1);
                vec_dot(ne00, &tmp_gate[ir0], 0, gate_cur + ir0 * nb01, 0, src1_row, 0, 1);

                // Apply activation to gate, multiply with up
                float gate_val = tmp_gate[ir0];
                switch (unary_op) {
                    case GGML_UNARY_OP_SILU:
                        gate_val = gate_val / (1.0f + expf(-gate_val));
                        break;
                    case GGML_UNARY_OP_GELU:
                        gate_val = gate_val * 0.5f * (1.0f + tanhf(0.7978845608f * (gate_val + 0.044715f * gate_val * gate_val * gate_val)));
                        break;
                    case GGML_UNARY_OP_RELU:
                        if (gate_val < 0) gate_val = 0;
                        break;
                    default:
                        break;
                }
                dst_row[ir0] = tmp_up[ir0] * gate_val;
            }
        }
    }
}
