// SPDX-License-Identifier: Apache-2.0
// One ordered binary reduction level. Adjacent nodes only; no padded GEMMs.
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace {
constexpr uint32_t DIM = 128;
using Matrix = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
using Matmul = matmul::Matmul<Matrix, Matrix, Matrix, Matrix>;
}

template<uint32_t ROWS>
__aicore__ inline void tree_level(
    GM_ADDR s_in, GM_ADDR t_in, GM_ADDR metadata, GM_ADDR s_out, GM_ADDR t_out,
    GM_ADDR workspace, GM_ADDR tiling_gm, uint32_t nodes, uint32_t heads,
    uint32_t input_slots, uint32_t workers)
{
    constexpr uint32_t TILE = ROWS * DIM;
    constexpr uint32_t ROW_TILES = DIM / ROWS;
    AscendC::TPipe pipe;
    TCubeTiling tiling;
    auto words = reinterpret_cast<__gm__ uint32_t *>(tiling_gm);
    auto local_words = reinterpret_cast<uint32_t *>(&tiling);
    for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); ++i) {
        local_words[i] = words[i];
    }
    AscendC::SetSysWorkspaceForce(workspace);
    if ASCEND_IS_AIC {
        matmul::clearWorkspace(workspace);
    }
    Matmul mm;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
    AscendC::TBuf<AscendC::TPosition::VECCALC> value_buf, add_buf;
    pipe.InitBuffer(value_buf, TILE * sizeof(float));
    pipe.InitBuffer(add_buf, TILE * sizeof(float));
    auto value = value_buf.Get<float>();
    auto add = add_buf.Get<float>();
    AscendC::GlobalTensor<float> si, ti, so, to;
    AscendC::GlobalTensor<int32_t> meta;
    si.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(s_in));
    ti.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(t_in));
    so.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(s_out));
    to.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(t_out));
    meta.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(metadata));
    const uint64_t slot_size = static_cast<uint64_t>(heads) * DIM * DIM;
    const uint32_t tasks = nodes * heads * ROW_TILES;
    for (uint32_t task = AscendC::GetBlockIdx(); task < tasks; task += workers) {
        const uint32_t row = task % ROW_TILES;
        const uint32_t head = (task / ROW_TILES) % heads;
        const uint32_t node = task / (ROW_TILES * heads);
        const uint64_t head_offset = static_cast<uint64_t>(head) * DIM * DIM;
        const uint64_t row_offset = head_offset + row * TILE;
        const uint64_t dest = node * slot_size + row_offset;
        const int32_t left = meta.GetValue(node * 3);
        const int32_t right = meta.GetValue(node * 3 + 1);
        const int32_t need_t = meta.GetValue(node * 3 + 2);
        const bool valid = left >= -1 && right >= -1 &&
            (left < 0 || static_cast<uint32_t>(left) < input_slots) &&
            (right < 0 || static_cast<uint32_t>(right) < input_slots) &&
            (left >= 0 || (right == -1 && need_t == 0)) && (need_t == 0 || need_t == 1);
        if (!valid || left == -1) {
            AscendC::Duplicate(value, valid ? 0.0f : __builtin_nanf(""), TILE);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopy(so[dest], value, TILE);
            if (need_t) {
                AscendC::DataCopy(to[dest], value, TILE);
            }
        } else if (right == -1) {
            // Odd trailing node: carry it verbatim, without multiplying by I.
            AscendC::DataCopy(value, si[left * slot_size + row_offset], TILE);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopy(so[dest], value, TILE);
            if (need_t) {
                AscendC::DataCopy(add, ti[left * slot_size + row_offset], TILE);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopy(to[dest], add, TILE);
            }
        } else {
            // S_AB = S_A @ T_B + S_B. A and C are explicit, disjoint GM slices.
            mm.SetTensorA(si[left * slot_size + row_offset]);
            mm.SetTensorB(ti[right * slot_size + head_offset]);
            mm.IterateAll(so[dest]);
            mm.End();
            AscendC::DataCopy(value, so[dest], TILE);
            AscendC::DataCopy(add, si[right * slot_size + row_offset], TILE);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Add(value, value, add, TILE);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopy(so[dest], value, TILE);
            if (need_t) {
                // The first node of every request never needs T: H starts at 0.
                mm.SetTensorA(ti[left * slot_size + row_offset]);
                mm.SetTensorB(ti[right * slot_size + head_offset]);
                mm.IterateAll(to[dest]);
                mm.End();
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

extern "C" __global__ __aicore__ void hypic_tree_level_kernel(
    GM_ADDR si, GM_ADDR ti, GM_ADDR packet, GM_ADDR so, GM_ADDR to, GM_ADDR workspace,
    GM_ADDR tiling, uint32_t nodes, uint32_t heads, uint32_t slots, uint32_t workers, uint32_t rows)
{
    if (rows == 32) {
        tree_level<32>(si, ti, packet, so, to, workspace, tiling, nodes, heads, slots, workers);
    } else if (rows == 64) {
        tree_level<64>(si, ti, packet, so, to, workspace, tiling, nodes, heads, slots, workers);
    } else {
        tree_level<128>(si, ti, packet, so, to, workspace, tiling, nodes, heads, slots, workers);
    }
}

extern "C" void launch_hypic_tree_level(
    uint32_t blocks, void *stream, void *si, void *ti, void *packet, void *so, void *to,
    void *workspace, void *tiling, uint32_t nodes, uint32_t heads, uint32_t slots, uint32_t rows)
{
    hypic_tree_level_kernel<<<blocks, nullptr, stream>>>(
        static_cast<GM_ADDR>(si), static_cast<GM_ADDR>(ti), static_cast<GM_ADDR>(packet),
        static_cast<GM_ADDR>(so), static_cast<GM_ADDR>(to), static_cast<GM_ADDR>(workspace),
        static_cast<GM_ADDR>(tiling), nodes, heads, slots, blocks * 2, rows);
}
