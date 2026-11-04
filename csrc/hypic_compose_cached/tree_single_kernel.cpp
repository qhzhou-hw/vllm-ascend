// SPDX-License-Identifier: Apache-2.0
// Ordered tree in one launch. AICs serve Matmul; AIVs synchronize between levels.
#include "kernel_operator.h"
#include "matmul_intf.h"

namespace {
constexpr uint32_t DIM = 128;
// Tree/chain Matmuls use IDs 0..3; CANN workspace initialization uses 15. Do not use
// mixed SyncAll here: the AICs remain inside the Matmul service loop.
constexpr uint16_t LEVEL_READY = 8;
using Matrix = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
using Matmul = matmul::Matmul<Matrix, Matrix, Matrix, Matrix>;
using LocalMatrix = matmul::MatmulType<AscendC::TPosition::VECOUT, CubeFormat::ND, float>;
using ChainMatmul = matmul::Matmul<LocalMatrix, Matrix, Matrix, Matrix>;
}

template<uint32_t MAX_ROWS, bool HAS_TAIL>
__aicore__ inline void tree_single_body(
    AscendC::TPipe &pipe, Matmul &mm, ChainMatmul *chain,
    GM_ADDR s_in, GM_ADDR t_in, GM_ADDR metadata, GM_ADDR output, GM_ADDR buffers,
    GM_ADDR workspace, uint32_t levels, uint32_t records,
    uint32_t batch, uint32_t heads, uint32_t input_slots, uint32_t workers,
    uint32_t capacity0, uint32_t capacity1, GM_ADDR tail_packet,
    uint32_t tail_stride, uint32_t tail_rows, uint64_t system_workspace_bytes)
{
    AscendC::TBuf<AscendC::TPosition::VECCALC> value_buf, add_buf;
    pipe.InitBuffer(value_buf, MAX_ROWS * DIM * sizeof(float));
    pipe.InitBuffer(add_buf, MAX_ROWS * DIM * sizeof(float));
    auto value = value_buf.Get<float>();
    auto add = add_buf.Get<float>();
    AscendC::GlobalTensor<float> si, ti, so, to;
    AscendC::GlobalTensor<int32_t> meta;
    meta.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(metadata));
    auto s_ptr = reinterpret_cast<__gm__ float *>(s_in);
    auto t_ptr = reinterpret_cast<__gm__ float *>(t_in);
    auto bank_ptr = reinterpret_cast<__gm__ float *>(buffers);
    const uint64_t slot_size = static_cast<uint64_t>(heads) * DIM * DIM;
    uint32_t slots = input_slots;
    for (uint32_t level = 0; level < levels; ++level) {
        const int32_t nodes = meta.GetValue(level * 3);
        const int32_t rows = meta.GetValue(level * 3 + 1);
        const int32_t offset = meta.GetValue(level * 3 + 2);
        const uint32_t capacity = level % 2 ? capacity1 : capacity0;
        const bool final = !HAS_TAIL && level + 1 == levels;
        const bool valid_level = nodes > 0 && static_cast<uint32_t>(nodes) <= capacity &&
            (!final || static_cast<uint32_t>(nodes) == batch) &&
            (rows == 32 || rows == 64 || rows == 128) && rows <= MAX_ROWS &&
            offset >= static_cast<int32_t>(levels) &&
            static_cast<uint64_t>(offset) + nodes <= records;
        if (!valid_level) {
            // All workers read the same immutable header and take this branch.
            so.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output));
            AscendC::Duplicate(value, __builtin_nanf(""), MAX_ROWS * DIM);
            AscendC::PipeBarrier<PIPE_ALL>();
            const uint64_t tiles = static_cast<uint64_t>(batch) * heads * (DIM / MAX_ROWS);
            for (uint64_t task = AscendC::GetBlockIdx(); task < tiles; task += workers) {
                AscendC::DataCopy(so[task * MAX_ROWS * DIM], value, MAX_ROWS * DIM);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            return;
        }
        auto next_s = final ? reinterpret_cast<__gm__ float *>(output) :
            bank_ptr + (level % 2 ? 2 * capacity0 * slot_size : 0);
        auto next_t = bank_ptr + (level % 2 ? 2 * capacity0 * slot_size : 0) + capacity * slot_size;
        si.SetGlobalBuffer(s_ptr);
        ti.SetGlobalBuffer(t_ptr);
        so.SetGlobalBuffer(next_s);
        to.SetGlobalBuffer(next_t);
        const uint32_t row_tiles = DIM / rows;
        const uint32_t tile = rows * DIM;
        const uint32_t tasks = nodes * heads * row_tiles;
        for (uint32_t task = AscendC::GetBlockIdx(); task < tasks; task += workers) {
            const uint32_t row = task % row_tiles;
            const uint32_t head = (task / row_tiles) % heads;
            const uint32_t node = task / (row_tiles * heads);
            const uint64_t head_offset = static_cast<uint64_t>(head) * DIM * DIM;
            const uint64_t row_offset = head_offset + row * tile;
            const uint64_t dest = node * slot_size + row_offset;
            const int32_t left = meta.GetValue((offset + node) * 3);
            const int32_t right = meta.GetValue((offset + node) * 3 + 1);
            const int32_t need_t = meta.GetValue((offset + node) * 3 + 2);
            const bool valid = left >= -1 && right >= -1 &&
                (left < 0 || static_cast<uint32_t>(left) < slots) &&
                (right < 0 || static_cast<uint32_t>(right) < slots) &&
                (left >= 0 || (right == -1 && need_t == 0)) && (need_t == 0 || need_t == 1);
            if (!valid || left == -1) {
                AscendC::Duplicate(value, valid ? 0.0f : __builtin_nanf(""), tile);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopy(so[dest], value, tile);
                if (need_t) AscendC::DataCopy(to[dest], value, tile);
            } else if (right == -1) {
                AscendC::DataCopy(value, si[left * slot_size + row_offset], tile);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopy(so[dest], value, tile);
                if (need_t) {
                    AscendC::DataCopy(add, ti[left * slot_size + row_offset], tile);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    AscendC::DataCopy(to[dest], add, tile);
                }
            } else {
                // Tail M can decrease as the tree narrows, without re-registering
                // the Matmul object or relaunching its AIC service workers.
                mm.SetTail(rows, DIM, DIM);
                mm.SetTensorA(si[left * slot_size + row_offset]);
                mm.SetTensorB(ti[right * slot_size + head_offset]);
                mm.IterateAll(so[dest]);
                mm.End();
                AscendC::DataCopy(value, so[dest], tile);
                AscendC::DataCopy(add, si[right * slot_size + row_offset], tile);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::Add(value, value, add, tile);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopy(so[dest], value, tile);
                if (need_t) {
                    mm.SetTail(rows, DIM, DIM);
                    mm.SetTensorA(ti[left * slot_size + row_offset]);
                    mm.SetTensorB(ti[right * slot_size + head_offset]);
                    mm.IterateAll(to[dest]);
                    mm.End();
                }
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        if (!final) {
            // Every AIV participates, including workers with no tasks this level.
            // Matmul calls are synchronous; MTE3 completes S writes before the
            // all-AIV rendezvous. AICs stay available to serve other workers.
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::CrossCoreSetFlag<0, PIPE_MTE3>(LEVEL_READY);
            AscendC::CrossCoreWaitFlag(LEVEL_READY);
        }
        s_ptr = next_s;
        t_ptr = next_t;
        slots = nodes;
    }
    if constexpr (HAS_TAIL) {
        // All summaries are ready after the last all-AIV barrier. From here
        // each worker owns a complete row tile of a request's sequential chain.
        si.SetGlobalBuffer(s_ptr);
        ti.SetGlobalBuffer(t_ptr);
        so.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output));
        meta.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(tail_packet));
        const uint32_t tile = tail_rows * DIM;
        const uint32_t row_tiles = DIM / tail_rows;
        const uint32_t tasks = batch * heads * row_tiles;
        AscendC::GlobalTensor<float> product_gm;
        product_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace + system_workspace_bytes) +
            static_cast<uint64_t>(AscendC::GetBlockIdx()) * tile, tile);
        for (uint32_t task = AscendC::GetBlockIdx(); task < tasks; task += workers) {
            const uint32_t row = task % row_tiles;
            const uint32_t head = (task / row_tiles) % heads;
            const uint32_t request = task / (row_tiles * heads);
            const uint64_t row_offset = static_cast<uint64_t>(head) * DIM * DIM + row * tile;
            const uint64_t meta_offset = static_cast<uint64_t>(request) * tail_stride;
            const int32_t count = meta.GetValue(meta_offset);
            bool valid = count >= 0 && static_cast<uint32_t>(count) < tail_stride;
            if (valid) {
                for (int32_t step = 0; step < count; ++step) {
                    const int32_t slot = meta.GetValue(meta_offset + 1 + step);
                    valid = valid && slot >= 0 && static_cast<uint32_t>(slot) < slots;
                }
            }
            if (!valid || count == 0) {
                AscendC::Duplicate(value, valid ? 0.0f : __builtin_nanf(""), tile);
            } else {
                for (int32_t step = 0; step < count; ++step) {
                    const uint64_t slot = meta.GetValue(meta_offset + 1 + step);
                    if (step == 0) {
                        AscendC::DataCopy(value, si[slot * slot_size + row_offset], tile);
                    } else {
                        AscendC::PipeBarrier<PIPE_ALL>();
                        chain->SetTensorA(value);
                        chain->SetTensorB(ti[slot * slot_size + static_cast<uint64_t>(head) * DIM * DIM]);
                        chain->IterateAll(product_gm);
                        chain->End();
                        AscendC::DataCopy(add, product_gm, tile);
                        // Matmul consumed A; reuse its UB for the next S.
                        AscendC::DataCopy(value, si[slot * slot_size + row_offset], tile);
                        AscendC::PipeBarrier<PIPE_ALL>();
                        AscendC::Add(value, add, value, tile);
                    }
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopy(so[static_cast<uint64_t>(request) * slot_size + row_offset], value, tile);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

template<uint32_t MAX_ROWS, bool HAS_TAIL = false>
__aicore__ inline void tree_single(
    GM_ADDR si, GM_ADDR ti, GM_ADDR packet, GM_ADDR output, GM_ADDR buffers,
    GM_ADDR workspace, GM_ADDR tiling_gm, uint32_t levels, uint32_t records,
    uint32_t batch, uint32_t heads, uint32_t slots, uint32_t workers,
    uint32_t capacity0, uint32_t capacity1, GM_ADDR tail_packet = nullptr,
    GM_ADDR tail_tiling_gm = nullptr, uint32_t tail_stride = 0, uint32_t tail_rows = 0,
    uint64_t system_workspace_bytes = 0)
{
    AscendC::TPipe pipe;
    TCubeTiling tiling;
    auto words = reinterpret_cast<__gm__ uint32_t *>(tiling_gm);
    auto local_words = reinterpret_cast<uint32_t *>(&tiling);
    for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); ++i) local_words[i] = words[i];
    AscendC::SetSysWorkspaceForce(workspace);
    if ASCEND_IS_AIC {
        matmul::clearWorkspace(workspace);
    }
    Matmul mm;
    if constexpr (HAS_TAIL) {
        TCubeTiling tail_tiling;
        auto tail_words = reinterpret_cast<__gm__ uint32_t *>(tail_tiling_gm);
        auto tail_local = reinterpret_cast<uint32_t *>(&tail_tiling);
        for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); ++i) tail_local[i] = tail_words[i];
        ChainMatmul chain;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling, chain, &tail_tiling);
        tree_single_body<MAX_ROWS, true>(pipe, mm, &chain, si, ti, packet, output, buffers,
            workspace, levels, records, batch, heads, slots, workers, capacity0, capacity1,
            tail_packet, tail_stride, tail_rows, system_workspace_bytes);
    } else {
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
        tree_single_body<MAX_ROWS, false>(pipe, mm, nullptr, si, ti, packet, output, buffers,
            workspace, levels, records, batch, heads, slots, workers, capacity0, capacity1,
            nullptr, 0, 0, 0);
    }
}

// Batch scheduling prevents partial residency deadlocks with other streams.
extern "C" __schedmode__(1) __global__ __mix__(1, 2) void hypic_tree_single_kernel(
    GM_ADDR si, GM_ADDR ti, GM_ADDR packet, GM_ADDR output, GM_ADDR buffers,
    GM_ADDR workspace, GM_ADDR tiling, uint32_t levels, uint32_t records,
    uint32_t batch, uint32_t heads, uint32_t slots, uint32_t workers,
    uint32_t capacity0, uint32_t capacity1, uint32_t rows)
{
    if (rows == 32) {
        tree_single<32>(si, ti, packet, output, buffers, workspace, tiling,
            levels, records, batch, heads, slots, workers, capacity0, capacity1);
    } else if (rows == 64) {
        tree_single<64>(si, ti, packet, output, buffers, workspace, tiling,
            levels, records, batch, heads, slots, workers, capacity0, capacity1);
    } else {
        tree_single<128>(si, ti, packet, output, buffers, workspace, tiling,
            levels, records, batch, heads, slots, workers, capacity0, capacity1);
    }
}

extern "C" void launch_hypic_tree_single(
    uint32_t blocks, void *stream, void *si, void *ti, void *packet, void *output,
    void *buffers, void *workspace, void *tiling, uint32_t levels, uint32_t records,
    uint32_t batch, uint32_t heads, uint32_t slots, uint32_t capacity0, uint32_t capacity1, uint32_t rows)
{
    hypic_tree_single_kernel<<<blocks, nullptr, stream>>>(
        (GM_ADDR)si, (GM_ADDR)ti, (GM_ADDR)packet, (GM_ADDR)output, (GM_ADDR)buffers,
        (GM_ADDR)workspace, (GM_ADDR)tiling, levels, records, batch, heads, slots, blocks * 2, capacity0, capacity1, rows);
}

extern "C" __schedmode__(1) __global__ __mix__(1, 2) void hypic_tree_hybrid_kernel(
    GM_ADDR si, GM_ADDR ti, GM_ADDR packet, GM_ADDR output, GM_ADDR buffers,
    GM_ADDR workspace, GM_ADDR tiling, uint32_t levels, uint32_t records,
    uint32_t batch, uint32_t heads, uint32_t slots, uint32_t workers,
    uint32_t capacity0, uint32_t capacity1, uint32_t rows,
    GM_ADDR tail_packet, GM_ADDR tail_tiling, uint32_t tail_stride, uint32_t tail_rows, uint64_t system_workspace_bytes)
{
    if (rows == 32) {
        tree_single<32, true>(si, ti, packet, output, buffers, workspace, tiling,
            levels, records, batch, heads, slots, workers, capacity0, capacity1,
            tail_packet, tail_tiling, tail_stride, tail_rows, system_workspace_bytes);
    } else if (rows == 64) {
        tree_single<64, true>(si, ti, packet, output, buffers, workspace, tiling,
            levels, records, batch, heads, slots, workers, capacity0, capacity1,
            tail_packet, tail_tiling, tail_stride, tail_rows, system_workspace_bytes);
    } else {
        tree_single<128, true>(si, ti, packet, output, buffers, workspace, tiling,
            levels, records, batch, heads, slots, workers, capacity0, capacity1,
            tail_packet, tail_tiling, tail_stride, tail_rows, system_workspace_bytes);
    }
}

extern "C" void launch_hypic_tree_hybrid(
    uint32_t blocks, void *stream, void *si, void *ti, void *packet, void *output,
    void *buffers, void *workspace, void *tiling, uint32_t levels, uint32_t records,
    uint32_t batch, uint32_t heads, uint32_t slots, uint32_t capacity0, uint32_t capacity1, uint32_t rows,
    void *tail_packet, void *tail_tiling, uint32_t tail_stride, uint32_t tail_rows, uint64_t system_workspace_bytes)
{
    hypic_tree_hybrid_kernel<<<blocks, nullptr, stream>>>(
        (GM_ADDR)si, (GM_ADDR)ti, (GM_ADDR)packet, (GM_ADDR)output, (GM_ADDR)buffers,
        (GM_ADDR)workspace, (GM_ADDR)tiling, levels, records, batch, heads, slots, blocks * 2, capacity0, capacity1, rows,
        (GM_ADDR)tail_packet, (GM_ADDR)tail_tiling, tail_stride, tail_rows, system_workspace_bytes);
}
