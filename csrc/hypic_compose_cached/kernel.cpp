// SPDX-License-Identifier: Apache-2.0
// Cached-only ordered FP32 affine composition, Ascend910B prototype.
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace {
constexpr uint32_t DIM = 128;

using AType = matmul::MatmulType<AscendC::TPosition::VECOUT, CubeFormat::ND, float>;
using BType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
using CType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
using BiasType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
using Matmul = matmul::Matmul<AType, BType, CType, BiasType>;
}  // namespace

template<uint32_t ROWS>
__aicore__ inline void compose_chain(
    GM_ADDR zero, GM_ADDR transition, GM_ADDR packet, GM_ADDR output,
    GM_ADDR workspace, GM_ADDR tiling_gm, uint32_t batch, uint32_t heads,
    uint32_t stride, uint32_t pool_slots, uint32_t workers, uint64_t system_workspace_bytes)
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
    // CANN 9 kernel-launch reads the KFC workspace register. The deprecated
    // SetSysWorkspace only writes a legacy global and does not initialize it.
    AscendC::SetSysWorkspaceForce(workspace);
    // This direct-launch signature is not a HAVE_WORKSPACE custom-op signature.
    // Match CANN's generated wrapper: clear KFC mailboxes and notify the AIVs
    // before REGIST's clients wait for WORKSPACE_SYNC_ID.
    if ASCEND_IS_AIC {
        matmul::clearWorkspace(workspace);
    }
    Matmul mm;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);

    // Matmul's AIC service and inter-core protocol are managed by REGIST.
    // Each AIV owns an independent row tile; columns are never partitioned.
    AscendC::TBuf<AscendC::TPosition::VECCALC> state_buf;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> product_queue;
    pipe.InitBuffer(state_buf, TILE * sizeof(float));
    pipe.InitBuffer(product_queue, 1, TILE * sizeof(float));
    auto state = state_buf.Get<float>();
    AscendC::GlobalTensor<float> s, t, out, product_gm;
    AscendC::GlobalTensor<int32_t> meta;
    s.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(zero));
    t.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(transition));
    out.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output));
    // A separate slice per AIV avoids using KFC's circular UB staging allocator
    // for C. In particular, two 64-KiB allocations per step can wrap that ring.
    product_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace + system_workspace_bytes) +
        static_cast<uint64_t>(AscendC::GetBlockIdx()) * TILE, TILE);
    meta.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(packet));
    const uint64_t head_size = DIM * DIM;
    const uint64_t slot_size = static_cast<uint64_t>(heads) * head_size;
    const uint32_t tasks = batch * heads * ROW_TILES;
    for (uint32_t task = AscendC::GetBlockIdx(); task < tasks; task += workers) {
        const uint32_t row_tile = task % ROW_TILES;
        const uint32_t head = (task / ROW_TILES) % heads;
        const uint32_t request = task / (ROW_TILES * heads);
        const uint64_t row_offset = static_cast<uint64_t>(head) * head_size + row_tile * TILE;
        const uint64_t meta_offset = static_cast<uint64_t>(request) * stride;
        const int32_t count = meta.GetValue(meta_offset);
        bool valid = count >= 0 && static_cast<uint32_t>(count) < stride;
        // Defensive validation before any pool access, including empty pools.
        if (valid) {
            for (int32_t step = 0; step < count; ++step) {
                const int32_t slot = meta.GetValue(meta_offset + 1 + step);
                valid = valid && slot >= 0 && static_cast<uint32_t>(slot) < pool_slots;
            }
        }
        if (!valid || count == 0) {
            AscendC::Duplicate(state, valid ? 0.0f : __builtin_nanf(""), TILE);
        } else {
            for (int32_t step = 0; step < count; ++step) {
                const uint64_t slot = static_cast<uint64_t>(meta.GetValue(meta_offset + 1 + step));
                if (step == 0) {
                    AscendC::DataCopy(state, s[slot * slot_size + row_offset], TILE);
                } else {
                    // Finish the preceding Vector update before Matmul consumes A.
                    AscendC::PipeBarrier<PIPE_ALL>();
                    mm.SetTensorA(state);
                    mm.SetTensorB(t[slot * slot_size + static_cast<uint64_t>(head) * head_size]);
                    auto product = product_queue.AllocTensor<float>();
                    mm.IterateAll(product_gm);
                    mm.End();
                    AscendC::DataCopy(product, product_gm, TILE);
                    product_queue.EnQue(product);
                    product = product_queue.DeQue<float>();
                    // A has been consumed: reuse its UB for S instead of reserving
                    // a third tile. This permits a complete 128-row state in UB.
                    AscendC::DataCopy(state, s[slot * slot_size + row_offset], TILE);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    AscendC::Add(state, product, state, TILE);
                    product_queue.FreeTensor(product);
                }
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(out[static_cast<uint64_t>(request) * slot_size + row_offset], state, TILE);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

extern "C" __global__ __aicore__ void hypic_cached_compose_kernel_v2(
    GM_ADDR zero, GM_ADDR transition, GM_ADDR packet, GM_ADDR output,
    GM_ADDR workspace, GM_ADDR tiling, uint32_t batch, uint32_t heads,
    uint32_t stride, uint32_t slots, uint32_t workers, uint32_t rows, uint64_t system_workspace_bytes)
{
    if (rows == 32) {
        compose_chain<32>(zero, transition, packet, output, workspace, tiling, batch, heads, stride, slots, workers, system_workspace_bytes);
    } else if (rows == 64) {
        compose_chain<64>(zero, transition, packet, output, workspace, tiling, batch, heads, stride, slots, workers, system_workspace_bytes);
    } else {
        compose_chain<128>(zero, transition, packet, output, workspace, tiling, batch, heads, stride, slots, workers, system_workspace_bytes);
    }
}

extern "C" void launch_hypic_cached_compose_v2(
    uint32_t blocks, void *stream, void *zero, void *transition, void *packet,
    void *output, void *workspace, void *tiling, uint32_t batch, uint32_t heads,
    uint32_t stride, uint32_t pool_slots, uint32_t rows, uint64_t system_workspace_bytes)
{
    // The Matmul mixed kernel launches two AIV workers per AIC.
    hypic_cached_compose_kernel_v2<<<blocks, nullptr, stream>>>(
        static_cast<GM_ADDR>(zero), static_cast<GM_ADDR>(transition), static_cast<GM_ADDR>(packet),
        static_cast<GM_ADDR>(output), static_cast<GM_ADDR>(workspace), static_cast<GM_ADDR>(tiling),
        batch, heads, stride, pool_slots, blocks * 2, rows, system_workspace_bytes);
}
