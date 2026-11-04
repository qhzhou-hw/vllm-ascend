// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <string>
#include <vector>

extern "C" void launch_hypic_cached_compose_v2(
    uint32_t, void *, void *, void *, void *, void *, void *, void *,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint64_t);
extern "C" void launch_hypic_tree_level(
    uint32_t, void *, void *, void *, void *, void *, void *, void *, void *,
    uint32_t, uint32_t, uint32_t, uint32_t);
extern "C" void launch_hypic_tree_single(
    uint32_t, void *, void *, void *, void *, void *, void *, void *, void *,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
extern "C" void launch_hypic_tree_hybrid(
    uint32_t, void *, void *, void *, void *, void *, void *, void *, void *,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    void *, void *, uint32_t, uint32_t, uint64_t);

namespace {
constexpr int64_t DIM = 128;
constexpr int64_t INT_LIMIT = std::numeric_limits<int32_t>::max();

std::tuple<at::Tensor, int64_t, int64_t, int64_t> make_matmul_tiling(
    int64_t rows, int64_t independent_heads, bool global_a, int64_t reserve_rows = 0)
{
    TORCH_CHECK(rows == 0 || rows == 32 || rows == 64 || rows == 128, "rows must be 0, 32, 64 or 128");
    TORCH_CHECK(independent_heads >= 0 && independent_heads <= INT_LIMIT, "invalid independent head count");
    const char *soc = aclrtGetSocName();
    TORCH_CHECK(soc && std::string(soc).find("Ascend910B") == 0,
                "cached compose Ascend C prototype requires Ascend910B");
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance(soc);
    TORCH_CHECK(platform, "CANN platform lookup failed");
    const int64_t cores = platform->GetCoreNumAic();
    TORCH_CHECK(cores > 0, "CANN reported no Cube cores");
    if (rows == 0) {
        // Keep small batches parallel; enlarge tiles only to reduce worker waves.
        rows = 32;
        while (rows < DIM && independent_heads * (DIM / rows) > 2 * cores) {
            rows *= 2;
        }
    }
    matmul_tiling::MatmulApiTiling api(*platform);
    api.SetAType(global_a ? matmul_tiling::TPosition::GM : matmul_tiling::TPosition::VECOUT,
                 matmul_tiling::CubeFormat::ND,
                 matmul_tiling::DataType::DT_FLOAT);
    api.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                 matmul_tiling::DataType::DT_FLOAT);
    api.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                 matmul_tiling::DataType::DT_FLOAT);
    api.SetShape(rows, DIM, DIM);
    api.SetOrgShape(rows, DIM, DIM);
    api.SetFixSplit(rows, DIM, -1);
    api.SetBias(false);
    api.SetMadType(matmul_tiling::MatrixMadType::NORMAL);  // Never HF32.
    uint64_t ub_bytes = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_bytes);
    const uint64_t local_bytes = 2 * std::max(rows, reserve_rows) * DIM * sizeof(float);
    TORCH_CHECK(ub_bytes > local_bytes, "insufficient UB for cached compose");
    api.SetBufferSpace(-1, -1, ub_bytes - local_bytes);
    optiling::TCubeTiling tiling;
    TORCH_CHECK(api.GetTiling(tiling) != -1, "FP32 cached compose Matmul tiling failed");
    TORCH_CHECK(tiling.get_baseM() == rows && tiling.get_baseN() == DIM,
                "unexpected Matmul output tile");
    auto data = at::empty({static_cast<int64_t>(tiling.GetDataSize())}, at::TensorOptions().dtype(at::kByte));
    tiling.SaveToBuffer(data.data_ptr(), data.numel());
    return {data, platform->GetLibApiWorkSpaceSize(), cores, rows};
}

auto make_tiling(int64_t rows, int64_t independent_heads)
{
    return make_matmul_tiling(rows, independent_heads, false);
}

auto make_tree_tiling(int64_t rows, int64_t independent_heads)
{
    return make_matmul_tiling(rows, independent_heads, true);
}

auto make_chain_tiling(int64_t rows, int64_t independent_heads, int64_t reserve_rows)
{
    TORCH_CHECK(reserve_rows == 32 || reserve_rows == 64 || reserve_rows == 128, "invalid chain UB reservation");
    return make_matmul_tiling(rows, independent_heads, false, reserve_rows);
}

void check_pool(const at::Tensor &s, const at::Tensor &t)
{
    TORCH_CHECK(s.device().type() == c10::DeviceType::PrivateUse1 && t.device() == s.device(),
                "S/T must be on the same NPU");
    TORCH_CHECK(s.scalar_type() == at::kFloat && t.scalar_type() == at::kFloat &&
                s.is_contiguous() && t.is_contiguous(), "S/T must be contiguous FP32");
    TORCH_CHECK(s.dim() == 4 && s.size(1) > 0 && s.size(2) == DIM && s.size(3) == DIM &&
                t.sizes() == s.sizes(), "expected S/T [slots, heads, 128, 128]");
    TORCH_CHECK(s.size(0) <= INT_LIMIT && s.size(1) <= INT_LIMIT, "pool dimensions exceed int32");
}

at::Tensor run(const at::Tensor &s, const at::Tensor &t, const at::Tensor &packet,
               const at::Tensor &tiling, int64_t workspace_bytes, int64_t max_blocks, int64_t rows)
{
    TORCH_CHECK(rows == 32 || rows == 64 || rows == 128, "rows must be 32, 64 or 128");
    check_pool(s, t);
    TORCH_CHECK(packet.device() == s.device() && packet.scalar_type() == at::kInt &&
                packet.is_contiguous() && packet.dim() == 2 && packet.size(1) >= 1 &&
                packet.size(1) <= INT_LIMIT, "invalid cached compose packet");
    optiling::TCubeTiling expected_tiling;
    TORCH_CHECK(tiling.device() == s.device() && tiling.scalar_type() == at::kByte &&
                tiling.is_contiguous() && tiling.dim() == 1 && tiling.numel() == expected_tiling.GetDataSize(),
                "invalid Matmul tiling tensor");
    const int64_t batch = packet.size(0), heads = s.size(1);
    TORCH_CHECK(batch <= INT_LIMIT / heads / (DIM / rows), "task count exceeds int32");
    TORCH_CHECK(max_blocks > 0 && max_blocks <= 64 && workspace_bytes > 0,
                "invalid Matmul workspace configuration");
    c10_npu::NPUGuard guard(s.device());
    auto output = at::empty({batch, heads, DIM, DIM}, s.options());
    if (batch == 0) {
        return output;
    }
    const auto blocks = static_cast<uint32_t>(std::min(max_blocks, (batch * heads * (DIM / rows) + 1) / 2));
    const int64_t scratch_bytes = blocks * 2 * rows * DIM * sizeof(float);
    TORCH_CHECK(workspace_bytes <= std::numeric_limits<int64_t>::max() - scratch_bytes,
                "workspace size overflow");
    auto workspace = at::empty({workspace_bytes + scratch_bytes}, s.options().dtype(at::kByte));
    const auto npu_stream = c10_npu::getCurrentNPUStream();
    auto stream = npu_stream.stream();
    const std::array<at::Tensor, 6> tensors{s, t, packet, output, workspace, tiling};
    std::array<void *, 6> addresses;
    for (size_t i = 0; i < tensors.size(); ++i) {
        addresses[i] = tensors[i].data_ptr();
        if (tensors[i].numel()) {
            c10_npu::NPUCachingAllocator::recordStream(tensors[i].storage().data_ptr(), npu_stream);
        }
    }
    const auto stride = static_cast<uint32_t>(packet.size(1));
    const auto slots = static_cast<uint32_t>(s.size(0));
    at_npu::native::OpCommand command;
    command.Name("hypic_cached_compose_ascendc");
    // Match the repository's direct-launch adapters: only POD launch arguments
    // in the queued callback. Keep tensor destruction on the caller thread;
    // recordStream and stream-ordered allocator reuse protect asynchronous work.
    command.SetCustomHandler([addresses, blocks, stream, batch, heads, stride, slots, rows, workspace_bytes]() -> int {
        launch_hypic_cached_compose_v2(blocks, stream, addresses[0], addresses[1],
            addresses[2], addresses[3], addresses[4], addresses[5],
            batch, heads, stride, slots, rows,
            workspace_bytes);
        return 0;
    });
    command.Run();
    return output;
}

struct TreeLaunch {
    void *si, *ti, *packet, *so, *to, *tiling;
    uint32_t nodes, slots, rows, blocks;
};

at::Tensor run_tree_single(const at::Tensor &s, const at::Tensor &t,
    const at::Tensor &packet, const at::Tensor &tiling, const std::vector<int64_t> &nodes,
    const std::vector<int64_t> &rows, int64_t workspace_bytes, int64_t max_blocks,
    const std::optional<at::Tensor> &tail_packet, const std::optional<at::Tensor> &tail_tiling, int64_t tail_rows)
{
    check_pool(s, t);
    TORCH_CHECK(!nodes.empty() && nodes.size() <= 32 && rows.size() == nodes.size(),
                "invalid single-kernel tree levels");
    c10_npu::NPUGuard guard(s.device());
    const char *soc = aclrtGetSocName();
    TORCH_CHECK(soc && std::string(soc).find("Ascend910B") == 0, "single tree requires Ascend910B");
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance(soc);
    TORCH_CHECK(platform && max_blocks > 0 && max_blocks <= platform->GetCoreNumAic() && workspace_bytes > 0,
                "single tree workers must fit physical cores");
    std::array<int64_t, 2> capacities{0, 0};
    int64_t records = nodes.size(), max_tasks = 0;
    const auto heads = s.size(1);
    for (size_t i = 0; i < nodes.size(); ++i) {
        TORCH_CHECK(rows[i] == 32 || rows[i] == 64 || rows[i] == 128, "invalid single tree row tile");
        TORCH_CHECK(rows[i] <= rows.front() && nodes[i] >= 0 && nodes[i] <= INT_LIMIT / heads / (DIM / 32),
                    "invalid single tree shape");
        if (i) TORCH_CHECK(nodes[i] <= nodes[i - 1], "single tree node counts must not grow");
        capacities[i % 2] = std::max(capacities[i % 2], nodes[i]);
        records += nodes[i];
        max_tasks = std::max(max_tasks, nodes[i] * heads * (DIM / rows[i]));
    }
    TORCH_CHECK(records <= INT_LIMIT / 3 && packet.device() == s.device() &&
                packet.scalar_type() == at::kInt && packet.is_contiguous() &&
                packet.dim() == 2 && packet.size(0) == records && packet.size(1) == 3,
                "invalid single tree packet");
    optiling::TCubeTiling expected_tiling;
    TORCH_CHECK(tiling.device() == s.device() && tiling.scalar_type() == at::kByte &&
                tiling.is_contiguous() && tiling.dim() == 1 && tiling.numel() == expected_tiling.GetDataSize(),
                "invalid single tree tiling");
    const bool has_tail = tail_packet.has_value();
    TORCH_CHECK(has_tail == tail_tiling.has_value(), "single tree tail packet/tiling must be provided together");
    int64_t batch = nodes.back(), tail_stride = 0;
    if (has_tail) {
        const auto &p = *tail_packet, &tt = *tail_tiling;
        TORCH_CHECK(p.device() == s.device() && p.scalar_type() == at::kInt && p.is_contiguous() &&
                    p.dim() == 2 && p.size(0) > 0 && p.size(0) <= nodes.back() && p.size(1) >= 1 &&
                    p.size(1) <= INT_LIMIT, "invalid single tree tail packet");
        TORCH_CHECK(tt.device() == s.device() && tt.scalar_type() == at::kByte && tt.is_contiguous() &&
                    tt.dim() == 1 && tt.numel() == expected_tiling.GetDataSize(), "invalid single tree tail tiling");
        TORCH_CHECK((tail_rows == 32 || tail_rows == 64 || tail_rows == 128) && tail_rows <= rows.front(),
                    "invalid single tree tail rows");
        batch = p.size(0);
        tail_stride = p.size(1);
        max_tasks = std::max(max_tasks, batch * heads * (DIM / tail_rows));
    } else {
        TORCH_CHECK(tail_rows == 0, "tail rows require a tail packet");
    }
    auto output = at::empty({batch, heads, DIM, DIM}, s.options());
    if (batch == 0) {
        TORCH_CHECK(max_tasks == 0, "nonempty intermediate tree for empty output");
        return output;
    }
    auto buffers = at::empty({2 * (capacities[0] + capacities[1]), heads, DIM, DIM}, s.options());
    const auto blocks = static_cast<uint32_t>(std::min(max_blocks, (max_tasks + 1) / 2));
    const int64_t scratch_bytes = static_cast<int64_t>(blocks) * 2 * tail_rows * DIM * sizeof(float);
    TORCH_CHECK(workspace_bytes <= std::numeric_limits<int64_t>::max() - scratch_bytes, "single tree workspace overflow");
    auto workspace = at::empty({workspace_bytes + scratch_bytes}, s.options().dtype(at::kByte));
    const auto npu_stream = c10_npu::getCurrentNPUStream();
    const auto stream = npu_stream.stream();
    const std::array<at::Tensor, 7> tensors{s, t, packet, output, buffers, workspace, tiling};
    std::array<void *, 7> addresses;
    for (size_t i = 0; i < tensors.size(); ++i) {
        addresses[i] = tensors[i].data_ptr();
        if (tensors[i].numel()) c10_npu::NPUCachingAllocator::recordStream(tensors[i].storage().data_ptr(), npu_stream);
    }
    const auto levels = static_cast<uint32_t>(nodes.size());
    const auto slots = static_cast<uint32_t>(s.size(0));
    const auto max_rows = static_cast<uint32_t>(rows.front());
    void *tail_packet_ptr = nullptr, *tail_tiling_ptr = nullptr;
    if (has_tail) {
        tail_packet_ptr = tail_packet->data_ptr();
        tail_tiling_ptr = tail_tiling->data_ptr();
        for (const auto &tensor : {*tail_packet, *tail_tiling}) {
            if (tensor.numel()) c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), npu_stream);
        }
    }
    at_npu::native::OpCommand command;
    command.Name("hypic_cached_compose_tree_single");
    command.SetCustomHandler([addresses, blocks, stream, levels, records, batch, heads, slots, capacities, max_rows,
                             has_tail, tail_packet_ptr, tail_tiling_ptr, tail_stride, tail_rows, workspace_bytes]() -> int {
        if (has_tail) {
            launch_hypic_tree_hybrid(blocks, stream, addresses[0], addresses[1], addresses[2], addresses[3],
                addresses[4], addresses[5], addresses[6], levels, records, batch, heads, slots,
                capacities[0], capacities[1], max_rows, tail_packet_ptr, tail_tiling_ptr, tail_stride, tail_rows, workspace_bytes);
        } else {
            launch_hypic_tree_single(blocks, stream, addresses[0], addresses[1], addresses[2], addresses[3],
                addresses[4], addresses[5], addresses[6], levels, records, batch, heads, slots,
                capacities[0], capacities[1], max_rows);
        }
        return 0;
    });
    command.Run();
    return output;
}

at::Tensor run_tree(const at::Tensor &s, const at::Tensor &t,
    const std::vector<at::Tensor> &packets, const std::vector<at::Tensor> &tilings,
    const std::vector<int64_t> &row_tiles, int64_t workspace_bytes, int64_t max_blocks,
    const at::Tensor &tail_packet, const at::Tensor &tail_tiling, int64_t tail_rows)
{
    check_pool(s, t);
    TORCH_CHECK(!packets.empty() && packets.size() <= 32 && tilings.size() == packets.size() &&
                row_tiles.size() == packets.size(), "invalid tree level metadata");
    TORCH_CHECK(workspace_bytes > 0 && max_blocks > 0 && max_blocks <= 64, "invalid tree workspace");
    optiling::TCubeTiling expected_tiling;
    std::array<int64_t, 2> capacities{0, 0};
    for (size_t i = 0; i < packets.size(); ++i) {
        const auto &p = packets[i];
        const auto &tiling = tilings[i];
        const auto rows = row_tiles[i];
        TORCH_CHECK(rows == 32 || rows == 64 || rows == 128, "invalid tree row tile");
        TORCH_CHECK(p.device() == s.device() && p.scalar_type() == at::kInt && p.is_contiguous() &&
                    p.dim() == 2 && p.size(1) == 3 && p.size(0) <= INT_LIMIT / s.size(1) / (DIM / rows),
                    "invalid tree packet");
        TORCH_CHECK(tiling.device() == s.device() && tiling.scalar_type() == at::kByte &&
                    tiling.is_contiguous() && tiling.dim() == 1 &&
                    tiling.numel() == expected_tiling.GetDataSize(), "invalid tree tiling");
        capacities[i % 2] = std::max(capacities[i % 2], p.size(0));
    }
    c10_npu::NPUGuard guard(s.device());
    TORCH_CHECK(tail_packet.device() == s.device() && tail_packet.scalar_type() == at::kInt &&
                tail_packet.is_contiguous() && tail_packet.dim() == 2 && tail_packet.size(1) >= 1 &&
                tail_packet.size(1) <= INT_LIMIT, "invalid tree tail packet");
    const bool has_tail = tail_packet.size(0) > 0;
    const auto batch = has_tail ? tail_packet.size(0) : packets.back().size(0);
    const auto heads = s.size(1);
    uint32_t tail_blocks = 0;
    if (has_tail) {
        TORCH_CHECK(tail_rows == 32 || tail_rows == 64 || tail_rows == 128, "invalid tree tail rows");
        TORCH_CHECK(batch <= INT_LIMIT / heads / (DIM / tail_rows), "tree tail task count exceeds int32");
        TORCH_CHECK(tail_tiling.device() == s.device() && tail_tiling.scalar_type() == at::kByte &&
                    tail_tiling.is_contiguous() && tail_tiling.dim() == 1 &&
                    tail_tiling.numel() == expected_tiling.GetDataSize(), "invalid tree tail tiling");
        tail_blocks = static_cast<uint32_t>(std::min<int64_t>(max_blocks, (batch * heads * (DIM / tail_rows) + 1) / 2));
    }
    auto output = at::empty({batch, heads, DIM, DIM}, s.options());
    if (batch == 0) {
        TORCH_CHECK(capacities[0] == 0 && capacities[1] == 0, "nonempty intermediate tree for empty output");
        return output;
    }
    // Size each ping-pong bank for its own largest level. A one-level hybrid
    // needs only one bank; later levels never overwrite a still-consumed input.
    auto buffers = at::empty({2 * (capacities[0] + capacities[1]), heads, DIM, DIM}, s.options());
    const int64_t scratch_bytes = tail_blocks * 2 * tail_rows * DIM * sizeof(float);
    TORCH_CHECK(workspace_bytes <= std::numeric_limits<int64_t>::max() - scratch_bytes, "tree workspace overflow");
    auto workspace = at::empty({workspace_bytes + scratch_bytes}, s.options().dtype(at::kByte));
    const auto npu_stream = c10_npu::getCurrentNPUStream();
    const auto stream = npu_stream.stream();
    const auto matrix_size = heads * DIM * DIM;
    auto base = buffers.data_ptr<float>();
    void *si = s.data_ptr(), *ti = t.data_ptr();
    auto slots = static_cast<uint32_t>(s.size(0));
    std::vector<TreeLaunch> launches;
    for (size_t i = 0; i < packets.size(); ++i) {
        const auto nodes = static_cast<uint32_t>(packets[i].size(0));
        TORCH_CHECK(nodes > 0, "empty intermediate tree level");
        const auto rows = static_cast<uint32_t>(row_tiles[i]);
        const auto blocks = static_cast<uint32_t>(std::min<int64_t>(
            max_blocks, (static_cast<int64_t>(nodes) * heads * (DIM / rows) + 1) / 2));
        const auto bank_start = i % 2 ? 2 * capacities[0] * matrix_size : 0;
        void *so = !has_tail && i + 1 == packets.size() ? output.data_ptr() : base + bank_start;
        void *to = base + bank_start + capacities[i % 2] * matrix_size;
        launches.push_back({si, ti, packets[i].data_ptr(), so, to, tilings[i].data_ptr(),
                            nodes, slots, rows, blocks});
        si = so;
        ti = to;
        slots = nodes;
    }
    auto record = [&](const at::Tensor &tensor) {
        if (tensor.numel()) {
            c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), npu_stream);
        }
    };
    for (const auto &tensor : {s, t, output, buffers, workspace}) record(tensor);
    for (const auto &tensor : packets) record(tensor);
    for (const auto &tensor : tilings) record(tensor);
    record(tail_packet);
    record(tail_tiling);
    auto workspace_ptr = workspace.data_ptr();
    const std::array<void *, 5> tail_addresses{si, ti, tail_packet.data_ptr(), tail_tiling.data_ptr(), output.data_ptr()};
    const auto tail_stride = static_cast<uint32_t>(tail_packet.size(1));
    at_npu::native::OpCommand command;
    command.Name("hypic_cached_compose_tree");
    command.SetCustomHandler([launches, stream, workspace_ptr, heads, has_tail, tail_blocks,
                            tail_addresses, batch, tail_stride, slots, tail_rows, workspace_bytes]() -> int {
        for (const auto &level : launches) {
            launch_hypic_tree_level(level.blocks, stream, level.si, level.ti, level.packet,
                level.so, level.to, workspace_ptr, level.tiling, level.nodes, heads, level.slots, level.rows);
        }
        if (has_tail) {
            launch_hypic_cached_compose_v2(tail_blocks, stream, tail_addresses[0], tail_addresses[1],
                tail_addresses[2], tail_addresses[4], workspace_ptr, tail_addresses[3],
                batch, heads, tail_stride, slots, tail_rows, workspace_bytes);
        }
        return 0;
    });
    command.Run();
    return output;
}
}  // namespace

#define HYPIC_LIBRARY(ns, m) TORCH_LIBRARY(ns, m)
#define HYPIC_IMPL(ns, m) TORCH_LIBRARY_IMPL(ns, PrivateUse1, m)
HYPIC_LIBRARY(HYPIC_OP_NAMESPACE, m)
{
    m.def("make_tiling(int rows=0, int independent_heads=0) -> (Tensor, int, int, int)", &make_tiling);
    m.def("cached_compose(Tensor s, Tensor t, Tensor packet, Tensor tiling, int workspace_bytes, int max_blocks, int rows) -> Tensor");
    m.def("make_tree_tiling(int rows=0, int independent_heads=0) -> (Tensor, int, int, int)", &make_tree_tiling);
    m.def("cached_compose_tree(Tensor s, Tensor t, Tensor[] packets, Tensor[] tilings, int[] row_tiles, int workspace_bytes, int max_blocks, Tensor tail_packet, Tensor tail_tiling, int tail_rows) -> Tensor");
    m.def("make_chain_tiling(int rows, int independent_heads, int reserve_rows) -> (Tensor, int, int, int)", &make_chain_tiling);
    m.def("cached_compose_tree_single(Tensor s, Tensor t, Tensor packet, Tensor tiling, int[] nodes, int[] rows, int workspace_bytes, int max_blocks, Tensor? tail_packet=None, Tensor? tail_tiling=None, int tail_rows=0) -> Tensor");
}

HYPIC_IMPL(HYPIC_OP_NAMESPACE, m)
{
    m.impl("cached_compose", &run);
    m.impl("cached_compose_tree", &run_tree);
    m.impl("cached_compose_tree_single", &run_tree_single);
}
