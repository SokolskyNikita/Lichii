// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

#include "shader_recompiler/frontend/ir/attribute.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/profile.h"

namespace Vulkan {

struct WorkgroupMemoryLayout {
    std::uint64_t size{};

    constexpr void Add(std::uint64_t bytes, std::uint64_t alignment) noexcept {
        size = ((size + alignment - 1) / alignment) * alignment + bytes;
    }
};

[[nodiscard]] inline std::uint64_t SharedAtomicExchange64LockSize(
    const Shader::IR::Program& program, const Shader::Profile& profile,
    bool mesh_as_compute) noexcept {
    using Shader::Stage;

    const bool has_non_block_workgroup_variables =
        program.stage == Stage::Mesh || (program.stage == Stage::Task && mesh_as_compute);
    const bool explicit_layout =
        !has_non_block_workgroup_variables && profile.support_explicit_workgroup_layout &&
        (!program.info.uses_int8 || profile.support_workgroup_layout_8bit_access) &&
        (!program.info.uses_int16 || profile.support_workgroup_layout_16bit_access);
    return program.info.uses_shared_atomic_exchange64 && profile.support_shared_int64_atomics &&
                   !explicit_layout
               ? sizeof(std::uint32_t)
               : 0;
}

[[nodiscard]] inline std::uint64_t MeshFallbackWorkgroupMemorySize(
    const Shader::IR::Program& program, const Shader::Profile& profile) {
    using Shader::Stage;

    if (program.stage != Stage::Task && program.stage != Stage::Mesh) {
        return 0;
    }

    WorkgroupMemoryLayout layout;
    layout.Add(static_cast<std::uint64_t>((std::max)(program.mesh.payload_words, 1U)) *
                   sizeof(std::uint32_t),
               sizeof(std::uint32_t));
    if (program.stage == Stage::Mesh) {
        const Shader::MeshInfo& mesh{program.mesh};
        const auto div_ceil = [](std::uint64_t value, std::uint64_t divisor) {
            return (value + divisor - 1) / divisor;
        };
        const std::uint64_t vertex_lines{mesh.vertex_lines.size()};
        const std::uint64_t primitive_lines{mesh.primitive_lines.size()};
        const std::uint64_t attribute_words{
            (std::max)(div_ceil(mesh.max_vertices, 32) * vertex_lines * 32 +
                           div_ceil(mesh.max_primitives, 32) * primitive_lines * 32,
                       std::uint64_t{1})};
        const std::uint64_t map_elements{
            4 + static_cast<std::uint64_t>(mesh.PrimitiveVertices()) * mesh.max_primitives};
        layout.Add(attribute_words * sizeof(std::uint32_t), sizeof(std::uint32_t));
        layout.Add(map_elements * sizeof(std::uint32_t), sizeof(std::uint32_t));
        layout.Add(sizeof(std::uint32_t), sizeof(std::uint32_t));
    }

    if (program.shared_memory_size != 0) {
        const std::uint64_t shared_size{
            (static_cast<std::uint64_t>(program.shared_memory_size) + 3) & ~std::uint64_t{3}};
        layout.Add(shared_size, sizeof(std::uint32_t));
    }
    const std::uint64_t lock_size{SharedAtomicExchange64LockSize(program, profile, true)};
    if (lock_size != 0) {
        layout.Add(lock_size, alignof(std::uint32_t));
    }
    return layout.size;
}

/// Threadgroup memory a native mesh program takes once MoltenVK has turned its SPIR-V into MSL,
/// by part. Metal fails the pipeline when the total exceeds maxThreadgroupMemoryLength (32 KiB
/// on Apple GPUs). It counts every `threadgroup` variable the MSL entry point declares:
///  - Lichii's own workgroup variables (EmitContext::DefineMesh / DefineSharedMemory): the ISBE
///    output region (isbe_attributes), the map region (isbe_map), the primitive count, and the
///    guest's shared memory (plus a lock word for 64-bit shared exchange);
///  - SPIRV-Cross's staged mesh outputs: Metal's mesh API takes outputs per vertex/primitive
///    through set_vertex/set_primitive, so SPIRV-Cross declares every Output array as a
///    threadgroup array and copies it out at the end. Every generic is a float4 (16 B) per vertex
///    or primitive, Position a float4, PointSize a float, per-primitive builtins a uint each, and
///    the index array one uint/uint2/uint3 (4/8/16 B) per primitive;
///  - spvMeshSizes (uint2) that SPIRV-Cross adds for SetMeshOutputsEXT.
/// The task payload is object_data in MSL, not threadgroup memory, so it is not counted.
struct NativeMeshThreadgroupMemory {
    std::uint64_t isbe{};    ///< isbe_attributes + isbe_map + isbe_primitive_count
    std::uint64_t outputs{}; ///< SPIRV-Cross's staged Output arrays, with the index array
    std::uint64_t shared{};  ///< guest shared memory and its atomic lock
    std::uint64_t fixed{};   ///< spvMeshSizes

    [[nodiscard]] constexpr std::uint64_t Total() const noexcept {
        return isbe + outputs + shared + fixed;
    }
};

[[nodiscard]] inline NativeMeshThreadgroupMemory NativeMeshThreadgroupMemorySize(
    const Shader::IR::Program& program, const Shader::Profile& profile) {
    NativeMeshThreadgroupMemory result;
    if (program.stage != Shader::Stage::Mesh) {
        return result;
    }
    const Shader::MeshInfo& mesh{program.mesh};
    const auto div_ceil = [](std::uint64_t value, std::uint64_t divisor) {
        return (value + divisor - 1) / divisor;
    };
    constexpr std::uint64_t word{sizeof(std::uint32_t)};
    // DefineMesh: the same arrays the emulation uses (MeshFallbackWorkgroupMemorySize).
    const std::uint64_t attribute_words{(std::max)(
        div_ceil(mesh.max_vertices, 32) * mesh.vertex_lines.size() * 32 +
            div_ceil(mesh.max_primitives, 32) * mesh.primitive_lines.size() * 32,
        std::uint64_t{1})};
    const std::uint64_t map_elements{
        4 + static_cast<std::uint64_t>(mesh.PrimitiveVertices()) * mesh.max_primitives};
    result.isbe = (attribute_words + map_elements + 1) * word;

    // DefineMesh's Output variables, one per generic index (per vertex and per primitive), as
    // SPIRV-Cross stages them.
    const std::uint64_t vertices{mesh.max_vertices};
    const std::uint64_t primitives{mesh.max_primitives};
    std::uint64_t outputs{};
    std::array<bool, Shader::IR::NUM_GENERICS> vertex_generic{}, primitive_generic{};
    bool position{}, point_size{mesh.PrimitiveVertices() == 1};
    bool primitive_id{}, layer{}, viewport{};
    for (const std::uint32_t line : mesh.vertex_lines) {
        const Shader::IR::Attribute attribute{line};
        if (Shader::IR::IsGeneric(attribute)) {
            vertex_generic[Shader::IR::GenericAttributeIndex(attribute)] = true;
        } else if (attribute >= Shader::IR::Attribute::PositionX &&
                   attribute <= Shader::IR::Attribute::PositionW) {
            position = true;
        } else if (attribute == Shader::IR::Attribute::PointSize) {
            point_size = true;
        }
    }
    for (const std::uint32_t line : mesh.primitive_lines) {
        const Shader::IR::Attribute attribute{line};
        if (Shader::IR::IsGeneric(attribute)) {
            primitive_generic[Shader::IR::GenericAttributeIndex(attribute)] = true;
        } else if (attribute == Shader::IR::Attribute::PrimitiveId) {
            primitive_id = true;
        } else if (attribute == Shader::IR::Attribute::Layer) {
            layer = true;
        } else if (attribute == Shader::IR::Attribute::ViewportIndex) {
            viewport = true;
        }
    }
    for (std::size_t index = 0; index < vertex_generic.size(); ++index) {
        outputs += (vertex_generic[index] ? vertices * 16 : 0) +
                   (primitive_generic[index] ? primitives * 16 : 0);
    }
    outputs += (position ? vertices * 16 : 0) + (point_size ? vertices * 4 : 0);
    outputs += (std::uint64_t{primitive_id} + layer + viewport) * primitives * 4;
    const std::uint32_t primitive_vertices{mesh.PrimitiveVertices()};
    outputs += primitives * (primitive_vertices == 1 ? 4 : primitive_vertices == 2 ? 8 : 16);
    result.outputs = outputs;

    // DefineSharedMemory without an explicit layout (always so for mesh programs).
    if (program.shared_memory_size != 0) {
        result.shared = div_ceil(program.shared_memory_size, word) * word +
                        SharedAtomicExchange64LockSize(program, profile, false);
    }
    result.fixed = 2 * word; // spvMeshSizes
    return result;
}

// The mesh fallback adds storage-buffer bindings for its output (mesh-only), or
// output, task payload, offsets and indirect arguments (task + mesh). Check
// against the final descriptor set limit without overflowing guest + auxiliary.
[[nodiscard]] constexpr bool FitsStorageBufferSetLimit(std::size_t guest_buffers,
                                                       std::size_t auxiliary_buffers,
                                                       std::size_t set_limit) noexcept {
    return set_limit == 0 ||
           (guest_buffers <= set_limit && auxiliary_buffers <= set_limit - guest_buffers);
}

} // namespace Vulkan
