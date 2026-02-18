/* Copyright (c) 2024-2026 LunarG, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <vulkan/vulkan.h>
#include <stdint.h>
#include "pass.h"
#include <map>

namespace gpuav {
namespace spirv {

class SharedMemoryDataRacePass : public Pass {
  public:
    SharedMemoryDataRacePass(Module& module, const vvl::span<const uint32_t>& input_spirv, const VkPhysicalDeviceProperties& phys_dev_props);
    const char* Name() const final { return "SharedMemoryDataRacePass"; }
    bool Instrument() final;
    void PrintDebugInfo() const final;

  private:
    // This is metadata tied to a single instruction gathered during RequiresInstrumentation() to be used later
    struct InstructionMeta {
        const Instruction* target_instruction = nullptr;
        uint32_t function_idx;
        uint32_t access_chain_idx_id;
        uint32_t start_id;
        uint32_t num_elements;
    };

    bool RequiresInstrumentation(const Function& function, BasicBlock &block, InstructionIt& inst_it, const Instruction& inst, InstructionMeta& meta);
    uint32_t CreateFunctionCall(BasicBlock& block, InstructionIt* inst_it, const InstructionMeta& meta);

    uint32_t GetLinkFunctionId(const InstructionMeta& meta);

    // Function IDs to link in
    uint32_t link_function_id_[4] {};
    std::map<const Variable*, uint32_t> slot_start;
    uint32_t num_slots {};
    const vvl::span<const uint32_t>& input_spirv;
    const VkPhysicalDeviceProperties& phys_dev_props;
};

}  // namespace spirv
}  // namespace gpuav
