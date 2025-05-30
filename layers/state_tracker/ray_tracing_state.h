/* Copyright (c) 2015-2025 The Khronos Group Inc.
 * Copyright (c) 2015-2025 Valve Corporation
 * Copyright (c) 2015-2025 LunarG, Inc.
 * Copyright (C) 2015-2025 Google Inc.
 * Modifications Copyright (C) 2020 Advanced Micro Devices, Inc. All rights reserved.
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
#include "state_tracker/device_memory_state.h"
#include "state_tracker/buffer_state.h"
#include "generated/dispatch_functions.h"

namespace vvl {
class AccelerationStructureNVSubState;

class AccelerationStructureNV : public Bindable, public SubStateManager<AccelerationStructureNVSubState> {
  public:
    AccelerationStructureNV(VkDevice device, VkAccelerationStructureNV handle,
                            const VkAccelerationStructureCreateInfoNV *pCreateInfo)
        : Bindable(handle, kVulkanObjectTypeAccelerationStructureNV, false, false, 0),
          safe_create_info(pCreateInfo),
          create_info(*safe_create_info.ptr()),
          memory_requirements(GetMemReqs(device, handle, VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_OBJECT_NV)),
          build_scratch_memory_requirements(
              GetMemReqs(device, handle, VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_BUILD_SCRATCH_NV)),
          update_scratch_memory_requirements(
              GetMemReqs(device, handle, VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_UPDATE_SCRATCH_NV)),
          tracker_(&memory_requirements) {
        Bindable::SetMemoryTracker(&tracker_);
    }
    AccelerationStructureNV(const AccelerationStructureNV &rh_obj) = delete;

    VkAccelerationStructureNV VkHandle() const { return handle_.Cast<VkAccelerationStructureNV>(); }

    void Destroy() override;
    void NotifyInvalidate(const StateObject::NodeList &invalid_nodes, bool unlink) override;

    void Build(const VkAccelerationStructureInfoNV *pInfo) {
        built = true;
        build_info.initialize(pInfo);
    };

    const vku::safe_VkAccelerationStructureCreateInfoNV safe_create_info;
    const VkAccelerationStructureCreateInfoNV &create_info;

    vku::safe_VkAccelerationStructureInfoNV build_info;
    const VkMemoryRequirements memory_requirements;
    const VkMemoryRequirements build_scratch_memory_requirements;
    const VkMemoryRequirements update_scratch_memory_requirements;
    uint64_t opaque_handle = 0;
    bool memory_requirements_checked = false;
    bool build_scratch_memory_requirements_checked = false;
    bool update_scratch_memory_requirements_checked = false;
    bool built = false;

  private:
    static VkMemoryRequirements GetMemReqs(VkDevice device, VkAccelerationStructureNV as,
                                           VkAccelerationStructureMemoryRequirementsTypeNV mem_type) {
        VkAccelerationStructureMemoryRequirementsInfoNV req_info = vku::InitStructHelper();
        req_info.type = mem_type;
        req_info.accelerationStructure = as;
        VkMemoryRequirements2 requirements = vku::InitStructHelper();
        DispatchGetAccelerationStructureMemoryRequirementsNV(device, &req_info, &requirements);
        return requirements.memoryRequirements;
    }
    BindableLinearMemoryTracker tracker_;
};

class AccelerationStructureNVSubState {
  public:
    explicit AccelerationStructureNVSubState(AccelerationStructureNV &ac) : base(ac) {}
    AccelerationStructureNVSubState(const AccelerationStructureNVSubState &) = delete;
    AccelerationStructureNVSubState &operator=(const AccelerationStructureNVSubState &) = delete;
    virtual ~AccelerationStructureNVSubState() {}
    virtual void Destroy() {}
    virtual void NotifyInvalidate(const StateObject::NodeList &invalid_nodes, bool unlink) {}

    AccelerationStructureNV &base;
};

inline void AccelerationStructureNV::Destroy() {
    for (auto &item : sub_states_) {
        item.second->Destroy();
    }
    Bindable::Destroy();
}

inline void AccelerationStructureNV::NotifyInvalidate(const StateObject::NodeList &invalid_nodes, bool unlink) {
    for (auto &item : sub_states_) {
        item.second->NotifyInvalidate(invalid_nodes, unlink);
    }
    Bindable::NotifyInvalidate(invalid_nodes, unlink);
}

class AccelerationStructureKHRSubState;

class AccelerationStructureKHR : public StateObject, public SubStateManager<AccelerationStructureKHRSubState> {
  public:
    AccelerationStructureKHR(VkAccelerationStructureKHR handle, const VkAccelerationStructureCreateInfoKHR *pCreateInfo,
                             std::shared_ptr<Buffer> &&buf_state, const VkDeviceAddress buffer_device_address)
        : StateObject(handle, kVulkanObjectTypeAccelerationStructureKHR),
          create_info(*pCreateInfo),
          buffer_state(buf_state),
          buffer_device_address(buffer_device_address) {}
    AccelerationStructureKHR(const AccelerationStructureKHR &rh_obj) = delete;

    virtual ~AccelerationStructureKHR() {
        if (!Destroyed()) {
            Destroy();
        }
    }

    VkAccelerationStructureKHR VkHandle() const { return handle_.Cast<VkAccelerationStructureKHR>(); }

    void LinkChildNodes() override {
        // Connect child node(s), which cannot safely be done in the constructor.
        buffer_state->AddParent(this);
    }

    void Destroy() override;
    void NotifyInvalidate(const StateObject::NodeList &invalid_nodes, bool unlink) override;

    void Build(const VkAccelerationStructureBuildGeometryInfoKHR *pInfo, const bool is_host,
               const VkAccelerationStructureBuildRangeInfoKHR *build_range_info) {
        is_built = true;
        if (!build_info_khr.has_value()) {
            build_info_khr = vku::safe_VkAccelerationStructureBuildGeometryInfoKHR();
        }
        build_info_khr->initialize(pInfo, is_host, build_range_info);
    };

    void UpdateBuildRangeInfos(const VkAccelerationStructureBuildRangeInfoKHR *p_build_range_infos, uint32_t geometry_count) {
        build_range_infos.resize(geometry_count);
        for (const auto [i, build_range] : vvl::enumerate(p_build_range_infos, geometry_count)) {
            build_range_infos[i] = build_range;
        }
    }

    // Returns the device address range effectively occupied by the acceleration structure,
    // as defined by its creation info.
    // It does NOT take into account the acceleration structure address as returned by
    // vkGetAccelerationStructureDeviceAddress, this address may be at an offset
    // of the buffer range backing the acceleration structure
    vvl::range<VkDeviceAddress> GetDeviceAddressRange() const {
        if (!buffer_state) {
            return {};
        }
        if (buffer_state->deviceAddress != 0) {
            return {buffer_state->deviceAddress + create_info.offset,
                    buffer_state->deviceAddress + create_info.offset + create_info.size};
        }
        return {buffer_device_address + create_info.offset, buffer_device_address + create_info.offset + create_info.size};
    }

    // At time of writing, havin a safe_VkAccelerationStructureCreateInfoKHR is not strictly necessary,
    // and https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/9669
    // showed that the underlying used to store host side acceleration structure
    // data seems to have hard to reproduce issues
    // => rely on a plain VkAccelerationStructureCreateInfoKHR
    VkAccelerationStructureCreateInfoKHR create_info;

    uint64_t opaque_handle = 0;
    std::shared_ptr<vvl::Buffer> buffer_state{};
    // Used in case buffer_state->deviceAddress is 0 (happens if app never queried address)
    const VkDeviceAddress buffer_device_address = 0;
    std::optional<vku::safe_VkAccelerationStructureBuildGeometryInfoKHR> build_info_khr{};
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> build_range_infos{};
    // You can't have is_built == false and a build_info_khr, but you can have is_built == true and no build_info_khr,
    // if the acceleration structure was filled by a call to vkCmdCopyMemoryToAccelerationStructure
    bool is_built = false;
};

class AccelerationStructureKHRSubState {
  public:
    explicit AccelerationStructureKHRSubState(AccelerationStructureKHR &ac) : base(ac) {}
    AccelerationStructureKHRSubState(const AccelerationStructureKHRSubState &) = delete;
    AccelerationStructureKHRSubState &operator=(const AccelerationStructureKHRSubState &) = delete;
    virtual ~AccelerationStructureKHRSubState() {}
    virtual void Destroy() {}
    virtual void NotifyInvalidate(const StateObject::NodeList &invalid_nodes, bool unlink) {}

    AccelerationStructureKHR &base;
};

inline void AccelerationStructureKHR::Destroy() {
    for (auto &item : sub_states_) {
        item.second->Destroy();
    }
    if (buffer_state) {
        buffer_state->RemoveParent(this);
        buffer_state = nullptr;
    }
    StateObject::Destroy();
}

inline void AccelerationStructureKHR::NotifyInvalidate(const StateObject::NodeList &invalid_nodes, bool unlink) {
    for (auto &item : sub_states_) {
        item.second->NotifyInvalidate(invalid_nodes, unlink);
    }
    StateObject::NotifyInvalidate(invalid_nodes, unlink);
}

}  // namespace vvl
