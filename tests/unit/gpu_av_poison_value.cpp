/* Copyright (c) 2026 The Khronos Group Inc.
 * Copyright (c) 2026 Valve Corporation
 * Copyright (c) 2026 LunarG, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#include "../framework/layer_validation_tests.h"
#include "../framework/pipeline_helper.h"
#include "../framework/shader_helper.h"

class NegativeGpuAVPoisonValue : public GpuAVTest {
  protected:
    void InitPoisonValue();
    void SimpleComputeTest(const char* shader, const char* expected_error, uint32_t error_count = 1,
                           SpvSourceType source_type = SPV_SOURCE_GLSL);
};

void NegativeGpuAVPoisonValue::InitPoisonValue() {
    SetTargetApiVersion(VK_API_VERSION_1_1);
    std::vector<VkLayerSettingEXT> settings = {
        {OBJECT_LAYER_NAME, "gpuav_poison_value", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &kVkTrue},
        {OBJECT_LAYER_NAME, "gpuav_descriptor_checks", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &kVkFalse},
        {OBJECT_LAYER_NAME, "gpuav_buffer_address_oob", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &kVkFalse},
        {OBJECT_LAYER_NAME, "gpuav_ray_query", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &kVkFalse},
        {OBJECT_LAYER_NAME, "gpuav_shader_sanitizer", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &kVkFalse},
        {OBJECT_LAYER_NAME, "gpuav_shared_memory_data_race", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &kVkFalse},
        {OBJECT_LAYER_NAME, "gpuav_debug_validate_instrumented_shaders", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &kVkTrue},
        {OBJECT_LAYER_NAME, "gpuav_debug_dump_instrumented_shaders", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &kVkTrue},
    };
    RETURN_IF_SKIP(InitGpuAvFramework(settings, false));
    RETURN_IF_SKIP(InitState());
}

void NegativeGpuAVPoisonValue::SimpleComputeTest(const char* shader, const char* expected_error, uint32_t error_count,
                                                  SpvSourceType source_type) {
    InitPoisonValue();

    CreateComputePipelineHelper pipe(*this);
    pipe.dsl_bindings_[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL, nullptr};
    pipe.cs_ = VkShaderObj(*m_device, shader, VK_SHADER_STAGE_COMPUTE_BIT, SPV_ENV_VULKAN_1_1, source_type);
    pipe.CreateComputePipeline();

    vkt::Buffer in_buffer(*m_device, 256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemProps);
    void* in_ptr = in_buffer.Memory().Map();
    memset(in_ptr, 0, 256);

    pipe.descriptor_set_.WriteDescriptorBufferInfo(0, in_buffer, 0, VK_WHOLE_SIZE, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    pipe.descriptor_set_.UpdateDescriptorSets();

    m_command_buffer.Begin();
    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vk::CmdBindDescriptorSets(m_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline_layout_, 0, 1,
                              &pipe.descriptor_set_.set_, 0, nullptr);
    vk::CmdDispatch(m_command_buffer, 1, 1, 1);
    m_command_buffer.End();

    m_errorMonitor->SetDesiredError(expected_error, error_count);
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVPoisonValue, BranchOnPoison) {
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint output_val;
        };
        void main() {
            bool x;
            if (x) {
                output_val = 1;
            } else {
                output_val = 0;
            }
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-BranchOnPoison");
}

TEST_F(NegativeGpuAVPoisonValue, SelectPoisonSideSelected) {
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint cond;
            uint output_val;
        };
        void main() {
            uint x;
            uint y = 42;
            uint result = (cond == 0) ? x : y;
            output_val = result;
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ExternalStoreOfPoison");
}

TEST_F(NegativeGpuAVPoisonValue, PoisonPropagateArith) {
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint output_val;
        };
        void main() {
            uint x;
            uint y = x + 1;
            output_val = y;
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ExternalStoreOfPoison");
}

TEST_F(NegativeGpuAVPoisonValue, PoisonAccessChainIndex) {
    // Hand-written SPIR-V: OpAccessChain with a poison index.
    // The resulting pointer is never dereferenced, so only the access chain
    // error fires (each thread can only report one error).
    const char* cs_source = R"asm(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
               OpDecorate %SSBO Block
               OpMemberDecorate %SSBO 0 Offset 0
               OpDecorate %arr_type ArrayStride 4
               OpDecorate %ssbo_var DescriptorSet 0
               OpDecorate %ssbo_var Binding 0
       %void = OpTypeVoid
    %void_fn = OpTypeFunction %void
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
    %uint_42 = OpConstant %uint 42
    %uint_16 = OpConstant %uint 16
   %arr_type = OpTypeArray %uint %uint_16
       %SSBO = OpTypeStruct %arr_type
   %ptr_ssbo = OpTypePointer StorageBuffer %SSBO
  %ptr_ssbo_u = OpTypePointer StorageBuffer %uint
  %ptr_func_u = OpTypePointer Function %uint
   %ssbo_var = OpVariable %ptr_ssbo StorageBuffer
       %main = OpFunction %void None %void_fn
      %entry = OpLabel
    %idx_var = OpVariable %ptr_func_u Function
        %idx = OpLoad %uint %idx_var
        %ptr = OpAccessChain %ptr_ssbo_u %ssbo_var %uint_0 %idx
               OpReturn
               OpFunctionEnd
    )asm";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-AddressFromPoison", 1, SPV_SOURCE_ASM);
}

TEST_F(NegativeGpuAVPoisonValue, PoisonExternalStore) {
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint output_val;
        };
        void main() {
            uint x;
            output_val = x;
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ExternalStoreOfPoison");
}

TEST_F(NegativeGpuAVPoisonValue, PoisonVectorSwizzle) {
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint output_val;
        };
        void main() {
            vec4 v;
            float f = v.x;
            output_val = uint(f);
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ExternalStoreOfPoison");
}

TEST_F(NegativeGpuAVPoisonValue, PoisonMultipleOperands) {
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint output_val;
        };
        void main() {
            uint a;
            uint b;
            output_val = a + b;
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ExternalStoreOfPoison");
}

TEST_F(NegativeGpuAVPoisonValue, PrivateScopeUninitialized) {
    // Global variable in GLSL becomes Private storage class in SPIR-V
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint output_val;
        };
        uint private_var;
        void main() {
            output_val = private_var;
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ExternalStoreOfPoison");
}

TEST_F(NegativeGpuAVPoisonValue, ContaminatedVariable) {
    // Poison flows through a store into an initialized variable
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint output_val;
        };
        void main() {
            uint x;
            uint y = 0u;
            y = x;
            output_val = y;
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ExternalStoreOfPoison");
}

// --- A1: Composite types ---

TEST_F(NegativeGpuAVPoisonValue, PoisonVectorStore) {
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            vec4 output_val;
        };
        void main() {
            vec4 v;
            output_val = v;
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ExternalStoreOfPoison");
}

TEST_F(NegativeGpuAVPoisonValue, PoisonStructMember) {
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint output_val;
        };
        struct S {
            uint a;
            float b;
        };
        void main() {
            S s;
            output_val = s.a;
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ExternalStoreOfPoison");
}

TEST_F(NegativeGpuAVPoisonValue, PoisonMatrixElement) {
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            float output_val;
        };
        void main() {
            mat4 m;
            output_val = m[0][0];
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ExternalStoreOfPoison");
}

TEST_F(NegativeGpuAVPoisonValue, PoisonArrayElement) {
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint output_val;
        };
        void main() {
            uint arr[4];
            output_val = arr[0];
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ExternalStoreOfPoison");
}

// --- A2: Tier 1 instructions (pass-through) ---

TEST_F(NegativeGpuAVPoisonValue, PoisonCompositeConstruct) {
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            vec4 output_val;
        };
        void main() {
            float a;
            float b;
            float c;
            float d;
            output_val = vec4(a, b, c, d);
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ExternalStoreOfPoison");
}

TEST_F(NegativeGpuAVPoisonValue, PoisonCompositeInsert) {
    // Inserting a poison scalar into an initialized vector
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            vec4 output_val;
        };
        void main() {
            float poison_val;
            vec4 v = vec4(1.0, 2.0, 3.0, 4.0);
            v.x = poison_val;
            output_val = v;
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ExternalStoreOfPoison");
}

TEST_F(NegativeGpuAVPoisonValue, PoisonVectorShuffleMixed) {
    // Shuffle components from one uninitialized and one initialized vector
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            vec4 output_val;
        };
        void main() {
            vec4 poison_vec;
            vec4 clean_vec = vec4(1.0, 2.0, 3.0, 4.0);
            output_val = vec4(poison_vec.xy, clean_vec.zw);
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ExternalStoreOfPoison");
}

TEST_F(NegativeGpuAVPoisonValue, PoisonPhi) {
    // Poison flows through one branch of if/else, merges via phi
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint cond;
            uint output_val;
        };
        void main() {
            uint x;
            uint result;
            if (cond == 0) {
                result = x;
            } else {
                result = 42u;
            }
            output_val = result;
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ExternalStoreOfPoison");
}

TEST_F(NegativeGpuAVPoisonValue, PoisonPhiScalarAsm) {
    // Hand-written SPIR-V with an actual OpPhi.
    // One branch loads from an uninitialized Function variable (poison),
    // the other uses a constant (clean). The OpPhi merges them, and the
    // result is stored to the SSBO -> should trigger an error when the
    // poison path is taken (cond==0).
    const char* cs_source = R"asm(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
               OpDecorate %SSBO Block
               OpMemberDecorate %SSBO 0 Offset 0
               OpMemberDecorate %SSBO 1 Offset 4
               OpDecorate %ssbo_var DescriptorSet 0
               OpDecorate %ssbo_var Binding 0
       %void = OpTypeVoid
   %void_fn  = OpTypeFunction %void
       %uint = OpTypeInt 32 0
    %uint_0  = OpConstant %uint 0
   %uint_42  = OpConstant %uint 42
       %bool = OpTypeBool
       %SSBO = OpTypeStruct %uint %uint
%ptr_ssbo    = OpTypePointer StorageBuffer %SSBO
%ptr_ssbo_u  = OpTypePointer StorageBuffer %uint
%ptr_func_u  = OpTypePointer Function %uint
   %ssbo_var = OpVariable %ptr_ssbo StorageBuffer
       %main = OpFunction %void None %void_fn
      %entry = OpLabel
  %uninit_var = OpVariable %ptr_func_u Function
  %cond_ptr  = OpAccessChain %ptr_ssbo_u %ssbo_var %uint_0
  %cond_val  = OpLoad %uint %cond_ptr
  %is_zero   = OpIEqual %bool %cond_val %uint_0
               OpSelectionMerge %merge None
               OpBranchConditional %is_zero %true_br %false_br
   %true_br  = OpLabel
  %poison_val = OpLoad %uint %uninit_var
               OpBranch %merge
  %false_br  = OpLabel
               OpBranch %merge
      %merge = OpLabel
     %result = OpPhi %uint %poison_val %true_br %uint_42 %false_br
  %out_ptr   = OpAccessChain %ptr_ssbo_u %ssbo_var %uint_0
               OpStore %out_ptr %result
               OpReturn
               OpFunctionEnd
    )asm";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ExternalStoreOfPoison", 1, SPV_SOURCE_ASM);
}

TEST_F(NegativeGpuAVPoisonValue, PoisonPhiVectorAsm) {
    // OpPhi with vec4 type. One branch carries a poison vector (loaded from
    // an uninitialized Function variable), the other a clean constant.
    const char* cs_source = R"asm(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
               OpDecorate %SSBO Block
               OpMemberDecorate %SSBO 0 Offset 0
               OpMemberDecorate %SSBO 1 Offset 16
               OpDecorate %ssbo_var DescriptorSet 0
               OpDecorate %ssbo_var Binding 0
       %void = OpTypeVoid
   %void_fn  = OpTypeFunction %void
       %uint = OpTypeInt 32 0
    %uint_0  = OpConstant %uint 0
    %uint_1  = OpConstant %uint 1
     %v4uint = OpTypeVector %uint 4
  %clean_vec = OpConstantComposite %v4uint %uint_1 %uint_1 %uint_1 %uint_1
       %bool = OpTypeBool
       %SSBO = OpTypeStruct %uint %v4uint
%ptr_ssbo    = OpTypePointer StorageBuffer %SSBO
%ptr_ssbo_u  = OpTypePointer StorageBuffer %uint
%ptr_ssbo_v  = OpTypePointer StorageBuffer %v4uint
%ptr_func_v  = OpTypePointer Function %v4uint
   %ssbo_var = OpVariable %ptr_ssbo StorageBuffer
       %main = OpFunction %void None %void_fn
      %entry = OpLabel
  %uninit_var = OpVariable %ptr_func_v Function
  %cond_ptr  = OpAccessChain %ptr_ssbo_u %ssbo_var %uint_0
  %cond_val  = OpLoad %uint %cond_ptr
  %is_zero   = OpIEqual %bool %cond_val %uint_0
               OpSelectionMerge %merge None
               OpBranchConditional %is_zero %true_br %false_br
   %true_br  = OpLabel
  %poison_vec = OpLoad %v4uint %uninit_var
               OpBranch %merge
  %false_br  = OpLabel
               OpBranch %merge
      %merge = OpLabel
     %result = OpPhi %v4uint %poison_vec %true_br %clean_vec %false_br
  %out_ptr   = OpAccessChain %ptr_ssbo_v %ssbo_var %uint_1
               OpStore %out_ptr %result
               OpReturn
               OpFunctionEnd
    )asm";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ExternalStoreOfPoison", 1, SPV_SOURCE_ASM);
}

TEST_F(NegativeGpuAVPoisonValue, PoisonPhiBothPoisonAsm) {
    // OpPhi where BOTH incoming values are poison (from two different
    // uninitialized Function variables). Should trigger regardless of
    // which branch is taken.
    const char* cs_source = R"asm(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
               OpDecorate %SSBO Block
               OpMemberDecorate %SSBO 0 Offset 0
               OpMemberDecorate %SSBO 1 Offset 4
               OpDecorate %ssbo_var DescriptorSet 0
               OpDecorate %ssbo_var Binding 0
       %void = OpTypeVoid
   %void_fn  = OpTypeFunction %void
       %uint = OpTypeInt 32 0
    %uint_0  = OpConstant %uint 0
    %uint_1  = OpConstant %uint 1
       %bool = OpTypeBool
       %SSBO = OpTypeStruct %uint %uint
%ptr_ssbo    = OpTypePointer StorageBuffer %SSBO
%ptr_ssbo_u  = OpTypePointer StorageBuffer %uint
%ptr_func_u  = OpTypePointer Function %uint
   %ssbo_var = OpVariable %ptr_ssbo StorageBuffer
       %main = OpFunction %void None %void_fn
      %entry = OpLabel
 %uninit_a   = OpVariable %ptr_func_u Function
 %uninit_b   = OpVariable %ptr_func_u Function
  %cond_ptr  = OpAccessChain %ptr_ssbo_u %ssbo_var %uint_0
  %cond_val  = OpLoad %uint %cond_ptr
  %is_zero   = OpIEqual %bool %cond_val %uint_0
               OpSelectionMerge %merge None
               OpBranchConditional %is_zero %true_br %false_br
   %true_br  = OpLabel
  %val_a     = OpLoad %uint %uninit_a
               OpBranch %merge
  %false_br  = OpLabel
  %val_b     = OpLoad %uint %uninit_b
               OpBranch %merge
      %merge = OpLabel
     %result = OpPhi %uint %val_a %true_br %val_b %false_br
  %out_ptr   = OpAccessChain %ptr_ssbo_u %ssbo_var %uint_1
               OpStore %out_ptr %result
               OpReturn
               OpFunctionEnd
    )asm";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ExternalStoreOfPoison", 1, SPV_SOURCE_ASM);
}

// --- A3: Tier 2 instructions (propagation through different op families) ---

TEST_F(NegativeGpuAVPoisonValue, PoisonPropagateFloat) {
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint output_val;
        };
        void main() {
            float f;
            float g = f * 2.0;
            output_val = uint(g);
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ExternalStoreOfPoison");
}

TEST_F(NegativeGpuAVPoisonValue, PoisonPropagateBitwise) {
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint output_val;
        };
        void main() {
            uint x;
            uint y = x & 0xFFu;
            output_val = y;
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ExternalStoreOfPoison");
}

TEST_F(NegativeGpuAVPoisonValue, PoisonPropagateComparison) {
    // Poison propagates through comparison, then used as branch condition
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint output_val;
        };
        void main() {
            uint x;
            if (x > 5u) {
                output_val = 1;
            } else {
                output_val = 0;
            }
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-BranchOnPoison");
}

TEST_F(NegativeGpuAVPoisonValue, PoisonPropagateConversion) {
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint output_val;
        };
        void main() {
            int x;
            float y = float(x);
            output_val = uint(y);
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ExternalStoreOfPoison");
}

TEST_F(NegativeGpuAVPoisonValue, PoisonPropagateDot) {
    // Dot product reduces vec4 to float, exercises dimensionality reduction
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint output_val;
        };
        void main() {
            vec4 v;
            float d = dot(v, vec4(1.0));
            output_val = uint(d);
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ExternalStoreOfPoison");
}

// --- A4: Tier 3 instructions ---

TEST_F(NegativeGpuAVPoisonValue, PoisonSwitchSelector) {
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint output_val;
        };
        void main() {
            uint x;
            switch (x) {
                case 0:  output_val = 0; break;
                case 1:  output_val = 1; break;
                default: output_val = 2; break;
            }
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-BranchOnPoison");
}

// --- A5: Chaining and complex flow ---

TEST_F(NegativeGpuAVPoisonValue, PoisonMultiStepChain) {
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint output_val;
        };
        void main() {
            uint x;
            uint y = x * 2u;
            uint z = y + 1u;
            uint w = z & 0xFFu;
            output_val = w;
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ExternalStoreOfPoison");
}

TEST_F(NegativeGpuAVPoisonValue, PoisonThroughLoop) {
    // Poison initial value accumulated through loop iterations
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint output_val;
        };
        void main() {
            uint sum;
            for (int i = 0; i < 4; i++) {
                sum += uint(i);
            }
            output_val = sum;
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ExternalStoreOfPoison");
}

TEST_F(NegativeGpuAVPoisonValue, PoisonPartialVectorPoison) {
    // Per-component: 3 clean + 1 poison, extract the poison component
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint output_val;
        };
        void main() {
            float d;
            vec4 v = vec4(1.0, 2.0, 3.0, d);
            output_val = uint(v.w);
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ExternalStoreOfPoison");
}

TEST_F(NegativeGpuAVPoisonValue, PoisonNestedStruct) {
    // Struct containing an array, all uninitialized
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint output_val;
        };
        struct Inner {
            uint data[4];
        };
        void main() {
            Inner s;
            output_val = s.data[0];
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ExternalStoreOfPoison");
}

TEST_F(NegativeGpuAVPoisonValue, PoisonContaminatedChain) {
    // Contamination + further arithmetic propagation
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint output_val;
        };
        void main() {
            uint x;
            uint y = 0u;
            y = x;
            uint z = y + 1u;
            output_val = z;
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ExternalStoreOfPoison");
}

TEST_F(NegativeGpuAVPoisonValue, PoisonStructOfStruct) {
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint output_val;
        };
        struct Inner { uint x; float y; };
        struct Outer { Inner i; uint z; };
        void main() {
            Outer o;
            output_val = o.i.x;
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ExternalStoreOfPoison");
}

TEST_F(NegativeGpuAVPoisonValue, PoisonArrayOfStructs) {
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint output_val;
        };
        struct S { uint x; float y; };
        void main() {
            S arr[3];
            output_val = arr[1].x;
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ExternalStoreOfPoison");
}

TEST_F(NegativeGpuAVPoisonValue, PoisonStructWithVector) {
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint output_val;
        };
        struct S { vec4 v; uint x; };
        void main() {
            S s;
            output_val = uint(s.v.x);
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ExternalStoreOfPoison");
}

TEST_F(NegativeGpuAVPoisonValue, PoisonStructWithMatrix) {
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint output_val;
        };
        struct S { mat4 m; };
        void main() {
            S s;
            output_val = uint(s.m[0][0]);
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ExternalStoreOfPoison");
}

TEST_F(NegativeGpuAVPoisonValue, PoisonArrayOfArrays) {
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint output_val;
        };
        void main() {
            uint arr[2][3];
            output_val = arr[0][0];
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ExternalStoreOfPoison");
}

TEST_F(NegativeGpuAVPoisonValue, PoisonDeepNesting) {
    // 3 levels: struct of struct of struct
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint output_val;
        };
        struct L1 { uint val; };
        struct L2 { L1 inner; };
        struct L3 { L2 mid; };
        void main() {
            L3 s;
            output_val = s.mid.inner.val;
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ExternalStoreOfPoison");
}

TEST_F(NegativeGpuAVPoisonValue, PoisonLargeStruct) {
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint output_val;
        };
        struct Big {
            uint a; uint b; uint c; uint d;
            uint e; uint f; uint g; uint h;
        };
        void main() {
            Big s;
            output_val = s.h;
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ExternalStoreOfPoison");
}

TEST_F(NegativeGpuAVPoisonValue, PoisonDoubleVector) {
    AddRequiredFeature(vkt::Feature::shaderFloat64);
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint output_val;
        };
        void main() {
            dvec4 v;
            output_val = uint(v.x);
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ExternalStoreOfPoison");
}

TEST_F(NegativeGpuAVPoisonValue, PoisonSpecConstantArray) {
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint output_val;
        };
        layout(constant_id = 0) const int N = 4;
        void main() {
            uint arr[N];
            output_val = arr[0];
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ExternalStoreOfPoison");
}

TEST_F(NegativeGpuAVPoisonValue, PoisonStructArrayOfVectors) {
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint output_val;
        };
        struct S { vec3 positions[2]; };
        void main() {
            S s;
            output_val = uint(s.positions[0].x);
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ExternalStoreOfPoison");
}

TEST_F(NegativeGpuAVPoisonValue, PoisonThroughFunctionCall) {
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint output_val;
        };
        uint identity(uint x) { return x; }
        void main() {
            uint x;
            output_val = identity(x);
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ExternalStoreOfPoison");
}

TEST_F(NegativeGpuAVPoisonValue, PoisonVectorThroughFunctionCall) {
    // Only one component is poison; the shadow reduction should still taint the result
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint output_val;
        };
        float sum_vec(vec4 v) { return v.x + v.y + v.z + v.w; }
        void main() {
            float d;
            vec4 v = vec4(1.0, 2.0, 3.0, d);
            output_val = uint(sum_vec(v));
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ExternalStoreOfPoison");
}

TEST_F(NegativeGpuAVPoisonValue, PoisonStructThroughFunctionCall) {
    // Only one member is poison; shadow reduction should detect it
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint output_val;
        };
        struct S { uint a; float b; };
        uint get_a(S s) { return s.a; }
        void main() {
            float uninit_f;
            S s = S(42u, uninit_f);
            output_val = get_a(s);
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ExternalStoreOfPoison");
}

TEST_F(NegativeGpuAVPoisonValue, PoisonMixedArgsFunctionCall) {
    // Two args, only one is poison
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint output_val;
        };
        uint add(uint a, uint b) { return a + b; }
        void main() {
            uint x;
            uint y = 42u;
            output_val = add(x, y);
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ExternalStoreOfPoison");
}

TEST_F(NegativeGpuAVPoisonValue, PoisonReturnFromFunction) {
    // Function has its own uninit local and returns it
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint output_val;
        };
        uint make_poison() {
            uint x;
            return x;
        }
        void main() {
            output_val = make_poison();
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ReturnOfPoison");
}

TEST_F(NegativeGpuAVPoisonValue, PoisonFunctionParamOut) {
    // Callee has uninit local and stores it to an out parameter
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint output_val;
        };
        void make_poison(out uint result) {
            uint x;
            result = x;
        }
        void main() {
            uint val;
            make_poison(val);
            output_val = val;
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-StoreToFunctionParam");
}

TEST_F(NegativeGpuAVPoisonValue, PoisonFunctionParamInout) {
    // Callee has uninit local and stores it to an inout parameter
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint output_val;
        };
        void replace_with_poison(inout uint val) {
            uint x;
            val = x;
        }
        void main() {
            uint val = 42u;
            replace_with_poison(val);
            output_val = val;
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-StoreToFunctionParam");
}

TEST_F(NegativeGpuAVPoisonValue, PoisonFunctionParamConstIn) {
    // const in passes by value; callee returns it, caller stores externally
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint output_val;
        };
        uint identity(const in uint x) { return x; }
        void main() {
            uint x;
            output_val = identity(x);
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ExternalStoreOfPoison");
}

TEST_F(NegativeGpuAVPoisonValue, PoisonNestedFunctionCall) {
    // Poison flows through two levels of function calls
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint output_val;
        };
        uint inner(uint x) { return x + 1u; }
        uint outer(uint x) { return inner(x); }
        void main() {
            uint x;
            output_val = outer(x);
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ExternalStoreOfPoison");
}

TEST_F(NegativeGpuAVPoisonValue, PoisonFunctionCallChain) {
    // Result of one function call feeds into another
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint output_val;
        };
        uint add_one(uint x) { return x + 1u; }
        uint double_it(uint x) { return x * 2u; }
        void main() {
            uint x;
            output_val = double_it(add_one(x));
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ExternalStoreOfPoison");
}

TEST_F(NegativeGpuAVPoisonValue, PoisonMultipleOutParams) {
    // Function writes poison to multiple out parameters
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint output_val;
        };
        void make_two_poisons(out uint a, out uint b) {
            uint x;
            a = x;
            b = x;
        }
        void main() {
            uint a, b;
            make_two_poisons(a, b);
            output_val = 0u;
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-StoreToFunctionParam");
}

TEST_F(NegativeGpuAVPoisonValue, PoisonOutParamPartialInit) {
    // Function writes poison to one out param and clean to another;
    // the poison one should be detected
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint output_val;
        };
        void partial_init(out uint good, out uint bad) {
            uint x;
            good = 42u;
            bad = x;
        }
        void main() {
            uint a, b;
            partial_init(a, b);
            output_val = 0u;
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-StoreToFunctionParam");
}

TEST_F(NegativeGpuAVPoisonValue, PoisonNestedOutParam) {
    // Inner function writes poison to out param, outer passes it through
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint output_val;
        };
        void make_poison(out uint result) {
            uint x;
            result = x;
        }
        void wrapper(out uint result) {
            make_poison(result);
        }
        void main() {
            uint val;
            wrapper(val);
            output_val = val;
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-StoreToFunctionParam");
}

TEST_F(NegativeGpuAVPoisonValue, PoisonReturnThenStore) {
    // Function returns poison, caller stores the result externally
    const char* cs_source = R"glsl(
        #version 450
        layout(set=0, binding=0) buffer SSBO {
            uint output_val;
        };
        uint get_poison() {
            uint x;
            return x;
        }
        uint relay() {
            return get_poison();
        }
        void main() {
            output_val = relay();
        }
    )glsl";
    SimpleComputeTest(cs_source, "SPIRV-PoisonValue-ReturnOfPoison");
}

// --- BDA and variable pointer poison ---

TEST_F(NegativeGpuAVPoisonValue, PoisonBdaPointerStore) {
    // Uninitialized uint64 converted to BDA pointer, then stored through.
    // The pointer value is poison; dereferencing it is UB.
    SetTargetApiVersion(VK_API_VERSION_1_2);
    AddRequiredExtensions(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::shaderInt64);

    std::vector<VkLayerSettingEXT> settings = {
        {OBJECT_LAYER_NAME, "gpuav_poison_value", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &kVkTrue},
        {OBJECT_LAYER_NAME, "gpuav_descriptor_checks", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &kVkFalse},
        {OBJECT_LAYER_NAME, "gpuav_buffer_address_oob", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &kVkFalse},
        {OBJECT_LAYER_NAME, "gpuav_ray_query", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &kVkFalse},
        {OBJECT_LAYER_NAME, "gpuav_shader_sanitizer", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &kVkFalse},
        {OBJECT_LAYER_NAME, "gpuav_shared_memory_data_race", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &kVkFalse},
        {OBJECT_LAYER_NAME, "gpuav_debug_validate_instrumented_shaders", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &kVkTrue},
        {OBJECT_LAYER_NAME, "gpuav_debug_dump_instrumented_shaders", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &kVkTrue},
    };
    RETURN_IF_SKIP(InitGpuAvFramework(settings, false));
    RETURN_IF_SKIP(InitState());

    // Hand-written SPIR-V: declares an uninitialized Function-scope uint64,
    // converts it to a PhysicalStorageBuffer pointer, and stores through it.
    const char* cs_source = R"asm(
               OpCapability Shader
               OpCapability Int64
               OpCapability PhysicalStorageBufferAddresses
               OpExtension "SPV_KHR_physical_storage_buffer"
               OpMemoryModel PhysicalStorageBuffer64 GLSL450
               OpEntryPoint GLCompute %main "main" %ssbo
               OpExecutionMode %main LocalSize 1 1 1
               OpDecorate %ssbo_type Block
               OpMemberDecorate %ssbo_type 0 Offset 0
               OpDecorate %ssbo DescriptorSet 0
               OpDecorate %ssbo Binding 0
       %void = OpTypeVoid
       %func = OpTypeFunction %void
       %uint = OpTypeInt 32 0
     %uint64 = OpTypeInt 64 0
  %ssbo_type = OpTypeStruct %uint
%ptr_sb_struct = OpTypePointer StorageBuffer %ssbo_type
 %ptr_sb_uint = OpTypePointer StorageBuffer %uint
%ptr_psb_uint = OpTypePointer PhysicalStorageBuffer %uint
%ptr_func_u64 = OpTypePointer Function %uint64
       %ssbo = OpVariable %ptr_sb_struct StorageBuffer
      %idx_0 = OpConstant %uint 0
    %val_42u = OpConstant %uint 42
       %main = OpFunction %void None %func
      %entry = OpLabel
   %addr_var = OpVariable %ptr_func_u64 Function
       %addr = OpLoad %uint64 %addr_var
        %ptr = OpConvertUToPtr %ptr_psb_uint %addr
               OpStore %ptr %val_42u Aligned 4
      %out_p = OpAccessChain %ptr_sb_uint %ssbo %idx_0
               OpStore %out_p %val_42u
               OpReturn
               OpFunctionEnd
    )asm";

    CreateComputePipelineHelper pipe(*this);
    pipe.dsl_bindings_[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL, nullptr};
    pipe.cs_ = VkShaderObj(*m_device, cs_source, VK_SHADER_STAGE_COMPUTE_BIT, SPV_ENV_VULKAN_1_2, SPV_SOURCE_ASM);
    pipe.CreateComputePipeline();

    vkt::Buffer buffer(*m_device, 256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemProps);
    void* ptr = buffer.Memory().Map();
    memset(ptr, 0, 256);

    pipe.descriptor_set_.WriteDescriptorBufferInfo(0, buffer, 0, VK_WHOLE_SIZE, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    pipe.descriptor_set_.UpdateDescriptorSets();

    m_command_buffer.Begin();
    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vk::CmdBindDescriptorSets(m_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline_layout_, 0, 1,
                              &pipe.descriptor_set_.set_, 0, nullptr);
    vk::CmdDispatch(m_command_buffer, 1, 1, 1);
    m_command_buffer.End();

    m_errorMonitor->SetDesiredError("SPIRV-PoisonValue-PoisonPointerDereference");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVPoisonValue, PoisonBdaPointerLoad) {
    // Uninitialized uint64 converted to BDA pointer, then loaded through.
    SetTargetApiVersion(VK_API_VERSION_1_2);
    AddRequiredExtensions(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::bufferDeviceAddress);
    AddRequiredFeature(vkt::Feature::shaderInt64);

    std::vector<VkLayerSettingEXT> settings = {
        {OBJECT_LAYER_NAME, "gpuav_poison_value", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &kVkTrue},
        {OBJECT_LAYER_NAME, "gpuav_descriptor_checks", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &kVkFalse},
        {OBJECT_LAYER_NAME, "gpuav_buffer_address_oob", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &kVkFalse},
        {OBJECT_LAYER_NAME, "gpuav_ray_query", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &kVkFalse},
        {OBJECT_LAYER_NAME, "gpuav_shader_sanitizer", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &kVkFalse},
        {OBJECT_LAYER_NAME, "gpuav_shared_memory_data_race", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &kVkFalse},
        {OBJECT_LAYER_NAME, "gpuav_debug_validate_instrumented_shaders", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &kVkTrue},
        {OBJECT_LAYER_NAME, "gpuav_debug_dump_instrumented_shaders", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &kVkTrue},
    };
    RETURN_IF_SKIP(InitGpuAvFramework(settings, false));
    RETURN_IF_SKIP(InitState());

    // Hand-written SPIR-V: declares an uninitialized Function-scope uint64,
    // converts it to a PhysicalStorageBuffer pointer, and loads through it.
    const char* cs_source = R"asm(
               OpCapability Shader
               OpCapability Int64
               OpCapability PhysicalStorageBufferAddresses
               OpExtension "SPV_KHR_physical_storage_buffer"
               OpMemoryModel PhysicalStorageBuffer64 GLSL450
               OpEntryPoint GLCompute %main "main" %ssbo
               OpExecutionMode %main LocalSize 1 1 1
               OpDecorate %ssbo_type Block
               OpMemberDecorate %ssbo_type 0 Offset 0
               OpDecorate %ssbo DescriptorSet 0
               OpDecorate %ssbo Binding 0
       %void = OpTypeVoid
       %func = OpTypeFunction %void
       %uint = OpTypeInt 32 0
     %uint64 = OpTypeInt 64 0
  %ssbo_type = OpTypeStruct %uint
%ptr_sb_struct = OpTypePointer StorageBuffer %ssbo_type
 %ptr_sb_uint = OpTypePointer StorageBuffer %uint
%ptr_psb_uint = OpTypePointer PhysicalStorageBuffer %uint
%ptr_func_u64 = OpTypePointer Function %uint64
       %ssbo = OpVariable %ptr_sb_struct StorageBuffer
      %idx_0 = OpConstant %uint 0
       %main = OpFunction %void None %func
      %entry = OpLabel
   %addr_var = OpVariable %ptr_func_u64 Function
       %addr = OpLoad %uint64 %addr_var
        %ptr = OpConvertUToPtr %ptr_psb_uint %addr
        %val = OpLoad %uint %ptr Aligned 4
      %out_p = OpAccessChain %ptr_sb_uint %ssbo %idx_0
               OpStore %out_p %val
               OpReturn
               OpFunctionEnd
    )asm";

    CreateComputePipelineHelper pipe(*this);
    pipe.dsl_bindings_[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL, nullptr};
    pipe.cs_ = VkShaderObj(*m_device, cs_source, VK_SHADER_STAGE_COMPUTE_BIT, SPV_ENV_VULKAN_1_2, SPV_SOURCE_ASM);
    pipe.CreateComputePipeline();

    vkt::Buffer buffer(*m_device, 256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemProps);
    void* ptr = buffer.Memory().Map();
    memset(ptr, 0, 256);

    pipe.descriptor_set_.WriteDescriptorBufferInfo(0, buffer, 0, VK_WHOLE_SIZE, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    pipe.descriptor_set_.UpdateDescriptorSets();

    m_command_buffer.Begin();
    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vk::CmdBindDescriptorSets(m_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline_layout_, 0, 1,
                              &pipe.descriptor_set_.set_, 0, nullptr);
    vk::CmdDispatch(m_command_buffer, 1, 1, 1);
    m_command_buffer.End();

    m_errorMonitor->SetDesiredError("SPIRV-PoisonValue-PoisonPointerDereference");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}

TEST_F(NegativeGpuAVPoisonValue, PoisonVariablePointerSelect) {
    // OpSelect on SSBO pointers with a poison condition (uninitialized bool).
    // The resulting pointer is poison; dereferencing it is UB.
    SetTargetApiVersion(VK_API_VERSION_1_1);
    AddRequiredExtensions(VK_KHR_VARIABLE_POINTERS_EXTENSION_NAME);
    AddRequiredFeature(vkt::Feature::variablePointersStorageBuffer);

    std::vector<VkLayerSettingEXT> settings = {
        {OBJECT_LAYER_NAME, "gpuav_poison_value", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &kVkTrue},
        {OBJECT_LAYER_NAME, "gpuav_descriptor_checks", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &kVkFalse},
        {OBJECT_LAYER_NAME, "gpuav_buffer_address_oob", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &kVkFalse},
        {OBJECT_LAYER_NAME, "gpuav_ray_query", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &kVkFalse},
        {OBJECT_LAYER_NAME, "gpuav_shader_sanitizer", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &kVkFalse},
        {OBJECT_LAYER_NAME, "gpuav_shared_memory_data_race", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &kVkFalse},
        {OBJECT_LAYER_NAME, "gpuav_debug_validate_instrumented_shaders", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &kVkTrue},
        {OBJECT_LAYER_NAME, "gpuav_debug_dump_instrumented_shaders", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &kVkTrue},
    };
    RETURN_IF_SKIP(InitGpuAvFramework(settings, false));
    RETURN_IF_SKIP(InitState());

    // Hand-written SPIR-V: uninitialized Function-scope bool used as
    // OpSelect condition between two SSBO pointers, then stored through.
    const char* cs_source = R"asm(
               OpCapability Shader
               OpCapability VariablePointersStorageBuffer
               OpExtension "SPV_KHR_variable_pointers"
               OpExtension "SPV_KHR_storage_buffer_storage_class"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
               OpDecorate %ssbo_type Block
               OpMemberDecorate %ssbo_type 0 Offset 0
               OpMemberDecorate %ssbo_type 1 Offset 4
               OpDecorate %ssbo DescriptorSet 0
               OpDecorate %ssbo Binding 0
       %void = OpTypeVoid
       %func = OpTypeFunction %void
       %uint = OpTypeInt 32 0
       %bool = OpTypeBool
  %ssbo_type = OpTypeStruct %uint %uint
%ptr_sb_struct = OpTypePointer StorageBuffer %ssbo_type
 %ptr_sb_uint = OpTypePointer StorageBuffer %uint
%ptr_func_bool = OpTypePointer Function %bool
       %ssbo = OpVariable %ptr_sb_struct StorageBuffer
      %idx_0 = OpConstant %uint 0
      %idx_1 = OpConstant %uint 1
    %val_42u = OpConstant %uint 42
       %main = OpFunction %void None %func
      %entry = OpLabel
   %cond_var = OpVariable %ptr_func_bool Function
       %cond = OpLoad %bool %cond_var
      %ptr_a = OpAccessChain %ptr_sb_uint %ssbo %idx_0
      %ptr_b = OpAccessChain %ptr_sb_uint %ssbo %idx_1
     %chosen = OpSelect %ptr_sb_uint %cond %ptr_a %ptr_b
               OpStore %chosen %val_42u
               OpReturn
               OpFunctionEnd
    )asm";

    CreateComputePipelineHelper pipe(*this);
    pipe.dsl_bindings_[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL, nullptr};
    pipe.cs_ = VkShaderObj(*m_device, cs_source, VK_SHADER_STAGE_COMPUTE_BIT, SPV_ENV_VULKAN_1_1, SPV_SOURCE_ASM);
    pipe.CreateComputePipeline();

    vkt::Buffer buffer(*m_device, 256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostVisibleMemProps);
    void* ptr = buffer.Memory().Map();
    memset(ptr, 0, 256);

    pipe.descriptor_set_.WriteDescriptorBufferInfo(0, buffer, 0, VK_WHOLE_SIZE, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    pipe.descriptor_set_.UpdateDescriptorSets();

    m_command_buffer.Begin();
    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vk::CmdBindDescriptorSets(m_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline_layout_, 0, 1,
                              &pipe.descriptor_set_.set_, 0, nullptr);
    vk::CmdDispatch(m_command_buffer, 1, 1, 1);
    m_command_buffer.End();

    m_errorMonitor->SetDesiredError("SPIRV-PoisonValue-PoisonPointerDereference");
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();
}
