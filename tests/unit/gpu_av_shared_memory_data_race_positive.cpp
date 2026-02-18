/*
 * Copyright (c) 2020-2026 The Khronos Group Inc.
 * Copyright (c) 2020-2026 Valve Corporation
 * Copyright (c) 2020-2026 LunarG, Inc.
 * Copyright (c) 2020-2026 Google, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#include <gtest/gtest.h>
#include "../framework/layer_validation_tests.h"
#include "../framework/pipeline_helper.h"
#include "../framework/descriptor_helper.h"
#include "generated/vk_function_pointers.h"

void GpuAVSharedMemoryDataRaceTest::InitSharedMemoryDataRace() {
    SetTargetApiVersion(VK_API_VERSION_1_2);

    RETURN_IF_SKIP(InitGpuAvFramework({}, false));
    RETURN_IF_SKIP(InitState());
}

class PositiveGpuAVSharedMemoryDataRaceTest : public GpuAVSharedMemoryDataRaceTest {
protected:
    void TestHelper(const char *source);
};

void PositiveGpuAVSharedMemoryDataRaceTest::TestHelper(const char *shader_source) {
    TEST_DESCRIPTION("Shared memory, no data race");
    RETURN_IF_SKIP(InitSharedMemoryDataRace());

    CreateComputePipelineHelper pipe(*this);
    pipe.cs_ = VkShaderObj(*m_device, shader_source, VK_SHADER_STAGE_COMPUTE_BIT);
    pipe.CreateComputePipeline();

    m_command_buffer.Begin();
    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vk::CmdDispatch(m_command_buffer, 1, 1, 1);
    m_command_buffer.End();

    m_default_queue->SubmitAndWait(m_command_buffer);
}

TEST_F(PositiveGpuAVSharedMemoryDataRaceTest, SingleScalar) {
    const char *shader_source = R"glsl(
        #version 450
        #extension GL_KHR_memory_scope_semantics : enable

        layout(local_size_x = 2) in;
        shared uint temp;
        void main() {
            atomicStore(temp, 0u, gl_ScopeWorkgroup, 0, 0);
        }
    )glsl";

    TestHelper(shader_source);
}

TEST_F(PositiveGpuAVSharedMemoryDataRaceTest, SingleElementAccess) {
    const char *shader_source = R"glsl(
        #version 450

        layout(local_size_x = 2) in;
        shared uint temp[2];
        void main() {
            temp[gl_LocalInvocationIndex] = 0;
        }
    )glsl";

    TestHelper(shader_source);
}

TEST_F(PositiveGpuAVSharedMemoryDataRaceTest, TwoThreadsShareValuesThroughArray) {
    const char *shader_source = R"glsl(
        #version 450

        layout(local_size_x = 2) in;
        shared uint temp[2];
        void main() {
            temp[gl_LocalInvocationIndex] = 0;
            barrier();
            uint x = temp[gl_LocalInvocationIndex ^ 1];
        }
    )glsl";

    TestHelper(shader_source);
}

TEST_F(PositiveGpuAVSharedMemoryDataRaceTest, TwoDimensionalArrayBarrier) {
    const char *shader_source = R"glsl(
        #version 450

        layout(local_size_x = 3) in;
        shared uint temp[3][2];
        void main() {
            temp[gl_LocalInvocationIndex][1] = 0;
            barrier();
            uint x = temp[(gl_LocalInvocationIndex + 1) % 3][1];
        }
    )glsl";

    TestHelper(shader_source);
}

TEST_F(PositiveGpuAVSharedMemoryDataRaceTest, TwoDimensionalArrayNoRace) {
    const char *shader_source = R"glsl(
        #version 450

        layout(local_size_x = 3) in;
        shared uint temp[3][2];
        void main() {
            temp[gl_LocalInvocationIndex][0] = 0;
            uint x = temp[(gl_LocalInvocationIndex + 1) % 3][1];
        }
    )glsl";

    TestHelper(shader_source);
}

TEST_F(PositiveGpuAVSharedMemoryDataRaceTest, BasicStructNoRace) {
    const char *shader_source = R"glsl(
        #version 450

        layout(local_size_x = 2) in;
        struct S { uint a, b; };
        shared S temp;
        void main() {
            if (gl_LocalInvocationIndex == 0) {
                temp.a = 0;
            } else {
                temp.b = 0;
            }
            barrier();
            if (gl_LocalInvocationIndex == 0) {
                temp.b = 0;
            } else {
                temp.a = 0;
            }
        }
    )glsl";

    TestHelper(shader_source);
}
