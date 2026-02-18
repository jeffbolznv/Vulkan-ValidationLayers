/*
 * Copyright (c) 2020-2025 The Khronos Group Inc.
 * Copyright (c) 2020-2025 Valve Corporation
 * Copyright (c) 2020-2025 LunarG, Inc.
 * Copyright (c) 2020-2025 Google, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#include "../framework/layer_validation_tests.h"
#include "../framework/pipeline_helper.h"
#include "../framework/descriptor_helper.h"
#include "../layers/containers/range.h"

class NegativeGpuAVSharedMemoryDataRaceTest : public GpuAVSharedMemoryDataRaceTest {
protected:
    void TestHelper(const char *source, uint32_t count);
};

void NegativeGpuAVSharedMemoryDataRaceTest::TestHelper(const char *shader_source, uint32_t count) {
    TEST_DESCRIPTION("Shared memory, data race");
    RETURN_IF_SKIP(InitSharedMemoryDataRace());

    CreateComputePipelineHelper pipe(*this);
    pipe.cs_ = VkShaderObj(*m_device, shader_source, VK_SHADER_STAGE_COMPUTE_BIT);
    pipe.CreateComputePipeline();

    m_command_buffer.Begin();
    vk::CmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vk::CmdDispatch(m_command_buffer, 1, 1, 1);
    m_command_buffer.End();

    m_errorMonitor->SetDesiredError("VUID-RuntimeSpirv-XXX", count);
    m_default_queue->SubmitAndWait(m_command_buffer);
    m_errorMonitor->VerifyFound();

}

TEST_F(NegativeGpuAVSharedMemoryDataRaceTest, SingleScalar) {
    const char *shader_source = R"glsl(
        #version 450

        layout(local_size_x = 2) in;
        shared uint temp;
        void main() {
            temp = 0;
        }
    )glsl";

    TestHelper(shader_source, 1);
}

TEST_F(NegativeGpuAVSharedMemoryDataRaceTest, SingleElementArray) {
    const char *shader_source = R"glsl(
        #version 450

        layout(local_size_x = 2) in;
        shared uint temp[1];
        void main() {
            temp[0] = 0;
        }
    )glsl";

    TestHelper(shader_source, 1);
}


TEST_F(NegativeGpuAVSharedMemoryDataRaceTest, TwoThreadsShareValuesThroughArray) {
    const char *shader_source = R"glsl(
        #version 450

        layout(local_size_x = 2) in;
        shared uint temp[2];
        void main() {
            temp[gl_LocalInvocationIndex] = 0;
            uint x = temp[gl_LocalInvocationIndex ^ 1];
        }
    )glsl";

    TestHelper(shader_source, 2);
}

TEST_F(NegativeGpuAVSharedMemoryDataRaceTest, TwoDimensionalArray) {
    const char *shader_source = R"glsl(
        #version 450

        layout(local_size_x = 3) in;
        shared uint temp[3][2];
        void main() {
            temp[gl_LocalInvocationIndex][1] = 0;
            uint x = temp[(gl_LocalInvocationIndex + 1) % 3][1];
        }
    )glsl";

    TestHelper(shader_source, 3);
}
