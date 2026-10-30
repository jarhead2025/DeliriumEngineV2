#include "pipeline.hpp"
#include <array>
#include <fstream>
#include <iostream>

namespace delirium {

bool Pipeline::initialize(VulkanDevice& device, Swapchain& swapchain,
                           const char* vert_path, const char* frag_path) {
    create_descriptor_set_layout(device);
    create_pipeline_layout(device);
    if (!create_graphics_pipeline(device, swapchain, vert_path, frag_path))
        return false;
    return true;
}

void Pipeline::shutdown(VulkanDevice& device) {
    if (graphics_pipeline_) vkDestroyPipeline(device.device(), graphics_pipeline_, nullptr);
    if (pipeline_layout_) vkDestroyPipelineLayout(device.device(), pipeline_layout_, nullptr);
    if (descriptor_set_layout_) vkDestroyDescriptorSetLayout(device.device(), descriptor_set_layout_, nullptr);
}

std::vector<uint32_t> Pipeline::load_spv(const char* path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open shader: " << path << "\n";
        return {};
    }
    size_t size = file.tellg();
    file.seekg(0);
    std::vector<uint32_t> buffer(size / 4);
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    return buffer;
}

void Pipeline::create_descriptor_set_layout(VulkanDevice& device) {
    // Zero bindings for debugging — test if descriptor layout causes the crash
    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = 0;

    vkCreateDescriptorSetLayout(device.device(), &layout_info, nullptr, &descriptor_set_layout_);
}

void Pipeline::create_pipeline_layout(VulkanDevice& device) {
    VkPipelineLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &descriptor_set_layout_;

    vkCreatePipelineLayout(device.device(), &layout_info, nullptr, &pipeline_layout_);
}

bool Pipeline::create_graphics_pipeline(VulkanDevice& device, Swapchain& swapchain,
                                         const char* vert_path, const char* frag_path) {
    std::cerr << "  [pipeline] Loading vertex shader...\n";
    auto vert_spv = load_spv(vert_path);
    std::cerr << "  [pipeline] Loading fragment shader...\n";
    auto frag_spv = load_spv(frag_path);

    if (vert_spv.empty() || frag_spv.empty()) {
        std::cerr << "  [pipeline] Failed to load shaders\n";
        return false;
    }
    std::cerr << "  [pipeline] Shaders loaded: vert=" << vert_spv.size() << " frag=" << frag_spv.size() << " uint32s\n";

    std::cerr << "  [pipeline] Creating shader modules...\n";
    VkShaderModuleCreateInfo vert_ci{};
    vert_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    vert_ci.codeSize = vert_spv.size() * sizeof(uint32_t);
    vert_ci.pCode = vert_spv.data();

    VkShaderModuleCreateInfo frag_ci{};
    frag_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    frag_ci.codeSize = frag_spv.size() * sizeof(uint32_t);
    frag_ci.pCode = frag_spv.data();

    VkShaderModule vert_module, frag_module;
    VkResult vr = vkCreateShaderModule(device.device(), &vert_ci, nullptr, &vert_module);
    std::cerr << "  [pipeline] vert module result=" << vr << "\n";
    VkResult fr = vkCreateShaderModule(device.device(), &frag_ci, nullptr, &frag_module);
    std::cerr << "  [pipeline] frag module result=" << fr << "\n";

    std::array<VkPipelineShaderStageCreateInfo, 2> stages;

    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert_module;
    stages[0].pName = "main";

    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag_module;
    stages[1].pName = "main";

    // Minimal pipeline for Intel Arc driver compatibility
    VkPipelineVertexInputStateCreateInfo vertex_input{};
    vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo input_assembly{};
    input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;

    VkPipelineViewportStateCreateInfo viewport{};
    viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo raster{};
    raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.lineWidth = 1.0f;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depth_stencil{};
    depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    // No depth testing for now

    VkPipelineColorBlendAttachmentState color_blend{};
    color_blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    // No blending for now

    VkPipelineColorBlendStateCreateInfo blend{};
    blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount = 1;
    blend.pAttachments = &color_blend;

    // No dynamic state for now

    VkGraphicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline_info.stageCount = static_cast<uint32_t>(stages.size());
    pipeline_info.pStages = stages.data();
    pipeline_info.pVertexInputState = &vertex_input;
    pipeline_info.pInputAssemblyState = &input_assembly;
    pipeline_info.pViewportState = &viewport;
    pipeline_info.pRasterizationState = &raster;
    pipeline_info.pMultisampleState = &multisample;
    pipeline_info.pDepthStencilState = &depth_stencil;
    pipeline_info.pColorBlendState = &blend;
    pipeline_info.pDynamicState = nullptr;
    pipeline_info.layout = pipeline_layout_;
    pipeline_info.renderPass = swapchain.render_pass();
    pipeline_info.subpass = 0;

    std::cerr << "  [pipeline] Creating graphics pipeline...\n";
    VkResult pr = vkCreateGraphicsPipelines(device.device(), VK_NULL_HANDLE, 1, &pipeline_info,
                                             nullptr, &graphics_pipeline_);
    std::cerr << "  [pipeline] Pipeline create result=" << pr << "\n";

    vkDestroyShaderModule(device.device(), frag_module, nullptr);
    vkDestroyShaderModule(device.device(), vert_module, nullptr);
    return pr == VK_SUCCESS;
}

} // namespace delirium