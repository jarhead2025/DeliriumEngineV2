#pragma once
#include "vulkan_device.hpp"
#include "swapchain.hpp"
#include <vector>

namespace delirium {

class Pipeline {
public:
    bool initialize(VulkanDevice& device, Swapchain& swapchain,
                    const char* vert_path, const char* frag_path);
    void shutdown(VulkanDevice& device);

    VkPipeline pipeline() const { return graphics_pipeline_; }
    VkPipelineLayout layout() const { return pipeline_layout_; }
    VkDescriptorSetLayout descriptor_layout() const { return descriptor_set_layout_; }

private:
    std::vector<uint32_t> load_spv(const char* path);
    void create_descriptor_set_layout(VulkanDevice& device);
    void create_pipeline_layout(VulkanDevice& device);
    bool create_graphics_pipeline(VulkanDevice& device, Swapchain& swapchain,
                                   const char* vert_path, const char* frag_path);

    VkDescriptorSetLayout descriptor_set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline graphics_pipeline_ = VK_NULL_HANDLE;
};

} // namespace delirium