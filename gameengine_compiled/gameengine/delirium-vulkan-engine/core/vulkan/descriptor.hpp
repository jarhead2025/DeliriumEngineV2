#pragma once
#include "vulkan_device.hpp"
#include <vector>

namespace delirium {

// Uniform buffer object — MVP matrix + extra uniforms
struct alignas(16) UniformBufferObject {
    alignas(16) float mvp[16];       // row-major MVP matrix
    alignas(16) float view[16];      // row-major view matrix
    alignas(16) float proj[16];      // row-major projection matrix
    alignas(16) float cam_pos[4];    // camera position + padding
};

class DescriptorManager {
public:
    bool initialize(VulkanDevice& device, VkDescriptorSetLayout layout, uint32_t frame_count);
    void shutdown(VulkanDevice& device);

    void update_ubo(VulkanDevice& device, uint32_t frame_index, const UniformBufferObject& ubo);

    // Bind the surfel storage buffer to binding 1 of the given frame's descriptor set
    void bind_surfel_buffer(VulkanDevice& device, uint32_t frame_index, VkBuffer buffer);

    VkDescriptorSet descriptor_set(uint32_t frame_index) const { return descriptor_sets_[frame_index]; }
    VkDescriptorPool descriptor_pool() const { return descriptor_pool_; }

private:
    void create_descriptor_pool(VulkanDevice& device, uint32_t frame_count);
    void allocate_descriptor_sets(VulkanDevice& device, VkDescriptorSetLayout layout,
                                   uint32_t frame_count);
    bool create_buffer(VulkanDevice& device, VkDeviceSize size, VkMemoryPropertyFlags properties,
                       VkBuffer& buffer, VkDeviceMemory& memory);
    void map_buffer(VulkanDevice& device, VkDeviceSize size, VkDeviceMemory& memory, void** data);

    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptor_sets_;

    struct UboData {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        void* mapped = nullptr;
    };
    std::vector<UboData> ubos_;
};

} // namespace delirium