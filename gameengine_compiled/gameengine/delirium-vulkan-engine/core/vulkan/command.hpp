#pragma once
#include "vulkan_device.hpp"
#include "swapchain.hpp"
#include <vector>

namespace delirium {

class CommandBuffers {
public:
    bool initialize(VulkanDevice& device, Swapchain& swapchain);
    void shutdown(VulkanDevice& device);

    void begin_frame(VulkanDevice& device);
    void end_frame(VulkanDevice& device, Swapchain& swapchain, uint32_t image_index);

    VkCommandBuffer current_command() const { return command_buffers_[current_]; }
    VkCommandPool command_pool() const { return command_pool_; }
    VkSemaphore image_available_semaphore(uint32_t i) const { return image_available_semaphores_[i]; }
    VkSemaphore render_finished_semaphore(uint32_t i) const { return render_finished_semaphores_[i]; }

private:
    void create_command_pool(VulkanDevice& device);
    void create_command_buffers(VulkanDevice& device, Swapchain& swapchain);
    void create_sync_objects(VulkanDevice& device, Swapchain& swapchain);

    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> command_buffers_;

    // Per-frame sync
    std::vector<VkSemaphore> image_available_semaphores_;
    std::vector<VkSemaphore> render_finished_semaphores_;
    std::vector<VkFence> in_flight_fences_;

    uint32_t current_ = 0;
    uint32_t max_frames_in_flight_ = 2;
};

} // namespace delirium