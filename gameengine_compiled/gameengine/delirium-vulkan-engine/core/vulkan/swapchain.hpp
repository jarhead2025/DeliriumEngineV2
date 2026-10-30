#pragma once
#include "vulkan_device.hpp"
#include <vector>

namespace delirium {

class Swapchain {
public:
    bool initialize(VulkanDevice& device, uint32_t width, uint32_t height);
    void shutdown(VulkanDevice& device);
    void recreate(VulkanDevice& device, uint32_t width, uint32_t height);

    VkFormat format() const { return format_; }
    VkExtent2D extent() const { return extent_; }
    uint32_t image_count() const { return static_cast<uint32_t>(swapchain_images_.size()); }
    const std::vector<VkImage>& images() const { return swapchain_images_; }
    const std::vector<VkImageView>& views() const { return swapchain_views_; }
    const std::vector<VkFramebuffer>& framebuffers() const { return framebuffers_; }
    VkRenderPass render_pass() const { return render_pass_; }
    VkSwapchainKHR swapchain() const { return swapchain_; }

private:
    bool pick_surface_format(VulkanDevice& device);
    bool pick_present_mode(VulkanDevice& device);
    bool pick_extent(VulkanDevice& device, uint32_t width, uint32_t height);
    bool create_swapchain(VulkanDevice& device);
    void create_image_views(VulkanDevice& device);
    void create_render_pass(VulkanDevice& device);
    void create_framebuffers(VulkanDevice& device);

    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat format_ = VK_FORMAT_B8G8R8A8_UNORM;
    VkColorSpaceKHR color_space_ = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    VkPresentModeKHR present_mode_ = VK_PRESENT_MODE_FIFO_KHR;
    VkExtent2D extent_{};
    uint32_t image_count_ = 0;

    std::vector<VkImage> swapchain_images_;
    std::vector<VkImageView> swapchain_views_;
    std::vector<VkFramebuffer> framebuffers_;
    VkRenderPass render_pass_ = VK_NULL_HANDLE;

    // Depth buffer
    VkImage depth_image_ = VK_NULL_HANDLE;
    VkDeviceMemory depth_image_memory_ = VK_NULL_HANDLE;
    VkImageView depth_view_ = VK_NULL_HANDLE;

    // Attachment descriptions (need to persist for render pass)
    VkAttachmentDescription attachments_[2]{};
};

} // namespace delirium