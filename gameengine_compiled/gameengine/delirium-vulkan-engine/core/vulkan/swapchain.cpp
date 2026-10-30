#include "swapchain.hpp"
#include <algorithm>
#include <set>
#include <cstdint>
#include <iostream>

namespace delirium {

bool Swapchain::initialize(VulkanDevice& device, uint32_t width, uint32_t height) {
    if (!pick_surface_format(device)) return false;
    if (!pick_present_mode(device)) return false;
    if (!pick_extent(device, width, height)) return false;
    if (!create_swapchain(device)) return false;
    create_image_views(device);
    create_render_pass(device);
    create_framebuffers(device);
    return true;
}

void Swapchain::shutdown(VulkanDevice& device) {
    for (VkFramebuffer fb : framebuffers_)
        vkDestroyFramebuffer(device.device(), fb, nullptr);
    vkDestroyRenderPass(device.device(), render_pass_, nullptr);
    for (VkImageView view : swapchain_views_)
        vkDestroyImageView(device.device(), view, nullptr);
    if (swapchain_) {
        vkDestroySwapchainKHR(device.device(), swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }
    if (depth_view_) vkDestroyImageView(device.device(), depth_view_, nullptr);
    if (depth_image_) vkDestroyImage(device.device(), depth_image_, nullptr);
    if (depth_image_memory_) vkFreeMemory(device.device(), depth_image_memory_, nullptr);
    framebuffers_.clear();
    swapchain_views_.clear();
    swapchain_images_.clear();
}

void Swapchain::recreate(VulkanDevice& device, uint32_t width, uint32_t height) {
    shutdown(device);
    initialize(device, width, height);
}

bool Swapchain::pick_surface_format(VulkanDevice& device) {
    uint32_t count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device.physical_device(), device.surface(), &count, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(device.physical_device(), device.surface(), &count, formats.data());

    for (const auto& fmt : formats) {
        if (fmt.format == VK_FORMAT_B8G8R8A8_UNORM && fmt.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            format_ = fmt.format;
            color_space_ = fmt.colorSpace;
            return true;
        }
    }
    format_ = formats[0].format;
    color_space_ = formats[0].colorSpace;
    return true;
}

bool Swapchain::pick_present_mode(VulkanDevice& device) {
    uint32_t count = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device.physical_device(), device.surface(), &count, nullptr);
    std::vector<VkPresentModeKHR> modes(count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(device.physical_device(), device.surface(), &count, modes.data());

    // Prefer mailbox (lowest latency), fall back to FIFO
    for (auto mode : modes) {
        if (mode == VK_PRESENT_MODE_MAILBOX_KHR) {
            present_mode_ = mode;
            return true;
        }
    }
    present_mode_ = VK_PRESENT_MODE_FIFO_KHR;
    return true;
}

bool Swapchain::pick_extent(VulkanDevice& device, uint32_t width, uint32_t height) {
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device.physical_device(), device.surface(), &caps);

    if (caps.currentExtent.width != UINT32_MAX) {
        extent_ = caps.currentExtent;
    } else {
        extent_.width = std::max(caps.minImageExtent.width,
                      std::min(caps.maxImageExtent.width, width));
        extent_.height = std::max(caps.minImageExtent.height,
                       std::min(caps.maxImageExtent.height, height));
    }
    return true;
}

bool Swapchain::create_swapchain(VulkanDevice& device) {
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device.physical_device(), device.surface(), &caps);

    uint32_t image_count = caps.minImageCount + 1;
    if (image_count > caps.maxImageCount && caps.maxImageCount > 0)
        image_count = caps.maxImageCount;

    VkSurfaceFormatKHR surface_format{};
    surface_format.format = format_;
    surface_format.colorSpace = color_space_;

    VkPresentModeKHR present_mode = present_mode_;

    uint32_t queue_families[] = {device.graphics_queue_family(), device.present_queue_family()};
    bool separate = device.graphics_queue_family() != device.present_queue_family();

    VkSwapchainCreateInfoKHR sci{};
    sci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    sci.surface = device.surface();
    sci.minImageCount = image_count;
    sci.imageFormat = surface_format.format;
    sci.imageColorSpace = surface_format.colorSpace;
    sci.imageExtent = extent_;
    sci.imageArrayLayers = 1;
    sci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    sci.imageSharingMode = separate ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
    sci.queueFamilyIndexCount = separate ? 2 : 0;
    sci.pQueueFamilyIndices = separate ? queue_families : nullptr;
    sci.preTransform = caps.currentTransform;
    sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sci.presentMode = present_mode;
    sci.clipped = VK_TRUE;

    VkResult res = vkCreateSwapchainKHR(device.device(), &sci, nullptr, &swapchain_);
    if (res != VK_SUCCESS) return false;

    vkGetSwapchainImagesKHR(device.device(), swapchain_, &image_count_, nullptr);
    swapchain_images_.resize(image_count_);
    vkGetSwapchainImagesKHR(device.device(), swapchain_, &image_count_, swapchain_images_.data());

    return true;
}

void Swapchain::create_image_views(VulkanDevice& device) {
    swapchain_views_.resize(swapchain_images_.size());
    for (size_t i = 0; i < swapchain_images_.size(); ++i) {
        VkImageViewCreateInfo ivci{};
        ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ivci.image = swapchain_images_[i];
        ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ivci.format = format_;
        ivci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        ivci.subresourceRange.levelCount = 1;
        ivci.subresourceRange.layerCount = 1;
        vkCreateImageView(device.device(), &ivci, nullptr, &swapchain_views_[i]);
    }
}

void Swapchain::create_render_pass(VulkanDevice& device) {
    VkAttachmentDescription color_att{};
    color_att.format = format_;
    color_att.samples = VK_SAMPLE_COUNT_1_BIT;
    color_att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color_att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color_att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color_att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color_att.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference color_ref{};
    color_ref.attachment = 0;
    color_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_ref;

    VkRenderPassCreateInfo rpci{};
    rpci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpci.attachmentCount = 1;
    rpci.pAttachments = &color_att;
    rpci.subpassCount = 1;
    rpci.pSubpasses = &subpass;

    VkResult rp_result = vkCreateRenderPass(device.device(), &rpci, nullptr, &render_pass_);
    std::cerr << "  [swapchain] Render pass (color-only) result=" << rp_result << " handle=" << (uintptr_t)render_pass_ << "\n";
}

void Swapchain::create_framebuffers(VulkanDevice& device) {
    framebuffers_.resize(swapchain_views_.size());
    for (size_t i = 0; i < swapchain_views_.size(); ++i) {
        VkFramebufferCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fci.renderPass = render_pass_;
        fci.attachmentCount = 1;
        fci.pAttachments = &swapchain_views_[i];
        fci.width = extent_.width;
        fci.height = extent_.height;
        fci.layers = 1;
        vkCreateFramebuffer(device.device(), &fci, nullptr, &framebuffers_[i]);
    }
}

} // namespace delirium