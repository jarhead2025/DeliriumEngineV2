#pragma once
#include <vulkan/vulkan.h>
#include <SDL.h>
#include <vector>
#include <string>

namespace delirium {

class VulkanDevice {
public:
    bool initialize(SDL_Window* window, const std::vector<const char*> extra_layers = {});
    void shutdown();

    VkInstance instance() const { return instance_; }
    VkPhysicalDevice physical_device() const { return physical_device_; }
    VkDevice device() const { return device_; }
    VkSurfaceKHR surface() const { return surface_; }

    uint32_t graphics_queue_family() const { return graphics_queue_family_; }
    uint32_t present_queue_family() const { return present_queue_family_; }
    VkQueue graphics_queue() const { return graphics_queue_; }
    VkQueue present_queue() const { return present_queue_; }

    VkPhysicalDeviceProperties device_properties() const { return device_properties_; }
    VkPhysicalDeviceFeatures device_features() const { return device_features_; }
    VkPhysicalDeviceMemoryProperties memory_properties() const { return memory_properties_; }

    VkFormat find_supported_format(const std::vector<VkFormat>& candidates,
                                    VkImageTiling tiling, VkFormatFeatureFlags features);
    VkFormat find_depth_format();
    uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties);

    // Check if 2D storage images are supported (for surfel splatting)
    bool supports_storage_image() const { return storage_image_; }

private:
    bool create_instance(SDL_Window* window, const std::vector<const char*>& layers);
    bool create_surface(SDL_Window* window);
    bool pick_physical_device();
    bool create_logical_device();
    void get_queues();

    VkInstance instance_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;

    uint32_t graphics_queue_family_ = 0;
    uint32_t present_queue_family_ = 0;
    VkQueue graphics_queue_ = VK_NULL_HANDLE;
    VkQueue present_queue_ = VK_NULL_HANDLE;

    VkPhysicalDeviceProperties device_properties_;
    VkPhysicalDeviceFeatures device_features_;
    VkPhysicalDeviceMemoryProperties memory_properties_;

    bool storage_image_ = false;
};

} // namespace delirium