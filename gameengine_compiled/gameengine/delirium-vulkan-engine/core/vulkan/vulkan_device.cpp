#include "vulkan_device.hpp"
#include <SDL_vulkan.h>
#include <iostream>

namespace delirium {

bool VulkanDevice::initialize(SDL_Window* window, const std::vector<const char*> extra_layers) {
    if (!create_instance(window, extra_layers)) {
        std::cerr << "  [Vulkan] create_instance FAILED\n";
        return false;
    }
    std::cerr << "  [Vulkan] instance created OK\n";

    if (!create_surface(window)) {
        std::cerr << "  [Vulkan] create_surface FAILED\n";
        return false;
    }
    std::cerr << "  [Vulkan] surface created OK\n";

    if (!pick_physical_device()) {
        std::cerr << "  [Vulkan] pick_physical_device FAILED\n";
        return false;
    }
    std::cerr << "  [Vulkan] physical device picked OK\n";

    if (!create_logical_device()) {
        std::cerr << "  [Vulkan] create_logical_device FAILED\n";
        return false;
    }
    std::cerr << "  [Vulkan] logical device created OK\n";

    get_queues();

    vkGetPhysicalDeviceProperties(physical_device_, &device_properties_);
    vkGetPhysicalDeviceFeatures(physical_device_, &device_features_);
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &memory_properties_);

    return true;
}

void VulkanDevice::shutdown() {
    if (device_) vkDestroyDevice(device_, nullptr);
    if (surface_) vkDestroySurfaceKHR(instance_, surface_, nullptr);
    if (instance_) vkDestroyInstance(instance_, nullptr);
}

bool VulkanDevice::create_instance(SDL_Window* window, const std::vector<const char*>& extra_layers) {
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "DeliRium Engine";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "DeliRium";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;

    // Collect instance extensions: always include VK_KHR_surface,
    // then try SDL — if SDL returns nothing, add common Linux surface extensions manually.
    std::vector<const char*> extensions;
    extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);

    uint32_t sdl_ext_count = 0;
    const char* sdl_extensions[32] = {};
    SDL_Vulkan_GetInstanceExtensions(window, &sdl_ext_count, sdl_extensions);
    std::cerr << "  [Vulkan] SDL reported " << (int)sdl_ext_count << " instance extensions\n";

    if (sdl_ext_count > 0) {
        for (uint32_t i = 0; i < sdl_ext_count; ++i)
            extensions.push_back(sdl_extensions[i]);
    } else {
        std::cerr << "  [Vulkan] SDL returned 0 extensions, adding surface extensions manually\n";
        extensions.push_back("VK_KHR_xlib_surface");
        extensions.push_back("VK_KHR_wayland_surface");
    }

    create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    create_info.ppEnabledExtensionNames = extensions.data();
    std::cerr << "  [Vulkan] Enabling " << (int)extensions.size() << " instance extensions:\n";
    for (const auto& ext : extensions)
        std::cerr << "    - " << ext << "\n";

#ifdef NDEBUG
    create_info.enabledLayerCount = 0;
    (void)extra_layers;
#else
    std::vector<const char*> layers(extra_layers.begin(), extra_layers.end());
    layers.push_back("VK_LAYER_KHRONOS_validation");
    create_info.enabledLayerCount = static_cast<uint32_t>(layers.size());
    create_info.ppEnabledLayerNames = layers.data();
#endif

    VkResult result = vkCreateInstance(&create_info, nullptr, &instance_);
    if (result != VK_SUCCESS) {
        std::cerr << "  [Vulkan] vkCreateInstance returned " << result
                  << " (SDL error: " << SDL_GetError() << ")\n";
        return false;
    }
    return true;
}

bool VulkanDevice::create_surface(SDL_Window* window) {
    if (!window) {
        std::cerr << "  [Vulkan] create_surface: window is null\n";
        return false;
    }
    if (instance_ == VK_NULL_HANDLE) {
        std::cerr << "  [Vulkan] create_surface: instance is null\n";
        return false;
    }
    SDL_bool result = SDL_Vulkan_CreateSurface(window, instance_, &surface_);
    if (result != SDL_TRUE) {
        std::cerr << "  [Vulkan] SDL_Vulkan_CreateSurface failed (SDL error: "
                  << SDL_GetError() << ")\n";
        return false;
    }
    return true;
}

bool VulkanDevice::pick_physical_device() {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance_, &count, nullptr);
    if (count == 0) return false;

    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance_, &count, devices.data());

    for (VkPhysicalDevice pd : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(pd, &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            physical_device_ = pd;
            break;
        }
        if (!physical_device_) physical_device_ = pd;
    }
    return physical_device_ != VK_NULL_HANDLE;
}

bool VulkanDevice::create_logical_device() {
    VkPhysicalDeviceFeatures features{};
    device_features_ = features;

    uint32_t queue_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(queue_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_count, families.data());

    uint32_t gq = 0, pq = 0;
    bool found_gq = false, found_pq = false;

    for (uint32_t i = 0; i < queue_count; ++i) {
        if (!found_gq && (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            gq = i; found_gq = true;
        }
        VkBool32 supports_present = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(physical_device_, i, surface_, &supports_present);
        if (!found_pq && supports_present) {
            pq = i; found_pq = true;
        }
        if (found_gq && found_pq) break;
    }
    if (!found_gq || !found_pq) return false;

    graphics_queue_family_ = gq;
    present_queue_family_ = pq;

    float priority = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> queue_creates;

    if (gq == pq) {
        VkDeviceQueueCreateInfo qc{};
        qc.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qc.queueFamilyIndex = gq;
        qc.queueCount = 1;
        qc.pQueuePriorities = &priority;
        queue_creates.push_back(qc);
    } else {
        VkDeviceQueueCreateInfo qc_g{}, qc_p{};
        qc_g.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qc_g.queueFamilyIndex = gq; qc_g.queueCount = 1; qc_g.pQueuePriorities = &priority;
        qc_p.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qc_p.queueFamilyIndex = pq; qc_p.queueCount = 1; qc_p.pQueuePriorities = &priority;
        queue_creates = {qc_g, qc_p};
    }

    VkDeviceCreateInfo di{};
    di.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    di.queueCreateInfoCount = static_cast<uint32_t>(queue_creates.size());
    di.pQueueCreateInfos = queue_creates.data();
    di.pEnabledFeatures = &features;

    // Enable VK_KHR_swapchain (required for vkCreateSwapchainKHR, vkAcquireNextImageKHR, etc.)
    const char* device_extensions[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
    di.enabledExtensionCount = 1;
    di.ppEnabledExtensionNames = device_extensions;

    return vkCreateDevice(physical_device_, &di, nullptr, &device_) == VK_SUCCESS;
}

void VulkanDevice::get_queues() {
    vkGetDeviceQueue(device_, graphics_queue_family_, 0, &graphics_queue_);
    vkGetDeviceQueue(device_, present_queue_family_, 0, &present_queue_);
}

VkFormat VulkanDevice::find_supported_format(const std::vector<VkFormat>& candidates,
                                              VkImageTiling tiling, VkFormatFeatureFlags features) {
    for (VkFormat fmt : candidates) {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(physical_device_, fmt, &props);
        if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features)) return fmt;
        if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features)) return fmt;
    }
    return VK_FORMAT_UNDEFINED;
}

VkFormat VulkanDevice::find_depth_format() {
    return find_supported_format(
        {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
        VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

uint32_t VulkanDevice::find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) {
    for (uint32_t i = 0; i < memory_properties_.memoryTypeCount; ++i) {
        if ((type_filter & (1 << i)) &&
            (memory_properties_.memoryTypes[i].propertyFlags & properties)) return i;
    }
    return 0xFFFFFFFF;
}

} // namespace delirium