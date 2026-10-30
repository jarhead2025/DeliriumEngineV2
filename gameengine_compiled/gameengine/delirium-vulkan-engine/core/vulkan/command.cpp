#include "command.hpp"

namespace delirium {

bool CommandBuffers::initialize(VulkanDevice& device, Swapchain& swapchain) {
    create_command_pool(device);
    create_command_buffers(device, swapchain);
    create_sync_objects(device, swapchain);
    return true;
}

void CommandBuffers::shutdown(VulkanDevice& device) {
    for (uint32_t i = 0; i < max_frames_in_flight_; ++i) {
        vkDestroySemaphore(device.device(), image_available_semaphores_[i], nullptr);
        vkDestroySemaphore(device.device(), render_finished_semaphores_[i], nullptr);
        vkDestroyFence(device.device(), in_flight_fences_[i], nullptr);
    }
    vkDestroyCommandPool(device.device(), command_pool_, nullptr);
}

void CommandBuffers::create_command_pool(VulkanDevice& device) {
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = device.graphics_queue_family();
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(device.device(), &pool_info, nullptr, &command_pool_);
}

void CommandBuffers::create_command_buffers(VulkanDevice& device, Swapchain& /*swapchain*/) {
    command_buffers_.resize(max_frames_in_flight_);

    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = command_pool_;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = static_cast<uint32_t>(command_buffers_.size());

    vkAllocateCommandBuffers(device.device(), &alloc_info, command_buffers_.data());
}

void CommandBuffers::create_sync_objects(VulkanDevice& device, Swapchain& /*swapchain*/) {
    image_available_semaphores_.resize(max_frames_in_flight_);
    render_finished_semaphores_.resize(max_frames_in_flight_);
    in_flight_fences_.resize(max_frames_in_flight_);

    VkSemaphoreCreateInfo semaphore_info{};
    semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (uint32_t i = 0; i < max_frames_in_flight_; ++i) {
        vkCreateSemaphore(device.device(), &semaphore_info, nullptr, &image_available_semaphores_[i]);
        vkCreateSemaphore(device.device(), &semaphore_info, nullptr, &render_finished_semaphores_[i]);
        vkCreateFence(device.device(), &fence_info, nullptr, &in_flight_fences_[i]);
    }
}

void CommandBuffers::begin_frame(VulkanDevice& device) {
    vkWaitForFences(device.device(), 1, &in_flight_fences_[current_],
                        VK_TRUE, UINT64_MAX);

    vkResetFences(device.device(), 1, &in_flight_fences_[current_]);

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    vkBeginCommandBuffer(command_buffers_[current_], &begin_info);
}

void CommandBuffers::end_frame(VulkanDevice& device, Swapchain& swapchain, uint32_t image_index) {
    vkEndCommandBuffer(command_buffers_[current_]);

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkSemaphore wait_semaphores[] = {image_available_semaphores_[current_]};
    VkPipelineStageFlags wait_stages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submit_info.waitSemaphoreCount = 1;
    submit_info.pWaitSemaphores = wait_semaphores;
    submit_info.pWaitDstStageMask = wait_stages;

    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffers_[current_];

    VkSemaphore signal_semaphores[] = {render_finished_semaphores_[current_]};
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = signal_semaphores;

    vkQueueSubmit(device.graphics_queue(), 1, &submit_info, in_flight_fences_[current_]);

    VkPresentInfoKHR present_info{};
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores = signal_semaphores;

    VkSwapchainKHR swapchains[] = {swapchain.swapchain()};
    present_info.swapchainCount = 1;
    present_info.pSwapchains = swapchains;
    present_info.pImageIndices = &image_index;

    vkQueuePresentKHR(device.present_queue(), &present_info);

    current_ = (current_ + 1) % max_frames_in_flight_;
}

} // namespace delirium