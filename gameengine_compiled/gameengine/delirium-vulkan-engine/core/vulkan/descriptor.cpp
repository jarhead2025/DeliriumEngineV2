#include "descriptor.hpp"
#include <array>
#include <cstring>

namespace delirium {

bool DescriptorManager::initialize(VulkanDevice& device, VkDescriptorSetLayout layout,
                                    uint32_t frame_count) {
    create_descriptor_pool(device, frame_count);
    allocate_descriptor_sets(device, layout, frame_count);

    ubos_.resize(frame_count);
    for (uint32_t i = 0; i < frame_count; ++i) {
        VkDeviceSize size = sizeof(UniformBufferObject);
        create_buffer(device, size,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      ubos_[i].buffer, ubos_[i].memory);
        map_buffer(device, size, ubos_[i].memory, &ubos_[i].mapped);
    }
    return true;
}

void DescriptorManager::shutdown(VulkanDevice& device) {
    for (auto& ubo : ubos_) {
        vkUnmapMemory(device.device(), ubo.memory);
        vkDestroyBuffer(device.device(), ubo.buffer, nullptr);
        vkFreeMemory(device.device(), ubo.memory, nullptr);
    }
    vkDestroyDescriptorPool(device.device(), descriptor_pool_, nullptr);
}

void DescriptorManager::update_ubo(VulkanDevice& /*device*/, uint32_t frame_index,
                                    const UniformBufferObject& ubo) {
    if (ubos_[frame_index].mapped) {
        std::memcpy(ubos_[frame_index].mapped, &ubo, sizeof(ubo));
    }
}

void DescriptorManager::bind_surfel_buffer(VulkanDevice& device, uint32_t frame_index,
                                            VkBuffer buffer) {
    VkDescriptorBufferInfo buffer_info{};
    buffer_info.buffer = buffer;
    buffer_info.offset = 0;
    buffer_info.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptor_sets_[frame_index];
    write.dstBinding = 1;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &buffer_info;

    vkUpdateDescriptorSets(device.device(), 1, &write, 0, nullptr);
}

void DescriptorManager::create_descriptor_pool(VulkanDevice& device, uint32_t frame_count) {
    std::array<VkDescriptorPoolSize, 2> pool_sizes;
    pool_sizes[0] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, frame_count};
    pool_sizes[1] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, frame_count};

    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
    pool_info.pPoolSizes = pool_sizes.data();
    pool_info.maxSets = frame_count * static_cast<uint32_t>(pool_sizes.size());

    vkCreateDescriptorPool(device.device(), &pool_info, nullptr, &descriptor_pool_);
}

void DescriptorManager::allocate_descriptor_sets(VulkanDevice& device, VkDescriptorSetLayout layout,
                                                   uint32_t frame_count) {
    descriptor_sets_.resize(frame_count);

    std::vector<VkDescriptorSetLayout> layouts(frame_count, layout);
    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = descriptor_pool_;
    alloc_info.descriptorSetCount = frame_count;
    alloc_info.pSetLayouts = layouts.data();

    vkAllocateDescriptorSets(device.device(), &alloc_info, descriptor_sets_.data());

    // Update each descriptor set with its UBO (binding 0)
    for (uint32_t i = 0; i < frame_count; ++i) {
        VkDescriptorBufferInfo buffer_info{};
        buffer_info.buffer = ubos_[i].buffer;
        buffer_info.offset = 0;
        buffer_info.range = sizeof(UniformBufferObject);

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = descriptor_sets_[i];
        write.dstBinding = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        write.pBufferInfo = &buffer_info;

        vkUpdateDescriptorSets(device.device(), 1, &write, 0, nullptr);
    }
}

bool DescriptorManager::create_buffer(VulkanDevice& device, VkDeviceSize size,
                                        VkMemoryPropertyFlags properties,
                                        VkBuffer& buffer, VkDeviceMemory& memory) {
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device.device(), &buffer_info, nullptr, &buffer) != VK_SUCCESS)
        return false;

    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(device.device(), buffer, &mem_reqs);

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = device.find_memory_type(mem_reqs.memoryTypeBits, properties);

    return vkAllocateMemory(device.device(), &alloc_info, nullptr, &memory) == VK_SUCCESS &&
           vkBindBufferMemory(device.device(), buffer, memory, 0) == VK_SUCCESS;
}

void DescriptorManager::map_buffer(VulkanDevice& device, VkDeviceSize size,
                                    VkDeviceMemory& memory, void** data) {
    vkMapMemory(device.device(), memory, 0, size, 0, data);
}

} // namespace delirium