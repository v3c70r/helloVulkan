#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLM_HAS_CXX11_STL
#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include "imgui/imgui.h"
#include "imgui_impl_glfw_vulkan.h"


#include <unordered_map>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <functional>
#include <cstring>
#include <set>
#include <limits>
#include <fstream>
#include <array>
#include <chrono>

static void check_vk_result(VkResult err)
{
    if (err == 0) return;
    printf("VkResult %d\n", err);
    if (err < 0) 
        abort();
}

const int WIDTH = 800;
const int HEIGHT = 600;

const std::string MODEL_PATH = "./mesh/chalet.obj";
const std::string TEXTURE_PATH = "./mesh/chalet.jpg";

const std::vector<const char*> validationLayers = {
    "VK_LAYER_LUNARG_standard_validation"
};

struct Vertex{
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec2 texCoord;

    static VkVertexInputBindingDescription getBindingDescription()
    {
        // Similar to a VAO
        VkVertexInputBindingDescription bindingDescription = {};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return bindingDescription;
    }

    bool operator==(const Vertex& other) const{
        return pos == other.pos && color == other.color && texCoord == other.texCoord;
    }

    static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions()
    {
        // Get attribute locations
        std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions = {};
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, color);

        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

        return attributeDescriptions;
    }
};

namespace std {
    template <> struct hash<Vertex> {
        size_t operator()(Vertex const& vertex) const{
            return ((hash<glm::vec3>()(vertex.pos) ^
                        (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^ 
                (hash<glm::vec2>()(vertex.texCoord) << 1);
        }
    };
}

//const std::vector<Vertex> vertices = {
//    {{-0.5f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
//    {{0.5f, -0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
//    {{0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
//    {{-0.5f, 0.5f, 0.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}},
//
//
//    {{-0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
//    {{0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
//    {{0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
//    {{-0.5f, 0.5f, -0.5f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}}
//
//};
//
//const std::vector<uint16_t> indices = {
//    0,1,2,2,3,0,
//    4, 5, 6, 6, 7, 4
//};

struct UniformBufferObject{
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};


const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};
#ifdef NDEBUG
const bool enableValidationLayers= false;
#else
const bool enableValidationLayers = true;
#endif

/////////////////// Helper functions /////////////
static std::vector<char> readFile(const std::string& fileName){
    std::ifstream file(fileName, std::ios::ate | std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Failed to open file");
    size_t fileSize = (size_t) file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
}

/////////////////// Callbacks ////////////////////



VkResult CreateDebugReportCallbackEXT(
        VkInstance instance, 
        const VkDebugReportCallbackCreateInfoEXT* pCreateInfo, 
        const VkAllocationCallbacks* pAllocator, 
        VkDebugReportCallbackEXT* pCallback) {
    auto func = (PFN_vkCreateDebugReportCallbackEXT) vkGetInstanceProcAddr(
            instance,
            "vkCreateDebugReportCallbackEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pCallback);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugReportCallbackEXT(VkInstance instance, VkDebugReportCallbackEXT callback
        , const VkAllocationCallbacks* pAllocator) {
    auto func = 
        (PFN_vkDestroyDebugReportCallbackEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT");
    if (func != nullptr) {
        func(instance, callback, pAllocator);
    }
}


/////////////////Main bodies///////////////////
//
struct QueueFamilyIndices {
    int graphicsFamily = -1;
    int presentFamily = -1;
    int computeFamily = -1;

    bool isComplete() {
        return graphicsFamily >= 0 && 
            presentFamily >=0;
    }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

template <typename T>
class VDeleter {
    public:
        VDeleter() : VDeleter([](T _) {}) {}

        VDeleter(std::function<void(T, VkAllocationCallbacks*)> deletef) {
            object = VK_NULL_HANDLE;
            this->deleter = [=](T obj) { deletef(obj, nullptr); };
        }

        VDeleter(const VDeleter<VkInstance>& instance, std::function<void(VkInstance, T, VkAllocationCallbacks*)> deletef) {
            object = VK_NULL_HANDLE;
            this->deleter = [&instance, deletef](T obj) { deletef(instance, obj, nullptr); };
        }

        VDeleter(const VDeleter<VkDevice>& device, std::function<void(VkDevice, T, VkAllocationCallbacks*)> deletef) {
            object = VK_NULL_HANDLE;
            this->deleter = [&device, deletef](T obj) { deletef(device, obj, nullptr); };
        }

        ~VDeleter() {
            cleanup();
        }

        T* operator &() {
            return &object;
        }

        operator T() const {
            return object;
        }

    private:
        T object;
        std::function<void(T)> deleter;

        void cleanup() {
            if (object != VK_NULL_HANDLE) {
                deleter(object);
            }
            object = VK_NULL_HANDLE;
        }
};

class HelloTriangleApplication {
    public:
        void run() {
            initWindow();
            initVulkan();
            mainLoop();
            ImGui_ImplGlfwVulkan_Shutdown();

        }

    private:

        // TODO: Investigate the order of destructions
        VDeleter<VkInstance> instance {vkDestroyInstance};
        VDeleter<VkDebugReportCallbackEXT> callback{instance, DestroyDebugReportCallbackEXT};
        VDeleter<VkSurfaceKHR> surface{instance, vkDestroySurfaceKHR};
        VDeleter<VkDevice> device{vkDestroyDevice};
        VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
        VDeleter<VkSwapchainKHR> swapChain{device, vkDestroySwapchainKHR};
        VDeleter<VkDescriptorSetLayout> descriptorSetLayout{device, vkDestroyDescriptorSetLayout};
        VDeleter<VkPipelineLayout> pipelineLayout{device, vkDestroyPipelineLayout};

        std::vector<VkImage> swapChainImages;
        std::vector<VDeleter<VkFramebuffer>> swapChainFramebuffers;
        std::vector<VDeleter<VkImageView>> swapChainImageViews;
        VkFormat swapChainImageFormat;
        VkExtent2D swapChainExtent;
        VDeleter<VkShaderModule> vertShaderModule{device, vkDestroyShaderModule};
        VDeleter<VkShaderModule> fragShaderModule{device, vkDestroyShaderModule};
        VDeleter<VkRenderPass> renderPass{device, vkDestroyRenderPass};
        VDeleter<VkRenderPass> guiRenderPass{device, vkDestroyRenderPass};
        VDeleter<VkPipeline> graphicsPipeline{device, vkDestroyPipeline};
        VDeleter<VkCommandPool> commandPool{device, vkDestroyCommandPool};
        VDeleter<VkCommandPool> dynamicCommandPool{device, vkDestroyCommandPool};   // This command pool contains buffers to draw frame dynamicly
        std::vector<VkCommandBuffer> commandBuffers;
        VkCommandBuffer dynamicCommandBuffer;

        // Two semaphores
        VDeleter<VkSemaphore> imageAvailableSemaphore{device, vkDestroySemaphore};
        VDeleter<VkSemaphore> renderFinishedSemaphore{device, vkDestroySemaphore};
        VDeleter<VkSemaphore> staticRenderFinishedSemaphore{device, vkDestroySemaphore};

        // Vertex buffer and index buffer
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;
        VDeleter<VkBuffer> vertexBuffer{device, vkDestroyBuffer};
        VDeleter<VkDeviceMemory> vertexBufferMemory{device, vkFreeMemory};
        VDeleter<VkBuffer> indexBuffer{device, vkDestroyBuffer};
        VDeleter<VkDeviceMemory> indexBufferMemory{device, vkFreeMemory};

        ///////////// Textures
        // Image buffers
        VDeleter<VkImage> textureImage{device, vkDestroyImage};
        VDeleter<VkDeviceMemory> textureImageMemory{device, vkFreeMemory};

        // Image view
        VDeleter<VkImageView> textureImageView{device, vkDestroyImageView};

        // Image Sampler
        VDeleter<VkSampler> textureSampler{device, vkDestroySampler};

        // Depth image
        VDeleter<VkImage> depthImage{device, vkDestroyImage};
        VDeleter<VkDeviceMemory> depthImageMemory{device, vkFreeMemory};
        VDeleter<VkImageView> depthImageView{device, vkDestroyImageView};

        // Uniform buffers
        VDeleter<VkDescriptorPool> descriptorPool{device, vkDestroyDescriptorPool};
        VkDescriptorSet descriptorSet;
        VDeleter<VkBuffer> uniformStagingBuffer{device, vkDestroyBuffer};
        VDeleter<VkDeviceMemory> uniformStagingBufferMemory{device, vkFreeMemory};
        VDeleter<VkBuffer> uniformBuffer{device, vkDestroyBuffer};
        VDeleter<VkDeviceMemory> uniformBufferMemory{device, vkFreeMemory};


        VkQueue presentQueue;
        VkQueue graphicsQueue;
        GLFWwindow* window;

        std::vector<const char*> getRequiredExtensions() {
            std::vector<const char*> extensions;

            unsigned int glfwExtensionCount = 0;
            const char** glfwExtensions;
            glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

            for (unsigned int i = 0; i < glfwExtensionCount; i++) {
                extensions.push_back(glfwExtensions[i]);
            }

            if (enableValidationLayers) {
                extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
            }

            return extensions;
        }
        void initWindow() {
            glfwInit();

            glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
            //glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

            window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
            ::glfwSetWindowUserPointer(window, this);
            ::glfwSetWindowSizeCallback(window, HelloTriangleApplication::onWindowResized);
        }

        static void onWindowResized(GLFWwindow* window, int width, int height)
        {
            if (width == 0 || height == 0) return;
            HelloTriangleApplication *app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
            app->recreateSwapChain();
        }

        void setupDebugCallback(){
            if (!enableValidationLayers) return;
            VkDebugReportCallbackCreateInfoEXT createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
            createInfo.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT;
            createInfo.pfnCallback = (PFN_vkDebugReportCallbackEXT) debugCallback;
            if (CreateDebugReportCallbackEXT(instance, &createInfo, nullptr, &callback) !=
                    VK_SUCCESS)
                throw std::runtime_error("Failed to set up debug callback!");

        }

        void initGUI()
        {
            VkAttachmentDescription attachment = {};
            attachment.format = swapChainImageFormat;
            attachment.samples = VK_SAMPLE_COUNT_1_BIT;
            attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

            // Depth attachment
            VkAttachmentDescription depthAttachment = {};
            depthAttachment.format = findDepthFormat();
            depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
            depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depthAttachment.initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

            VkAttachmentReference color_attachment = {};
            color_attachment.attachment = 0;
            color_attachment.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

            VkAttachmentReference depthAttachmentRef = {};
            depthAttachmentRef.attachment = 1;
            depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

            VkSubpassDescription subpass = {};
            subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass.colorAttachmentCount = 1;
            subpass.pColorAttachments = &color_attachment;
            subpass.pDepthStencilAttachment = &depthAttachmentRef;

            std::array<VkAttachmentDescription, 2> attachments = {attachment, depthAttachment};
            VkRenderPassCreateInfo info = {};
            info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
            info.attachmentCount = attachments.size();
            info.pAttachments = attachments.data();
            info.subpassCount = 1;
            info.pSubpasses = &subpass;
            if (vkCreateRenderPass(device, &info, nullptr, &guiRenderPass) != VK_SUCCESS)
                throw std::runtime_error( "Failed to create GUI renderpass");
            // Create render pass
            ::ImGui_ImplGlfwVulkan_Init_Data initData={};
            initData.allocator = nullptr;
            initData.gpu = physicalDevice;
            initData.device = device;
            initData.render_pass = guiRenderPass;
            initData.pipeline_cache = nullptr;
            initData.descriptor_pool = descriptorPool;
            initData.check_vk_result = check_vk_result;
            ImGui_ImplGlfwVulkan_Init(window, true, &initData);

            // Upload Fonts
            {
                VkCommandBuffer commandBuffer = beginSingleTimeCommands();
                ImGui_ImplGlfwVulkan_CreateFontsTexture(commandBuffer);
                endSingleTimeCommands(commandBuffer);
                ImGui_ImplGlfwVulkan_InvalidateFontUploadObjects();
                vkDeviceWaitIdle(device);
            }

            //bool show_test_window = true;
            //bool show_another_window = false;
            //ImVec4 clear_color = ImColor(114, 144, 154);


                

        }

        void initVulkan() {
            createInstance();
            setupDebugCallback();
            createSurface();
            pickPhysicalDevice();
            createLogicalDevice();
            createSwapChain();
            createImageViews();
            createRenderPass();
            createDescriptorSetLayout();
            createGraphicsPipeline();
            createCommandPool();
            createDepthResources();
            createFramebuffers();
            createTextureImage();
            createTextureImageView();
            createTextureSampler();
            loadModel();
            createVertexBuffer();
            createIndexBuffer();
            createUniformBuffer();
            createDescriptorPool();
            createDescriptorSet();
            initGUI();
            createCommandBuffers();
            createSemaphores();

        }

        void loadModel()
        {
            tinyobj::attrib_t attrib;
            std::vector<tinyobj::shape_t> shapes;
            std::vector<tinyobj::material_t> materials;
            std::string err;
            if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &err, MODEL_PATH.c_str()))
                throw std::runtime_error(err);
            std::unordered_map<Vertex, int> uniqueVertices = {};
            for (const auto& shape: shapes){
                for (const auto& index: shape.mesh.indices){
                    Vertex vertex = {};
                    vertex.pos = {
                        attrib.vertices[3 * index.vertex_index + 0],
                        attrib.vertices[3 * index.vertex_index + 1],
                        attrib.vertices[3 * index.vertex_index + 2]
                    };

                    vertex.texCoord = {
                        attrib.texcoords[2 * index.texcoord_index + 0],
                        1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
                    };
                    if (uniqueVertices.count(vertex) == 0){
                        uniqueVertices[vertex] = vertices.size();
                        vertices.push_back(vertex);
                    }
                    indices.push_back(uniqueVertices[vertex]);
                }
            }
        }

        void createDepthResources() {
            VkFormat depthFormat = findDepthFormat();
            createImage(swapChainExtent.width, swapChainExtent.height, depthFormat, 
                    VK_IMAGE_TILING_OPTIMAL, ::VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, ::VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);
            createImageView(depthImage, depthFormat, ::VK_IMAGE_ASPECT_DEPTH_BIT,depthImageView);
            transitionImageLayout(depthImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        }

        void createTextureSampler()
        {
            VkSamplerCreateInfo samplerInfo = {};
            samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            samplerInfo.magFilter = VK_FILTER_LINEAR;
            samplerInfo.minFilter = VK_FILTER_LINEAR;
            samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            samplerInfo.anisotropyEnable = VK_TRUE; // Free anistropy?
            samplerInfo.maxAnisotropy = 16;
            // What color beyond the boarder
            samplerInfo.borderColor = ::VK_BORDER_COLOR_INT_OPAQUE_BLACK;
            samplerInfo.unnormalizedCoordinates = VK_FALSE;
            // Useful for shadowmaping antialiasing
            samplerInfo.compareEnable = VK_FALSE;
            samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
            // Mipmap
            samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            samplerInfo.mipLodBias = 0.0f;
            samplerInfo.minLod = 0.0f;
            samplerInfo.maxLod = 0.0f;
            if (vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS)
                throw std::runtime_error("Failed to create texture sampler!");
        }

        void createTextureImageView()
        {
            // Create an image view to access texture image
            createImageView(textureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT,textureImageView);
        }
        void createTextureImage()
        {
            int texWidth, texHeight, texChannels;
            ::stbi_uc* pixels = stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight, &texChannels, ::STBI_rgb_alpha);
            VkDeviceSize imageSize = texWidth * texHeight * 4;
            if (!pixels)
                throw std::runtime_error("Failed to load texture image");

            VDeleter<VkImage> stagingImage{device, vkDestroyImage};
            VDeleter<VkDeviceMemory> stagingImageMemory{device, vkFreeMemory};

            // Create staging image
            createImage( texWidth, texHeight, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_LINEAR, VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingImage, stagingImageMemory);


            void* data;
            vkMapMemory(device, stagingImageMemory, 0, imageSize, 0, &data);
            memcpy(data, pixels, (size_t) imageSize);
            vkUnmapMemory(device, stagingImageMemory);

            // Free the host image
            stbi_image_free(pixels);

            // Create device image
            createImage(
                    texWidth, texHeight,
                    VK_FORMAT_R8G8B8A8_UNORM,
                    VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    textureImage,
                    textureImageMemory
                    );

            // Transit images' layouts to the layouts we want.
            //
            // Staging: PREINTIALIZED -> SRC
            // texture: PREINTIALIZED -> DST
            // COPY
            // texture: DST->SHADER_OPTIMAL
            transitionImageLayout(stagingImage, VK_IMAGE_LAYOUT_PREINITIALIZED, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            transitionImageLayout(textureImage, VK_IMAGE_LAYOUT_PREINITIALIZED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            copyImage(stagingImage, textureImage, texWidth, texHeight);
            transitionImageLayout(textureImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

            
        }

        void createDescriptorSet()
        {
            VkDescriptorSetLayout layouts[] = {descriptorSetLayout};
            VkDescriptorSetAllocateInfo allocInfo = {};
            allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocInfo.descriptorPool = descriptorPool;  // Bind descriptor to the pool
            allocInfo.descriptorSetCount = 1;
            allocInfo.pSetLayouts = layouts;

            if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet) != VK_SUCCESS) {
                throw std::runtime_error("failed to allocate descriptor set!");
            }

            VkDescriptorBufferInfo bufferInfo = {};
            bufferInfo.buffer = uniformBuffer;
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);

            VkDescriptorImageInfo imageInfo = {};
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo.imageView = textureImageView;
            imageInfo.sampler = textureSampler;

            std::array<::VkWriteDescriptorSet, 2> descriptorWrites = {};
            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = descriptorSet;
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo = &bufferInfo;

            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet = descriptorSet;
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pImageInfo = &imageInfo;



            vkUpdateDescriptorSets(device, descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);

        }


        void createDescriptorPool()
        {
            /*
            std::array<VkDescriptorPoolSize, 2> poolSizes = {};
            poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            poolSizes[0].descriptorCount = 100;
            poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            poolSizes[1].descriptorCount = 100;

            VkDescriptorPoolCreateInfo poolInfo = {};
            poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            poolInfo.poolSizeCount = poolSizes.size();
            poolInfo.pPoolSizes = poolSizes.data();
            poolInfo.maxSets = 2*100;
            */
            // Reconfig
            VkDescriptorPoolSize poolSizes[11] = 
            {
                { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
                { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
                { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
                { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
                { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
                { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
                { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
                { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
                { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
                { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
                { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
            };
            VkDescriptorPoolCreateInfo poolInfo = {};
            poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
            poolInfo.maxSets = 1000 * 11;
            poolInfo.poolSizeCount = 11;
            poolInfo.pPoolSizes = poolSizes;

            if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
                throw std::runtime_error("failed to create descriptor pool!");
            }
        }

        void createUniformBuffer(){
            VkDeviceSize bufferSize = sizeof(UniformBufferObject);

            createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformStagingBuffer, uniformStagingBufferMemory);
            createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, uniformBuffer, uniformBufferMemory);
        }
        void createDescriptorSetLayout()
        {
            VkDescriptorSetLayoutBinding uboLayoutBinding = {};
            uboLayoutBinding.binding = 0;
            uboLayoutBinding.descriptorCount = 1;
            uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            uboLayoutBinding.pImmutableSamplers = nullptr;
            uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

            // Sampler binding
            VkDescriptorSetLayoutBinding samplerLayoutBinding = {};
            samplerLayoutBinding.binding = 1;
            samplerLayoutBinding.descriptorCount = 1;
            samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            samplerLayoutBinding.pImmutableSamplers = nullptr;
            samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

            std::array<VkDescriptorSetLayoutBinding, 2> bindings = {uboLayoutBinding, samplerLayoutBinding};

            VkDescriptorSetLayoutCreateInfo layoutInfo = {};
            layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layoutInfo.bindingCount = bindings.size();
            layoutInfo.pBindings = bindings.data();

            if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
                throw std::runtime_error("failed to create descriptor set layout!");
            }
        }
        void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VDeleter<VkImage>& image, VDeleter<VkDeviceMemory>& imageMemory) {
            VkImageCreateInfo imageInfo = {};
            imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            imageInfo.imageType = VK_IMAGE_TYPE_2D;
            imageInfo.extent.width = width;
            imageInfo.extent.height = height;
            imageInfo.extent.depth = 1;
            imageInfo.mipLevels = 1;
            imageInfo.arrayLayers = 1;
            imageInfo.format = format;
            imageInfo.tiling = tiling;
            imageInfo.initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;
            imageInfo.usage = usage;
            imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
                throw std::runtime_error("failed to create image!");
            }

            VkMemoryRequirements memRequirements;
            vkGetImageMemoryRequirements(device, image, &memRequirements);

            VkMemoryAllocateInfo allocInfo = {};
            allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocInfo.allocationSize = memRequirements.size;
            allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

            if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
                throw std::runtime_error("failed to allocate image memory!");
            }

            vkBindImageMemory(device, image, imageMemory, 0);
        }

        void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VDeleter<VkBuffer>& buffer, VDeleter<VkDeviceMemory>& bufferMemory) {
            VkBufferCreateInfo bufferInfo = {};
            bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bufferInfo.size = size;
            bufferInfo.usage = usage;
            bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
                throw std::runtime_error("failed to create buffer!");
            }

            VkMemoryRequirements memRequirements;
            vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

            VkMemoryAllocateInfo allocInfo = {};
            allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocInfo.allocationSize = memRequirements.size;
            allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

            if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
                throw std::runtime_error("failed to allocate buffer memory!");
            }

            vkBindBufferMemory(device, buffer, bufferMemory, 0);
        }



        void createVertexBuffer()
        {
            VkDeviceSize bufferSize = sizeof(vertices[0])*vertices.size();

            /***********************
            // Using unified memory
            createBuffer(bufferSize, ::VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, ::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | ::VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, vertexBuffer, vertexBufferMemory);
            
            // OOoh, unified memory, good to know
            void *data;
            ::vkMapMemory(device, vertexBufferMemory, 0, bufferSize, 0, &data);
            ::memcpy(data, vertices.data(), (size_t) bufferSize);
            ::vkUnmapMemory(device, vertexBufferMemory);
            **********************/

            /****************/
            // Using device only memory
            // Host memory -> staging buffer -> vertex buffer

            VDeleter<VkBuffer> stagingBuffer{device, vkDestroyBuffer};
            VDeleter<VkDeviceMemory> stagingBufferMemory{device, vkFreeMemory};
            createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

            void* data;
            vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
            memcpy(data, vertices.data(), (size_t) bufferSize);
            vkUnmapMemory(device, stagingBufferMemory);

            // Create vertex buffer
            createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);

            // staging buffer -> vertex buffer
            copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

        }

        void createIndexBuffer()
        {
            ::VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();
            ::VDeleter<VkBuffer> stagingBuffer{ device, ::vkDestroyBuffer};
            VDeleter<::VkDeviceMemory> stagingBufferMemory{device, vkFreeMemory};
            createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
            void* data;
            vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
            memcpy(data, indices.data(), (size_t) bufferSize);
            vkUnmapMemory(device, stagingBufferMemory);

            createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

            copyBuffer(stagingBuffer, indexBuffer, bufferSize);
        }


        uint32_t findMemoryType(uint32_t typeFilter, ::VkMemoryPropertyFlags properties)
        {
            ::VkPhysicalDeviceMemoryProperties memProperties;
            ::vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
            for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
                if (typeFilter & (1<<i) &&
                (memProperties.memoryTypes[i].propertyFlags & properties) == properties) 
                    return i;
            throw std::runtime_error("failed to find suitable memory type!");
        }

        void recreateSwapChain(){
            vkDeviceWaitIdle(device);

            createSwapChain();
            createImageViews();
            createRenderPass();
            createGraphicsPipeline();
            createDepthResources();
            createFramebuffers();
            createCommandBuffers();

        }

        void createSemaphores(){
            VkSemaphoreCreateInfo semaphoreInfo = {};
            semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphore) != VK_SUCCESS ||
                    vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphore) != VK_SUCCESS ||
                    vkCreateSemaphore(device, &semaphoreInfo, nullptr, &staticRenderFinishedSemaphore) != VK_SUCCESS 
                    ) {

                throw std::runtime_error("failed to create semaphores!");
            }
        }

        void createCommandBuffers()
        {

            if (commandBuffers.size()>0){
                vkFreeCommandBuffers(device, commandPool, commandBuffers.size(), commandBuffers.data());
                vkFreeCommandBuffers(device, dynamicCommandPool, 1, &dynamicCommandBuffer);
            }

            // https://vulkan-tutorial.com/Drawing_a_triangle/Drawing/Command_buffers
            commandBuffers.resize(swapChainFramebuffers.size());
            VkCommandBufferAllocateInfo allocInfo = {};
            allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocInfo.commandPool = commandPool;
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocInfo.commandBufferCount = (uint32_t) commandBuffers.size();

            if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS)
                throw std::runtime_error("failed to allocate command buffers!");

            allocInfo.commandPool = dynamicCommandPool;
            allocInfo.commandBufferCount = 1;
            if (vkAllocateCommandBuffers(device, &allocInfo, &dynamicCommandBuffer) != VK_SUCCESS)
                throw std::runtime_error("failed to allocate command buffers!");


            // Record command buffer
            for (size_t i=0; i < commandBuffers.size(); i++)
            {
                VkCommandBufferBeginInfo beginInfo = {};
                beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
                beginInfo.pInheritanceInfo = nullptr;
                vkBeginCommandBuffer(commandBuffers[i], &beginInfo);

                // Starting render pass
                VkRenderPassBeginInfo  renderPassInfo = {};
                renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
                renderPassInfo.renderPass = renderPass;
                renderPassInfo.framebuffer = swapChainFramebuffers[i];

                renderPassInfo.renderArea.offset = {0, 0};
                renderPassInfo.renderArea.extent = swapChainExtent;

                std::array<VkClearValue, 2> clearValues = {};
                clearValues[0].color = {0.0f, 0.0f, 0.0f, 1.0f};
                clearValues[1].depthStencil = {1.0f, 0};
                renderPassInfo.clearValueCount = clearValues.size();
                renderPassInfo.pClearValues = clearValues.data();

                vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
                vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
                VkBuffer vertexBuffers[] = {vertexBuffer};
                VkDeviceSize offsets[] = {0};
                ::vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, vertexBuffers, offsets);
                ::vkCmdBindIndexBuffer(commandBuffers[i], indexBuffer, 0, ::VK_INDEX_TYPE_UINT32);
                ::vkCmdBindDescriptorSets(commandBuffers[i], ::VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
                //vkCmdDraw(commandBuffers[i], 
                //        3,  // Number of vertices
                //        1,  // instance count
                //        0,  // First offset
                //        0); // First instance
                
                // Draw with index buffer
                vkCmdDrawIndexed(commandBuffers[i],
                        indices.size(),1,0,0,0);


                vkCmdEndRenderPass(commandBuffers[i]);
                if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS)
                    throw std::runtime_error("Failed to record command buffer!");
            }


        }

        void createCommandPool()
        {
            QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);
            VkCommandPoolCreateInfo poolInfo = {};
            poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily;
            poolInfo.flags = 0;
            if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS)
                throw std::runtime_error("Failed to create command pool!");

            poolInfo.flags |= ::VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            if (vkCreateCommandPool(device, &poolInfo, nullptr, &dynamicCommandPool) != VK_SUCCESS)
                throw std::runtime_error("Failed to create command pool!");

            
        }

        void createFramebuffers()
        {
            // Size of framebuffers eqaul to size of swapChainImages
            swapChainFramebuffers.resize(swapChainImageViews.size(), VDeleter<VkFramebuffer>{device, vkDestroyFramebuffer});


            // Create a framebuffer for each swapchainimage
            for (size_t i = 0; i < swapChainImageViews.size(); i++) {
                std::array<VkImageView, 2> attachments = {
                    swapChainImageViews[i], depthImageView
                };

                VkFramebufferCreateInfo framebufferInfo = {};
                framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
                framebufferInfo.renderPass = renderPass;
                framebufferInfo.attachmentCount = attachments.size();
                framebufferInfo.pAttachments = attachments.data();
                framebufferInfo.width = swapChainExtent.width;
                framebufferInfo.height = swapChainExtent.height;
                framebufferInfo.layers = 1;

                if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
                    throw std::runtime_error("failed to create framebuffer!");
                }
            }
        }


        void createShaderModule(const std::vector<char> &code, VDeleter<VkShaderModule> &VkShaderModule)
        {
            VkShaderModuleCreateInfo createInfo={};
            createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            createInfo.codeSize = code.size();
            createInfo.pCode = (uint32_t*)code.data();
            if (vkCreateShaderModule(device, &createInfo, nullptr, &VkShaderModule) != VK_SUCCESS)
                throw std::runtime_error("Failed to create shader module!");
        }
        void createGraphicsPipeline() {
            auto vertShaderCode = readFile("./shaders/vert.spv");
            auto fragShaderCode = readFile("./shaders/frag.spv");
            createShaderModule(vertShaderCode, vertShaderModule);
            createShaderModule(fragShaderCode, fragShaderModule);
            // There's no linking infromations above
            // Create vert pipeline
            VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
            vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
            vertShaderStageInfo.module = vertShaderModule;
            vertShaderStageInfo.pName = "main";

            // Create frag pipeline
            VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
            fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            fragShaderStageInfo.module = fragShaderModule;
            fragShaderStageInfo.pName = "main";

            VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

            // Specify vertex input
            // Create vertex buffer
            auto bindingDescription = Vertex::getBindingDescription();
            auto attributeDescriptions = Vertex::getAttributeDescriptions();

            VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
            vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            vertexInputInfo.vertexBindingDescriptionCount = 1;
            vertexInputInfo.pVertexBindingDescriptions = &bindingDescription; // Optional
            vertexInputInfo.vertexAttributeDescriptionCount = attributeDescriptions.size();
            vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data(); // Optional

            // Pipeline assembly infos
            // How primitives are assembled, e.g. POINT_LIST, LINE_LIST ...
            VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
            inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            inputAssembly.primitiveRestartEnable = VK_FALSE;

            // Viewport
            VkViewport viewport = {};
            viewport.x = 0.0f;
            viewport.y = 0.0f;
            viewport.width = (float) swapChainExtent.width;
            viewport.height = (float) swapChainExtent.height;
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;

            // rectangle scissor 
            VkRect2D scissor = {};
            scissor.offset = {0, 0};
            scissor.extent = swapChainExtent;

            // Create
            VkPipelineViewportStateCreateInfo viewportState = {};
            viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            viewportState.viewportCount = 1;
            viewportState.pViewports = &viewport;
            viewportState.scissorCount = 1;
            viewportState.pScissors = &scissor;

            // Rasterizer
            VkPipelineRasterizationStateCreateInfo rasterizer = {};
            rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            rasterizer.depthClampEnable = VK_FALSE;
            rasterizer.rasterizerDiscardEnable = VK_FALSE;
            rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
            rasterizer.lineWidth = 1.0f;
            rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
            rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

            rasterizer.depthBiasEnable = VK_FALSE;
            rasterizer.depthBiasConstantFactor = 0.0f; // Optional
            rasterizer.depthBiasClamp = 0.0f; // Optional
            rasterizer.depthBiasSlopeFactor = 0.0f; // Optional

            // MSAA
            VkPipelineMultisampleStateCreateInfo multisampling = {};
            multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            multisampling.sampleShadingEnable = VK_FALSE;
            multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
            multisampling.minSampleShading = 1.0f; // Optional
            multisampling.pSampleMask = nullptr; /// Optional
            multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
            multisampling.alphaToOneEnable = VK_FALSE; // Optional

            // Stencil
            //
            // Color blending
            VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
            colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            colorBlendAttachment.blendEnable = VK_FALSE;
            colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
            colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
            colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // Optional
            colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
            colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
            colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; // Optional

            VkPipelineColorBlendStateCreateInfo colorBlending = {};
            colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            colorBlending.logicOpEnable = VK_FALSE;
            colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
            colorBlending.attachmentCount = 1;
            colorBlending.pAttachments = &colorBlendAttachment;
            colorBlending.blendConstants[0] = 0.0f; // Optional
            colorBlending.blendConstants[1] = 0.0f; // Optional
            colorBlending.blendConstants[2] = 0.0f; // Optional
            colorBlending.blendConstants[3] = 0.0f; // Optional

            /* Those functions can be dynamic, too
             * VkDynamicState dynamicStates[] = {
             * VK_DYNAMIC_STATE_VIEWPORT,
             * VK_DYNAMIC_STATE_LINE_WIDTH
             * };

             * VkPipelineDynamicStateCreateInfo dynamicState = {};
             * dynamicState.dynamicStateCount = 2;
             * dynamicState.pDynamicStates = dynamicStates;
             */

            // Pipeline layout, i.e. uniforms
            VkDescriptorSetLayout setLayouts[] = {descriptorSetLayout};
            VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
            pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pipelineLayoutInfo.setLayoutCount = 1; // Optional
            pipelineLayoutInfo.pSetLayouts = setLayouts; // Optional
            pipelineLayoutInfo.pushConstantRangeCount = 0; // Optional
            pipelineLayoutInfo.pPushConstantRanges = 0; // Optional

            if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr,
                        &pipelineLayout) != VK_SUCCESS) {
                throw std::runtime_error("failed to create pipeline layout!");
            }

            // Enable depth test
            ::VkPipelineDepthStencilStateCreateInfo depthStencil = {};
            depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            depthStencil.depthTestEnable = VK_TRUE;
            depthStencil.depthWriteEnable = VK_TRUE;
            depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
            depthStencil.depthBoundsTestEnable = VK_FALSE;
            depthStencil.minDepthBounds = 0.0f; // Optional
            depthStencil.maxDepthBounds = 1.0f; // Optional
            // Disable stencil
            depthStencil.stencilTestEnable = VK_FALSE;
            depthStencil.front = {}; // Optional
            depthStencil.back = {}; // Optional
            // Finally, create the pipeline
            VkGraphicsPipelineCreateInfo pipelineInfo = {};
            pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pipelineInfo.stageCount = 2;
            pipelineInfo.pStages = shaderStages;
            pipelineInfo.pVertexInputState = &vertexInputInfo;
            pipelineInfo.pInputAssemblyState = &inputAssembly;
            pipelineInfo.pViewportState = &viewportState;
            pipelineInfo.pRasterizationState = &rasterizer;
            pipelineInfo.pMultisampleState = &multisampling;
            pipelineInfo.pDepthStencilState = &depthStencil; // Optional
            pipelineInfo.pColorBlendState = &colorBlending;
            pipelineInfo.pDynamicState = nullptr; // Optional
            pipelineInfo.layout = pipelineLayout;
            pipelineInfo.renderPass = renderPass;
            pipelineInfo.subpass = 0;

            pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
            pipelineInfo.basePipelineIndex = -1;

            if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS){
                throw std::runtime_error("failed to create graphics pipeline!");
            }
            

        }

        void createRenderPass()
        {
            // Create attachment
            VkAttachmentDescription colorAttachment = {};
            colorAttachment.format = swapChainImageFormat;
            colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
            colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

            // Depth attachment
            VkAttachmentDescription depthAttachment = {};
            depthAttachment.format = findDepthFormat();
            depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
            depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depthAttachment.initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

            VkAttachmentReference colorAttachmentRef = {};
            colorAttachmentRef.attachment = 0;      //  Attachment binding point
            colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

            // Depth attachment reference
            VkAttachmentReference depthAttachmentRef = {};
            depthAttachmentRef.attachment = 1;
            depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

            VkSubpassDescription subPass = {};
            subPass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subPass.colorAttachmentCount = 1;
            subPass.pColorAttachments = &colorAttachmentRef;
            subPass.pDepthStencilAttachment = &depthAttachmentRef;

            std::array<VkAttachmentDescription, 2> attachments = {colorAttachment, depthAttachment};
            VkRenderPassCreateInfo renderPassInfo = {};
            renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
            renderPassInfo.attachmentCount = attachments.size();
            renderPassInfo.pAttachments = attachments.data();
            renderPassInfo.subpassCount = 1;
            renderPassInfo.pSubpasses = &subPass;

            // Subpass dependencies
            VkSubpassDependency dependency = {};
            dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
            dependency.dstSubpass = 0;
            dependency.srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
            dependency.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
            dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            renderPassInfo.dependencyCount = 1;
            renderPassInfo.pDependencies = &dependency;

            if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
                throw std::runtime_error("failed to create render pass!");
            }
        }

        void createImageViews() {
            swapChainImageViews.resize(swapChainImages.size(), VDeleter<VkImageView>{device, vkDestroyImageView});
            for (uint32_t i = 0; i<swapChainImages.size(); i++){
                createImageView(swapChainImages[i], swapChainImageFormat,::VK_IMAGE_ASPECT_COLOR_BIT, swapChainImageViews[i]);

            }
        }
        void createSwapChain() {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);
            VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
            VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
            VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

            uint32_t imageCount = swapChainSupport.capabilities.minImageCount +1;

            if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
            {
                imageCount = swapChainSupport.capabilities.maxImageCount;
            }

            // Create swapchain with info
            VkSwapchainCreateInfoKHR createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
            createInfo.surface = surface;
            createInfo.minImageCount = imageCount;
            createInfo.imageFormat = surfaceFormat.format;
            createInfo.imageColorSpace = surfaceFormat.colorSpace;
            createInfo.imageExtent = extent;
            createInfo.imageArrayLayers = 1;
            createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

            QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
            uint32_t queueFamilyIndices[] = {(uint32_t) indices.graphicsFamily, (uint32_t) indices.presentFamily};

            if (indices.graphicsFamily != indices.presentFamily) {
                createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
                createInfo.queueFamilyIndexCount = 2;
                createInfo.pQueueFamilyIndices = queueFamilyIndices;
            } else {
                createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
                createInfo.queueFamilyIndexCount = 0; // Optional
                createInfo.pQueueFamilyIndices = nullptr; // Optional
            }
            createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
            createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
            createInfo.presentMode = presentMode;
            createInfo.clipped = VK_TRUE;

            VkSwapchainKHR oldSwapchain = swapChain;
            createInfo.oldSwapchain = oldSwapchain;

            VkSwapchainKHR newSwapchain;
            if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &newSwapchain) != VK_SUCCESS) {
                throw std::runtime_error("failed to create swap chain!");
            }
            *&swapChain = newSwapchain;
            vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
            swapChainImages.resize(imageCount);
            vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

            swapChainImageFormat = surfaceFormat.format;
            swapChainExtent = extent;

        }
        VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
            if (availableFormats.size() == 1 && availableFormats[0].format == VK_FORMAT_UNDEFINED) {
                return {VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
            }

            for (const auto& availableFormat : availableFormats) {
                if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                    return availableFormat;
                }
            }
            return availableFormats[0];
        }

        VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR> availablePresentModes) {
            for (const auto& availablePresentMode : availablePresentModes) {
                if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                    return availablePresentMode;
                }
            }

            return VK_PRESENT_MODE_FIFO_KHR;
        }

        VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
            if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
            {
                return capabilities.currentExtent;
            }
            else{
                VkExtent2D actualExtent = {WIDTH, HEIGHT};
                actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
                actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));
                return actualExtent;
            }
        }


        SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device)
        {
            SwapChainSupportDetails details;

            // Query capabilities
            vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

            // Query formats
            uint32_t formatCount;
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

            if (formatCount !=0 ) {
                details.formats.resize(formatCount);
                vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
            }
            uint32_t presentModeCount;
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

            if (presentModeCount != 0) {
                details.presentModes.resize(presentModeCount);
                vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
            }


            
            
            return details;
        }

        void createSurface(){
            if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS){
                throw std::runtime_error("failed to create window surface!");
            }

        }

        void createLogicalDevice(){

            // Query queue familise
            QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

            std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;

            // Set contains unique set of numbers
            // Fall into the same queue here
            std::set<int> uniqueQueueFamilise = {indices.graphicsFamily, indices.presentFamily};

            // Create two queues for graphics family and indices.presentFamily
            for (int queueFamily: uniqueQueueFamilise) {

                VkDeviceQueueCreateInfo queueCreateInfo = {};
                queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
                queueCreateInfo.queueFamilyIndex = queueFamily;
                queueCreateInfo.queueCount = 1;
                float queuePriority = 1.0f;
                queueCreateInfo.pQueuePriorities = &queuePriority;
                queueCreateInfos.push_back(queueCreateInfo);

            }
            VkPhysicalDeviceFeatures deviceFeatures = {};

            // Create logical device with a queue
            VkDeviceCreateInfo createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
            createInfo.pQueueCreateInfos = queueCreateInfos.data();
            createInfo.queueCreateInfoCount = (uint32_t)queueCreateInfos.size();
            std::cout<<"Queue numbers "<<(uint32_t)queueCreateInfos.size()<<std::endl;
            createInfo.pEnabledFeatures = &deviceFeatures;

            // Add extensiosns
            createInfo.enabledExtensionCount = deviceExtensions.size();
            createInfo.ppEnabledExtensionNames = deviceExtensions.data();

            // enable validation layer
            if (enableValidationLayers) {
                createInfo.enabledLayerCount = validationLayers.size();
                createInfo.ppEnabledLayerNames = validationLayers.data();
            } else {
                createInfo.enabledLayerCount = 0;
            }

            // Instantiate logical device
            if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
                throw std::runtime_error("failed to create logical device!");
            }


            // Get queue handle
            vkGetDeviceQueue(device, indices.graphicsFamily, 0, &graphicsQueue);
            vkGetDeviceQueue(device, indices.presentFamily, 0, &presentQueue);
        }

        QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
            QueueFamilyIndices indices;
            uint32_t queueFamilyCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

            std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
            vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());
            int i = 0;
            for (const auto& queueFamily : queueFamilies) {
                if (queueFamily.queueCount > 0 && queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                    indices.graphicsFamily = i;
                    std::cout<<"Graphics family at "<<i<<std::endl;
                }
                if (queueFamily.queueCount > 0 && queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT) {

                    std::cout<<"Compute family at "<<i<<std::endl;
                }
                if (queueFamily.queueCount > 0 && queueFamily.queueFlags & VK_QUEUE_TRANSFER_BIT) {
                    std::cout<<"transfer family at "<<i<<std::endl;
                }
                if (queueFamily.queueCount > 0 && queueFamily.queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) {
                    std::cout<<"queue_sparse family at "<<i<<std::endl;
                }
                VkBool32 presentSupport = false;
                vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
                if (queueFamily.queueCount > 0 && presentSupport)
                    indices.presentFamily = i;
                if (indices.isComplete()) {
                    break;
                }
                i++;
            }
            return indices;
        }


        void pickPhysicalDevice() {
            uint32_t deviceCount = 0;
            vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
            if (deviceCount == 0)
                throw std::runtime_error("Failed to find GPUs with Vulkan support");
            std::vector<VkPhysicalDevice> devices(deviceCount);
            vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
            // Use the first device
            for (const auto& device: devices){
                if (isDeviceSuitable(device)){
                    physicalDevice = device;
                    break;
                }
            }
            if (physicalDevice == VK_NULL_HANDLE){
                throw std::runtime_error("failed to find a suitable GPU!");
            }
        }

        /*
         * Check if a physical device suitable for this application
         */
        bool isDeviceSuitable(VkPhysicalDevice device)
        {
            QueueFamilyIndices indices = findQueueFamilies(device);
            bool extensionSupported = checkDeviceExtensionSupport(device);
            bool swapChainAdequate = false;
            if (extensionSupported) {
                SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
                swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
            }
            return indices.isComplete() && extensionSupported && swapChainAdequate;
        }

        bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
            uint32_t extensionCount;
            vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
            std::vector<VkExtensionProperties> availableExtensions(extensionCount);
            vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());
            std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());
            for (const auto &extension: availableExtensions){
                requiredExtensions.erase(extension.extensionName);
            }
            return requiredExtensions.empty();
        }
        


        bool checkValidationLayerSupport() {
            /*
             * Check validation layer support
             */
            uint32_t layerCount;
            // Get number of layers
            vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
            std::vector<VkLayerProperties> availableLayers(layerCount);
            // Write validation stuff to vector
            vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());


            // Check if validation layer is in available layers
            for (const char* layerName : validationLayers) {
                bool layerFound = false;
                for (const auto& layerProperties : availableLayers) {
                    if (strcmp(layerName, layerProperties.layerName) == 0) {
                        layerFound = true;
                        break;
                    }
                }
                if (!layerFound) {
                    return false;
                }
            }
            return true;
        }
        void createInstance()
        {
            if (enableValidationLayers && !checkValidationLayerSupport()) {
                throw std::runtime_error("validation layers requested, but not available!");
            }
            // Construct VkApplicationInfo struct
            VkApplicationInfo appInfo = {};
            appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
            appInfo.pApplicationName = "Hello Triangle";
            appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
            appInfo.pEngineName = "No Engine";
            appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
            appInfo.apiVersion = VK_API_VERSION_1_0;

            // Construct VkInstanceCreateInfo struct with VkApplicationInfo
            VkInstanceCreateInfo createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
            createInfo.pApplicationInfo = &appInfo;

            // Get extensions
            //unsigned int glfwExtensionCount = 0;
            //const char** glfwExtensions;
            //glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

            //// Query vkEumerateInstanceExtensionProperties
            //uint32_t extensionCount = 0;
            //vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
            //std::vector<VkExtensionProperties> extensions(extensionCount);
            //vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());
            //std::cout<<"Aavailable extensions\n";
            //for (const auto &extension: extensions)
            //    std::cout<<"\t"<<extension.extensionName<<std::endl;
            //// Assign extensions to createInfo
            //createInfo.enabledExtensionCount = glfwExtensionCount;
            //createInfo.ppEnabledExtensionNames = glfwExtensions;
            
            auto extensions = getRequiredExtensions();


            createInfo.enabledExtensionCount = extensions.size();
            createInfo.ppEnabledExtensionNames = extensions.data();


            // Validation layer?
            if (enableValidationLayers){
                createInfo.enabledLayerCount = validationLayers.size();
                createInfo.ppEnabledLayerNames = validationLayers.data();
            }
            else
                createInfo.enabledLayerCount = 0;

            // Finally, create the instance
            if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
                throw std::runtime_error("failed to create instance!");
            }
        }

        void mainLoop() {
            while (!glfwWindowShouldClose(window)) {
                glfwPollEvents();
                updateUniformBuffer();
                ImGui_ImplGlfwVulkan_NewFrame();
                drawFrame();
            }
            vkDeviceWaitIdle(device);
        }

        void updateUniformBuffer()
        {
            static auto startTime = std::chrono::high_resolution_clock::now();
            auto currentTime = std::chrono::high_resolution_clock::now();
            float time = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime).count() /  1000.0f;

            // Define matrices
            UniformBufferObject ubo = {};
            ubo.model = glm::rotate( glm::mat4(), time*0.5f*glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
            ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
            ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width/(float)swapChainExtent.height, 0.1f, 10.f);
            // Flip Y coordinate in clip space !!
            // WTF, different from OpenGL
            ubo.proj[1][1] *= -1;
            void* data;
            // Since the memory is bound already, uploading to the memory will change the uniform
            vkMapMemory(device, uniformStagingBufferMemory, 0, sizeof(ubo), 0, &data);
            memcpy(data, &ubo, sizeof(ubo));
            vkUnmapMemory(device, uniformStagingBufferMemory);

            copyBuffer(uniformStagingBuffer, uniformBuffer, sizeof(ubo));
        }

        void drawFrame(){
            uint32_t imageIndex;
            
            // Grab next image from swapchain
            vkAcquireNextImageKHR(device, swapChain, std::numeric_limits<uint64_t>::max(), imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);
                ImGui_ImplGlfwVulkan_NewFrame();

            // Draw static command buffers
            VkSubmitInfo submitInfo = {};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

            //Submit recorded command buffer
            VkSemaphore waitSemaphores[] = {imageAvailableSemaphore};
            VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
            submitInfo.waitSemaphoreCount = 1;
            submitInfo.pWaitSemaphores = waitSemaphores;
            submitInfo.pWaitDstStageMask = waitStages;

            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

            // Signal when command buffers finished execution
            VkSemaphore signalSemaphores[] = {staticRenderFinishedSemaphore};
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = signalSemaphores;
            if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
                throw std::runtime_error("failed to submit draw command buffer!");
            }

            // Finished rendering recorded command buffer, now, create a new command
            // buffer to render dynamic stuffs.
            
                // reset command buffer
                ::vkResetCommandBuffer(dynamicCommandBuffer, 0);
                ///// Start recording
                ::VkCommandBufferBeginInfo cmdBufferInfo = {};
                cmdBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                cmdBufferInfo.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                if (vkBeginCommandBuffer(dynamicCommandBuffer, &cmdBufferInfo) != VK_SUCCESS)
                    throw std::runtime_error("FAiled to start command buffer");

                //TODO: How to not clean the framebuffer
                VkRenderPassBeginInfo info = {};
                info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
                info.renderPass = guiRenderPass;
                info.framebuffer = swapChainFramebuffers[imageIndex];
                info.renderArea.offset = {0,0};
                info.renderArea.extent = swapChainExtent;
                std::array<VkClearValue, 2> clearValues = {};
                clearValues[0].color = {0.0f, 0.0f, 0.0f, 1.0f};
                clearValues[1].depthStencil = {1.0f, 0};
                info.clearValueCount = clearValues.size();
                info.pClearValues = clearValues.data();
                vkCmdBeginRenderPass(dynamicCommandBuffer, &info, VK_SUBPASS_CONTENTS_INLINE);

                ImGui::Text("HELLO");
                ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

                ImGui_ImplGlfwVulkan_Render(dynamicCommandBuffer);

                
                vkCmdEndRenderPass(dynamicCommandBuffer);
                ::vkEndCommandBuffer(dynamicCommandBuffer);

                //////////// Finished recording


                submitInfo = {};
                submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                submitInfo.commandBufferCount = 1;
                submitInfo.pCommandBuffers = &dynamicCommandBuffer;


                // Wait for semaphore
                submitInfo.waitSemaphoreCount = 1;
                submitInfo.pWaitSemaphores = signalSemaphores;
                submitInfo.pWaitDstStageMask = waitStages;

                // Signal semaphore
                VkSemaphore dynamicSignal[] = {renderFinishedSemaphore};
                submitInfo.signalSemaphoreCount = 1;
                submitInfo.pSignalSemaphores = dynamicSignal;

                if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
                    throw std::runtime_error("failed to submit draw command buffer!");
                }
                //vkDeviceWaitIdle(device);


            VkPresentInfoKHR presentInfo = {};
            presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;


            presentInfo.waitSemaphoreCount = 1;
            presentInfo.pWaitSemaphores = dynamicSignal;

            VkSwapchainKHR swapChains[] = {swapChain};
            presentInfo.swapchainCount = 1;
            presentInfo.pSwapchains = swapChains;

            presentInfo.pImageIndices = &imageIndex;

            // Show this image
            vkQueuePresentKHR(presentQueue, &presentInfo);
     
        }

        /////// Helper functions
        /// Fire command that only run onece
        // Those two functions act like a decorator in Python
        VkCommandBuffer beginSingleTimeCommands() {
            VkCommandBufferAllocateInfo allocInfo = {};
            allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocInfo.commandPool = commandPool;
            allocInfo.commandBufferCount = 1;

            VkCommandBuffer commandBuffer;
            vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

            VkCommandBufferBeginInfo beginInfo = {};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

            vkBeginCommandBuffer(commandBuffer, &beginInfo);

            return commandBuffer;
        }

        void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
            vkEndCommandBuffer(commandBuffer);

            VkSubmitInfo submitInfo = {};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &commandBuffer;

            vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
            vkQueueWaitIdle(graphicsQueue);

            vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
        }

        // Copy from VK_BUFFER_USAGE_TRANSFER_SRC_BIT to VK_BUFFER_USAGE_DST_BIT buffer
        void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
        {
            VkCommandBuffer commandBuffer = beginSingleTimeCommands();

            VkBufferCopy copyRegion = {};
            copyRegion.size = size;
            vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

            endSingleTimeCommands(commandBuffer);
        }

        // Transit image layout
        void transitionImageLayout(VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout) {
            VkCommandBuffer commandBuffer = beginSingleTimeCommands();
            VkImageMemoryBarrier barrier = {};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.oldLayout = oldLayout;
            barrier.newLayout = newLayout;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = image;
            if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
                barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            } else {
                barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            }

            barrier.subresourceRange.baseMipLevel = 0;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.baseArrayLayer = 0;
            barrier.subresourceRange.layerCount = 1;

            //
            // Preinitialized → transfer source: transfer reads should wait on host writes
            // Preinitialized → transfer destination: transfer writes should wait on host writes
            // Transfer destination → shader reading: shader reads should wait on transfer writes
            //
            if (oldLayout == VK_IMAGE_LAYOUT_PREINITIALIZED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
                barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            } else if (oldLayout == VK_IMAGE_LAYOUT_PREINITIALIZED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
                barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            } else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
                barrier.srcAccessMask = 0;
                barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            }
            else
            {
                throw std::invalid_argument("unsupported layout transition!");
            }
            vkCmdPipelineBarrier(
                    commandBuffer,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    0,
                    0, nullptr,
                    0, nullptr,
                    1, &barrier
                    );


            endSingleTimeCommands(commandBuffer);
        }

        // Copy one image to another
        void copyImage(VkImage srcImage, VkImage dstImage, uint32_t width, uint32_t height) {
            VkCommandBuffer commandBuffer = beginSingleTimeCommands();

            // Define which mip level, etc of image to copy
            VkImageSubresourceLayers subResource = {};
            subResource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            subResource.baseArrayLayer = 0;
            subResource.mipLevel = 0;
            subResource.layerCount = 1;

            // Define an extent in image to copy
            VkImageCopy region = {};
            region.srcSubresource = subResource;
            region.dstSubresource = subResource;
            region.srcOffset = {0, 0, 0};
            region.dstOffset = {0, 0, 0};
            region.extent.width = width;
            region.extent.height = height;
            region.extent.depth = 1;

            // Do the copy
            vkCmdCopyImage(
                    commandBuffer,
                    srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    1, &region
                    );

            endSingleTimeCommands(commandBuffer);
        }

        void createImageView(VkImage image, VkFormat format, ::VkImageAspectFlags aspectFlags ,VDeleter<VkImageView>& imageView) {
            // Abstact image view creation
            VkImageViewCreateInfo viewInfo = {};
            viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            viewInfo.image = image;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = format;
            viewInfo.subresourceRange.aspectMask = aspectFlags;
            viewInfo.subresourceRange.baseMipLevel = 0;
            viewInfo.subresourceRange.levelCount = 1;
            viewInfo.subresourceRange.baseArrayLayer = 0;
            viewInfo.subresourceRange.layerCount = 1;

            if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
                throw std::runtime_error("failed to create texture image view!");
            }
        }



        VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
            for (::VkFormat format : candidates){
                ::VkFormatProperties props;
                ::vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);
                if (tiling == ::VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) 
                    return format;
                else if (tiling == ::VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features ) == features)
                    return format;
            }
            throw std::runtime_error("Failed to find supported format!");

        }

        VkFormat findDepthFormat() {
            return findSupportedFormat(
                    {::VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, ::VK_FORMAT_D24_UNORM_S8_UINT}, VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
        }
        ////////////////// Callback functions
        static VkBool32 debugCallback(
                VkDebugReportFlagsEXT flags,
                VkDebugReportObjectTypeEXT objType,
                uint64_t obj,
                size_t location,
                int32_t code,
                const char* layerPrefix,
                const char* msg,
                void* userData) {

            std::cerr << "validation layer: " << msg << std::endl;

            return VK_FALSE;
        }
};


int main() {
    HelloTriangleApplication app;

    try {
        app.run();
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

