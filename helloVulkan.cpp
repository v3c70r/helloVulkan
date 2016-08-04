#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>
#include <vector>
#include <stdexcept>
#include <functional>
#include <cstring>
#include <set>
#include <limits>

const int WIDTH = 800;
const int HEIGHT = 600;

const std::vector<const char*> validationLayers = {
    "VK_LAYER_LUNARG_standard_validation"
};

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

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
            cleanup();
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
        }

    private:
        VDeleter<VkInstance> instance {vkDestroyInstance};
        VDeleter<VkDevice> device{vkDestroyDevice};
        VDeleter<VkDebugReportCallbackEXT> callback{instance, DestroyDebugReportCallbackEXT};
        VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
        VDeleter<VkSurfaceKHR> surface{instance, vkDestroySurfaceKHR};
        VDeleter<VkSwapchainKHR> swapChain{device, vkDestroySwapchainKHR};
        std::vector<VkImage> swapChainImages;
        std::vector<VDeleter<VkImageView>> swapChainImageViews;
        VkFormat swapChainImageFormat;
        VkExtent2D swapChainExtent;
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
            glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

            window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
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

        void initVulkan() {
            createInstance();
            setupDebugCallback();
            createSurface();
            pickPhysicalDevice();
            createLogicalDevice();
            createSwapChain();
            createImageViews();
        }

        void createGraphicsPipeline() {
        }

        void createImageViews() {
            swapChainImageViews.resize(swapChainImages.size(), VDeleter<VkImageView>{device, vkDestroyImageView});
            for (uint32_t i = 0; i<swapChainImages.size(); i++){
                VkImageViewCreateInfo createInfo = {};
                createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                createInfo.image = swapChainImages[i];
                createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
                createInfo.format = swapChainImageFormat;
                createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
                createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
                createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
                createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
                createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                createInfo.subresourceRange.baseMipLevel = 0;
                createInfo.subresourceRange.levelCount = 1;
                createInfo.subresourceRange.baseArrayLayer = 0;
                createInfo.subresourceRange.layerCount = 1;
                if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
                    throw std::runtime_error("failed to create image views!");
                }
                std::cout<<swapChainImageViews.size()<<std::endl;


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
            createInfo.oldSwapchain = VK_NULL_HANDLE;
            if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
                throw std::runtime_error("failed to create swap chain!");
            }
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

            // Create a queue
            QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

            std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
            std::set<int> uniqueQueueFamilise = {indices.graphicsFamily, indices.presentFamily};

            for (int queueFamily: uniqueQueueFamilise) {

                VkDeviceQueueCreateInfo queueCreateInfo = {};
                queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
                queueCreateInfo.queueFamilyIndex = indices.graphicsFamily;
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

            std::cout<< "Found "<<layerCount <<" layers\n";

            for (const auto & layerProperty: availableLayers)
                std::cout<<"Available layer: "<<layerProperty.layerName<<std::endl;

            // Check if validation layer is in available layers
            for (const char* layerName : validationLayers) {

                bool layerFound = false;
                std::cout<<"Checking layer: "<<layerName<<std::endl;

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

            std::cout<<"Required glfw extensions:\n";
            for (const auto & extension: extensions)
                std::cout<<extension<<std::endl;

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
            }
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

