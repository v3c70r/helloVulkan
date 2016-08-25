	
Vulkan seems to be a very verbos yet powerful graphics API. It takes more than 800 lines of code to draw a triangle with hard coded vertices in the shader. However, due to its very low level control over graphics hardwares, the overhead is lower than OpenGL. Also, learning Vulkan provides a alternative way to have more understanding on graphics rendering pipeline. This markdown file is used to note some concept come along the code. 

# Creation of Objects

Creation of objects in Vulkan is governed by a create information structure. For examle, `VkApplicationInfo` provides creation information of an application, which is used by `VkInstanceCreateInfo`. Create informations are passed to a create function to create corresponding object. In this case, `vkCreateInstance()` is called  by passing `VkInstanceCreateInfo` to generate an instance. 

# Pipeline Initialization

## Create Instance
Unlike OpenGL, Vulkan eliminated global states to avoid unwanted default behaviors. Instance is used to store the pre-application configurations as well as its communication with hardware. Obviously, most of the information can be inferred from their create information structure.  The typedef [struct VkInstanceCreateInfo](https://www.khronos.org/registry/vulkan/specs/1.0/xhtml/vkspec.html#VkInstanceCreateInfo) goes like this:

```
typedef struct VkInstanceCreateInfo {
    VkStructureType             sType;
    const void*                 pNext;
    VkInstanceCreateFlags       flags;
    const VkApplicationInfo*    pApplicationInfo;
    uint32_t                    enabledLayerCount;
    const char* const*          ppEnabledLayerNames;
    uint32_t                    enabledExtensionCount;
    const char* const*          ppEnabledExtensionNames;
} VkInstanceCreateInfo;
```

Note that in there are some more details to fill, namely, Application info, extensions and layers.

### Application Info

```
typedef struct VkApplicationInfo {
    VkStructureType    sType;
    const void*        pNext;
    const char*        pApplicationName;
    uint32_t           applicationVersion;
    const char*        pEngineName;
    uint32_t           engineVersion;
    uint32_t           apiVersion;
} VkApplicationInfo;
```
[Application info](https://www.khronos.org/registry/vulkan/specs/1.0/xhtml/vkspec.html#VkApplicationInfo) contains application informations. 

### Extensions

Extensions is listed as a set if extension names. In this case, our extension info is queried by the [`glfwGetRequiredInstanceExtensions()`](http://www.glfw.org/docs/latest/group__vulkan.html). Also, since we will use validation layer, `VK_EXT_DEBUG_REPORT_EXTENSION_NAME` is also required. 
By running the example, the following extensions are required in our application. 
```
VK_KHR_surface
VK_KHR_xcb_surface
VK_EXT_debug_report
```

### Layers

For now, the only enabled layer is validation layer: `VK_LAYER_LUNARG_standard_validation`.  Desired layer is checked over all of available layers to test support. Along with the validation layer, there are totally 9 layers supported by my machine. 
```
Found 9 layers
Available layer: VK_LAYER_LUNARG_device_limits
Available layer: VK_LAYER_LUNARG_image
Available layer: VK_LAYER_GOOGLE_threading
Available layer: VK_LAYER_GOOGLE_unique_objects
Available layer: VK_LAYER_LUNARG_object_tracker
Available layer: VK_LAYER_LUNARG_core_validation
Available layer: VK_LAYER_LUNARG_parameter_validation
Available layer: VK_LAYER_LUNARG_swapchain
Available layer: VK_LAYER_LUNARG_standard_validation
```
Validation layer is provided to validate Vulkan function calls. 

`Extensions` and `Layers` are [Extended Functionality](https://www.khronos.org/registry/vulkan/specs/1.0/xhtml/vkspec.html#extended-functionality-layers) of Vulkan. 

## setupDebugCallback
Looks like Java LWJGL has a good [explanation](http://javadoc.lwjgl.org/org/lwjgl/vulkan/VkDebugReportCallbackCreateInfoEXT.html) on this struct. 
When validation layer is enabled, a debug callback is compulsory to invoke when validation failed. The create info of debug callback is:

```
// From mesa mailing list
typedef struct VkDebugReportCallbackCreateInfoEXT {
    VkStructureType                 sType;
    const void*                     pNext;
    VkDebugReportFlagsEXT           flags;
    PFN_vkDebugReportCallbackEXT    pfnCallback;
    void*                           pUserData;
} VkDebugReportCallbackCreateInfoEXT;
```

Where the `pfnCallback` is a pointer to callback function, given as:

```
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
```
Obviously, this function simply output the message to `cerr`. 

## createSurface
The Vulkan surface is similar to a window. It depends on different platforms. Fortunately, we have GLFW to handle this part of work. [`glfwCreateWindowsSurface`](http://www.glfw.org/docs/latest/group__vulkan.html#ga1a24536bec3f80b08ead18e28e6ae965) will create a surface we want. Surface is highly depended on platforms. 

## pickPhysicalDevice
A host may have multiple physical device that can run our application. In order to decide which physical device to use, Vulkan provides [vkEnumeratePhysicalDevices](https://www.khronos.org/registry/vulkan/specs/1.0/man/html/vkEnumeratePhysicalDevices.html) to enumerate all available physical devices. return a arry of `VkPhysicalDevice` handles.

### Check if the device is compatible
Compatibility check is performed by query the physical devices' properties, extension support and swap chain adequate. 

#### Queue Family properties
Queue family describes the capability of queues in the family. Here by checking queue family properties, we need to find at least one queue supports `VK_QUEUE_GRAPHICS_BIT` and `presentation` checked by [`vkGetPhysicalDeviceSurfaceSupportKHR`](https://www.khronos.org/registry/vulkan/specs/1.0-wsi_extensions/xhtml/vkspec.html#vkGetPhysicalDeviceQueueFamilyProperties). 

#### Extension supports

We have queried required extensions from GLFW, now it's time to check whether they are supported by our device. By enumerating device extension properties with [vkEnumerateDeviceExtensionProperties](https://www.khronos.org/registry/vulkan/specs/1.0/man/html/vkEnumerateDeviceExtensionProperties.html), we can easily have it checked.

#### Swap chain support
One last thing to check is swap chain support. 

Swap chain is an extensions since not all applications need one.  So we need to ask for compatibility. 
* Swap chain presents frames

Swap chain support is defined by this `SwapChainSupportDetails` struct:
```
struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
}
```
* [VkSurfaceCapabilitiesKHR](http://nopper.tv/Vulkan/1.0/VkSurfaceCapabilitiesKHR.html) defines the capability of the swap chain like `minImageCount`, `maxImageCount`, etc. 
* `VkSurfaceFormatKHR` describes the format of each pixel and color space.
* `VkPresentModeKHR` describes how the images are presented to the screen. 

```
typedef enum VkPresentModeKHR {
    VK_PRESENT_MODE_IMMEDIATE_KHR = 0,
    VK_PRESENT_MODE_MAILBOX_KHR = 1,
    VK_PRESENT_MODE_FIFO_KHR = 2,
    VK_PRESENT_MODE_FIFO_RELAXED_KHR = 3,
} VkPresentModeKHR;
```
In this case, we only need to make sure that the supported format is not empty and the present mode is not empty. 
```
swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty()
```

By checking all of the devices, we simply pick the first supported device as our physical device. The next step is to create a logical device, which is an abstraction of the physical device. 

##createLogicalDevice
A Vulkan [logical device](https://www.khronos.org/registry/vulkan/specs/1.0/xhtml/vkspec.html#VkDevice) is a connection to physical device. Similar to creating an instance, it requires a create info structure as well as a bunch of informations. 
```
typedef struct VkDeviceCreateInfo {
    VkStructureType                    sType;
    const void*                        pNext;
    VkDeviceCreateFlags                flags;
    uint32_t                           queueCreateInfoCount;
    const VkDeviceQueueCreateInfo*     pQueueCreateInfos;
    uint32_t                           enabledLayerCount;
    const char* const*                 ppEnabledLayerNames;
    uint32_t                           enabledExtensionCount;
    const char* const*                 ppEnabledExtensionNames;
    const VkPhysicalDeviceFeatures*    pEnabledFeatures;
} VkDeviceCreateInfo;
```

### Queue Creation
The first thing we need to specify in the struct is `VkDeviceQueueCreateInfo*`. 
```
typedef struct VkDeviceQueueCreateInfo {
    VkStructureType             sType;
    const void*                 pNext;
    VkDeviceQueueCreateFlags    flags;
    uint32_t                    queueFamilyIndex;
    uint32_t                    queueCount;
    const float*                pQueuePriorities;
} VkDeviceQueueCreateInfo;
```
The queue family index we can extract from physical device queue families. In this case, we need to create one queue of `graphicsFamily` and one queue of `presentFamily`.

### Features
`VkphysicalDeviceFeatures` describes the features we want to use. Here we leave it to empty. 

### Extensions and layers
Extensions are the extensions we used earlier and the only surface used here is validation surface.

### Get queue handle
The last step is to get a queue handle for each queue family we want to use later. 
```
vkGetDeviceQueue(device, indices.graphicsFamily, 0, &graphicsQueue);
vkGetDeviceQueue(device, indices.presentFamily, 0, &presentQueue);
```
## Create Swap Chain

A swap chain is a series of images waiting to show to the screen, as well as a surface to draw on.  The create info is as following:
```
typedef struct VkSwapchainCreateInfoKHR {
  VkStructureType sType;
  const void* pNext;
  VkSwapchainCreateFlagsKHR flags;
  VkSurfaceKHR surface;
  uint32_t minImageCount;
  VkFormat imageFormat;
  VkColorSpaceKHR imageColorSpace;
  VkExtent2D imageExtent;
  uint32_t imageArrayLayers;
  VkImageUsageFlags imageUsage;
  VkSharingMode imageSharingMode;
  uint32_t queueFamilyIndexCount;
  const uint32_t* pQueueFamilyIndices;
  VkSurfaceTransformFlagBitsKHR preTransform;
  VkCompositeAlphaFlagBitsKHR compositeAlpha;
  VkPresentModeKHR presentMode;
  VkBool32 clipped;
  VkSwapchainKHR oldSwapchain;
} VkSwapchainCreateInfoKHR
```
Firstly, swap chain support is tested to return a swap chain support details, like what we have done in [selecting physical device](#swap-chain-support). 

Then we need to choose a format for surface. Each `VkSurfaceFormatKHR` entry contains `format` and `colorSpace` member. Here we want it to be `{VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR}`. Thus, if the surface has no restrictions on surface format, the preferred format will be used. Otherwise, a query is needed to determine the support. A swap chain also contains 

Present mode is also checked. Present mode describes how the images in swap chain are presented to the surface. In this case, we choose `VK_PRESENT_MODE_MAILBOX_KHR`. 

The size of framebuffer is described by `VkExtent2D`. We set it to window size. 

Number of images in the swap chain is set to `minImageCount +1`.

After creation of swap chain, a set of handles are created to retrieve images from swap chain. 

## Create Image Views

`VkImageView` is a handle to view image. It describes how images are treated. i.e. the meta data of images. In this case, we create one image view for each images in the swap chain. 
## Create Render Pass

A render pass consists of attachments, attachment references, subpass and subpass dependency.  Number of samples, how many attachments to draw and MSAA stuff. Similar to create rendering targets for framebuffers in OpenGL. 

It also provide subpasses for better memory management[?].

## Create Graphics Pipeline

1. Create shaders from *.spv.
2. Specify vertex inputs. Empty for now. Probably gonna need more for mesh loading. Also specify how the vertices are assembled. 
3. Specify view port, min and max depth and scissor extent. 
4. Create rasterizer, which is in charge of culling modes, dpeth clamp, front face culling, etc. 
5. Set MSAA details.
6. Setup color blending. 
7. Pipeline layouts, the uniforms that can be updated during the draw time.
8. Use all of the informations to create a rendering pipeline. 
## Create Framebuffers
Create a framebuffer for each image view in swap chain. When draw to a image on the swap chain, switch to that framebuffer. 
## Create Command pool
The execution of commands are put in the command buffer. Command pool is used to allocate command buffers.
```
typedef struct VkCommandPoolCreateInfo {
    VkStructureType             sType;
    const void*                 pNext;
    VkCommandPoolCreateFlags    flags;
    uint32_t                    queueFamilyIndex;
} VkCommandPoolCreateInfo;
```

Here queueFmailyIndex indicates that the command buffers in this command pool can only submit to one kind of queue on device. 
## Create Command buffers
*Create a framebuffer for each image view in swap chain*
Use `vkAllocateCommandBuffers` to allocate number of framebuffers. Then for each command buffer, begin command buffer with `vkBeginCommandBuffer` and of course the `VkCommandBufferBeginInfo`. Then start render pass with `VkRenderPassBeginInfo` and `vkCmdBeginRenderPass`. 
### Record commands to command buffers
```
vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
vkCmdDraw(commandBuffers[i], 3, 1, 0, 0);
vkCmdEndRenderPass(commandBuffers[i]);
```
Draw calls are recorded in the render buffer and executed later. 


## Create Semaphores
There are two mutexes for image is ready from swap chain and image is ready to present to swap chain.  

## Render image

1. Get image to draw from swap chain. 
2. Submit command to execute
3. Get present image in the present queue. 
	
