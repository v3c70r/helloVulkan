	
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

// TODO: querySwapChainSupport()














	
