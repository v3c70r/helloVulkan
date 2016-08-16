	
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
<img src='http://g.gravizo.com/g?digraph G{main -> parse -> execute;main -> init;main -> cleanup;execute -> make_string;execute -> printf;init -> make_string;main -> printf;execute -> compare;}'/>


