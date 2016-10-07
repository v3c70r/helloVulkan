CXX=g++
FLAGS=-std=c++11 -Wall -pthread -g
SOURCE=helloVulkan.cpp imgui_impl_glfw_vulkan.cpp imgui/*.cpp

# Set your own SDK file path
SDK_PATH=/home/tsing/tools/VulkanSDK/1.0.24.0

INCLUDES=-I$(SDK_PATH)/x86_64/include -I imgui
LIBS=$(shell pkg-config --static --libs glfw3) -L$(SDK_PATH)/x86_64/lib  -lvulkan

run.sh: helloVulkan
	@echo "LD_LIBRARY_PATH=$(SDK_PATH)/x86_64/lib VK_LAYER_PATH=$(SDK_PATH)/x86_64/etc/explicit_layer.d ./$<" > $@
helloVulkan: $(SOURCE)
	$(CXX) $(FLAGS) $(INCLUDES) -o $@ $(SOURCE)  $(LIBS)
