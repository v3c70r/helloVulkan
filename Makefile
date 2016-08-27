CXX=clang++
FLAGS=-Wall -std=c++11 -pthread -g
SOURCE=helloVulkan.cpp
INCLUDES=-I/home/tsing/Downloads/VulkanSDK/1.0.21.1/x86_64/include
LIBS=$(shell pkg-config --static --libs glfw3) -L/home/tsing/Downloads/VulkanSDK/1.0.21.1/x86_64/lib -lvulkan
helloVulkan: helloVulkan.cpp
	$(CXX) $(FLAGS) $(INCLUDES) -o $@ $<  $(LIBS)
