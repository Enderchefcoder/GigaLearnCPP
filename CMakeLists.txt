cmake_minimum_required(VERSION 3.18)

project("GigaLearnBot" LANGUAGES CXX)

file(GLOB_RECURSE FILES_SRC "src/*.cpp" "src/*.h" "src/*.hpp")
add_executable(GigaLearnBot ${FILES_SRC})

# Set C++ version to 20
set_target_properties(GigaLearnBot PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(GigaLearnBot PROPERTIES CXX_STANDARD 20)
set_target_properties(GigaLearnBot PROPERTIES CXX_STANDARD_REQUIRED ON)

# Make sure GigaLearnCPP is going to build in the same directory as us
# Otherwise, we won't be able to import it at runtime
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}")

# Include GigaLearnCPP
# (With -DGGL_BUILD_TESTS=ON, unit tests build too and can be run with ctest)
if (GGL_BUILD_TESTS)
	enable_testing()
endif()
add_subdirectory(GigaLearnCPP)
target_link_libraries(GigaLearnBot GigaLearnCPP)

# Include RLBot
add_subdirectory(RLBotCPP)
target_link_libraries(GigaLearnBot RLBotCPP)
