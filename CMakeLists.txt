cmake_minimum_required(VERSION 3.18)

project("GigaLearnBot" LANGUAGES CXX)

option(GGL_BUILD_RLBOT_EXAMPLE "Build the RLBot bot example executable (GigaLearnRLBot)" OFF)

file(GLOB_RECURSE FILES_SRC "src/*.cpp" "src/*.h" "src/*.hpp")

# The RLBot example has its own main() and builds as a separate executable
list(FILTER FILES_SRC EXCLUDE REGEX "rlbot_example/")

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

# RLBot deployment example (see src/rlbot_example/RLBotExampleMain.cpp)
if (GGL_BUILD_RLBOT_EXAMPLE)
	add_executable(GigaLearnRLBot
		"src/rlbot_example/RLBotExampleMain.cpp"
		"src/RLBotClient.cpp"
		"src/RLBotClient.h"
	)
	set_target_properties(GigaLearnRLBot PROPERTIES CXX_STANDARD 20)
	set_target_properties(GigaLearnRLBot PROPERTIES CXX_STANDARD_REQUIRED ON)
	target_include_directories(GigaLearnRLBot PRIVATE "src")
	target_link_libraries(GigaLearnRLBot GigaLearnCPP RLBotCPP)
endif()
