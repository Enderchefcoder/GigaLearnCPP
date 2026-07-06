cmake_minimum_required(VERSION 3.18)

# GigaLearnCPP unit tests
# Enable with -DGGL_BUILD_TESTS=ON on the GigaLearnCPP project

project("GigaLearnTests" LANGUAGES CXX)

# CONFIGURE_DEPENDS makes the build re-glob when test files are added/removed
file(GLOB TEST_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

# The tests compile the (private) source files they test directly,
#	so they don't depend on which symbols the shared library exports
set(GGL_TESTED_SRC
	"${CMAKE_CURRENT_SOURCE_DIR}/../src/private/GigaLearnCPP/PPO/GAE.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/../src/private/GigaLearnCPP/PPO/ExperienceBuffer.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/../src/private/GigaLearnCPP/PPO/PPOLearner.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/../src/private/GigaLearnCPP/SAC/ReplayBuffer.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/../src/private/GigaLearnCPP/SAC/SACLearner.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/../src/private/GigaLearnCPP/Util/Models.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/../src/private/GigaLearnCPP/Util/MagSGD.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/../src/private/GigaLearnCPP/Util/PolicyInference.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/../src/public/GigaLearnCPP/Util/Utils.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/../src/public/GigaLearnCPP/Util/Report.cpp"
)

add_executable(GigaLearnTests ${TEST_SRC} ${GGL_TESTED_SRC})

set_target_properties(GigaLearnTests PROPERTIES CXX_STANDARD 20)
set_target_properties(GigaLearnTests PROPERTIES CXX_STANDARD_REQUIRED ON)

target_compile_definitions(GigaLearnTests PRIVATE -DWITHIN_GGL)

target_include_directories(GigaLearnTests PRIVATE
	"${CMAKE_CURRENT_SOURCE_DIR}/../src"
	"${CMAKE_CURRENT_SOURCE_DIR}/../src/public"
	"${CMAKE_CURRENT_SOURCE_DIR}/../src/private"
	"${CMAKE_CURRENT_SOURCE_DIR}/../libsrc/json"
)

target_link_libraries(GigaLearnTests PRIVATE RLGymCPP "${TORCH_LIBRARIES}")

# End-to-end integration test (trains a tiny model for real, on CPU)
add_executable(GigaLearnIntegrationTest "${CMAKE_CURRENT_SOURCE_DIR}/integration/IntegrationTest.cpp")
set_target_properties(GigaLearnIntegrationTest PROPERTIES CXX_STANDARD 20)
set_target_properties(GigaLearnIntegrationTest PROPERTIES CXX_STANDARD_REQUIRED ON)
target_link_libraries(GigaLearnIntegrationTest PRIVATE GigaLearnCPP)

enable_testing()
add_test(NAME GigaLearnTests COMMAND GigaLearnTests)
add_test(NAME GigaLearnIntegrationTest COMMAND GigaLearnIntegrationTest)
