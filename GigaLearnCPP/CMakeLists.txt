cmake_minimum_required(VERSION 3.18)

project("GigaLearnCPP" LANGUAGES CXX)

option(GGL_BUILD_TESTS "Build GigaLearnCPP unit tests" OFF)

# Compile for the machine you're building on (faster simulation, non-portable binaries)
option(GGL_NATIVE_ARCH "Optimize for the host CPU architecture (-march=native / /arch:AVX2)" OFF)

include_directories("${PROJECT_SOURCE_DIR}/src/")

# Make sure CMake finds libtorch if it's in this directory
if (EXISTS "${PROJECT_SOURCE_DIR}/libtorch/")
	message(STATUS "GigaLearnCPP: Using local libtorch folder...")
	list(APPEND CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/libtorch")
endif()

# Add libtorch (https://pytorch.org/cppdocs/installing.html#minimal-example)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Add all headers and code files
file(GLOB_RECURSE FILES_SRC "src/*.cpp" "src/*.h" "src/*.hpp" "libsrc/*.cpp" "libsrc/*.h" "libsrc/*.hpp")

add_library(GigaLearnCPP SHARED ${FILES_SRC})
target_compile_definitions(GigaLearnCPP PRIVATE -DWITHIN_GGL)
target_include_directories(GigaLearnCPP PUBLIC "src/public")
target_include_directories(GigaLearnCPP PRIVATE "src/private")

# Include libtorch
target_link_libraries(GigaLearnCPP PRIVATE "${TORCH_LIBRARIES}")

if (TORCH_CUDA_LIBRARIES)
	message(STATUS "GigaLearnCPP: Enabling CUDA support...")
	target_compile_definitions(GigaLearnCPP PRIVATE -DRG_CUDA_SUPPORT)
endif()

# Set C++ version to 20
set_target_properties(GigaLearnCPP PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(GigaLearnCPP PROPERTIES CXX_STANDARD 20)
set_target_properties(GigaLearnCPP PROPERTIES CXX_STANDARD_REQUIRED ON)

# Include RLGymCPP
add_subdirectory(RLGymCPP)
target_link_libraries(GigaLearnCPP PUBLIC RLGymCPP)

if (GGL_NATIVE_ARCH)
	message(STATUS "GigaLearnCPP: Optimizing for the host CPU (binaries won't be portable)...")
	if (MSVC)
		set(GGL_ARCH_FLAGS "/arch:AVX2")
	else()
		set(GGL_ARCH_FLAGS "-march=native")
	endif()
	target_compile_options(GigaLearnCPP PRIVATE ${GGL_ARCH_FLAGS})
	target_compile_options(RLGymCPP PRIVATE ${GGL_ARCH_FLAGS})
	target_compile_options(RocketSim PRIVATE ${GGL_ARCH_FLAGS})
endif()

# Include JSON
target_include_directories(GigaLearnCPP PUBLIC "${PROJECT_SOURCE_DIR}/libsrc/json")

# Include Python
# NOTE: FindPythonLibs/FindPythonInterp are deprecated and often find mismatched
#	interpreter/library pairs, so the modern FindPython module is used instead
find_package(Python COMPONENTS Interpreter Development REQUIRED)
target_link_libraries(GigaLearnCPP PUBLIC Python::Python)
message(STATUS "GigaLearnCPP: Found Python:")
message(STATUS "  Python_EXECUTABLE: ${Python_EXECUTABLE}")
message(STATUS "  Python_LIBRARIES: ${Python_LIBRARIES}")
message(STATUS "  Python_INCLUDE_DIRS: ${Python_INCLUDE_DIRS}")
message(STATUS "  Python_RUNTIME_LIBRARY_DIRS: ${Python_RUNTIME_LIBRARY_DIRS}")
target_compile_definitions(GigaLearnCPP PRIVATE PY_EXEC_PATH="${Python_EXECUTABLE}") # Give C++ access to the executable path

# Include pybind11
add_subdirectory(pybind11)
target_link_libraries(GigaLearnCPP PUBLIC pybind11::embed)

# MSVC fails to find python DLLs even though they are in my path. Good job MSVC. Well done.
# This copies the python DLLs to the output directory
if (MSVC)
    file(GLOB PYTHON_DLLS "${Python_RUNTIME_LIBRARY_DIRS}/*.dll")
	message(STATUS "GigaLearnCPP: Adding Python DLLs: ${PYTHON_DLLS}")
    add_custom_command(TARGET GigaLearnCPP
                 POST_BUILD
                 COMMAND ${CMAKE_COMMAND} -E copy_if_different
                 ${PYTHON_DLLS}
                 $<TARGET_FILE_DIR:GigaLearnCPP>)
endif (MSVC)

# Make our python files copy over to our build dir
# (The learner imports them at runtime relative to the executable)
configure_file("./python_scripts/metric_receiver.py" "${CMAKE_BINARY_DIR}/python_scripts/metric_receiver.py" COPYONLY)
configure_file("./python_scripts/render_receiver.py" "${CMAKE_BINARY_DIR}/python_scripts/render_receiver.py" COPYONLY)

# MSVC sometimes won't link to the libtorch DLLs unless you do this
# This is also from https://pytorch.org/cppdocs/installing.html#minimal-example
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
	message(STATUS "GigaLearnCPP: Adding TORCH_DLLS: ${TORCH_DLLS}")
    add_custom_command(TARGET GigaLearnCPP
                 POST_BUILD
                 COMMAND ${CMAKE_COMMAND} -E copy_if_different
                 ${TORCH_DLLS}
                 $<TARGET_FILE_DIR:GigaLearnCPP>)
endif (MSVC)

# Unit tests
if (GGL_BUILD_TESTS)
	enable_testing()
	add_subdirectory(tests)
endif()
