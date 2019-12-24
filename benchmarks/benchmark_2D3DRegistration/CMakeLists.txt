# CmakeLists.txt

# backward compatibility
cmake_minimum_required(VERSION 1.6)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR})

PROJECT(Benchmark_2D3DRegistration)
find_package( RegTools REQUIRED )

# all source files
SET(FILES main.cpp)

# Add include directory
INCLUDE_DIRECTORIES(			${REG_TOOLS_INCLUDE_DIR}
					)

# Add link directory
LINK_DIRECTORIES(${REG_TOOLS_LIBRARY_DIR})

# set the library name
ADD_EXECUTABLE( Benchmark_2D3DRegistration ${FILES} )

# Link libraries
target_link_libraries( Benchmark_2D3DRegistration RegToolsLIB )
