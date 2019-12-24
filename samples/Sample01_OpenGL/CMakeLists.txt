# CmakeLists.txt

# backward compatibility
cmake_minimum_required(VERSION 1.6)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR})

PROJECT(Sample01_OpenGL)
find_package(GLEW)
find_package(OpenGL)
find_package(GLUT)
find_package( RegTools REQUIRED )

# all source files
SET(FILES
	main.cpp
	../Sample_Common/zpr.c
	../Sample_Common/Vector3.h
	../Sample_Common/zpr.h
	../Sample_Common/FileMHD.h
	../Sample_Common/FileMHD.cxx
    )

if(CMAKE_SIZEOF_VOID_P EQUAL 4)
set(TARGET "32")
else(CMAKE_SIZEOF_VOID_P EQUAL 4)
set(TARGET "64")
endif(CMAKE_SIZEOF_VOID_P EQUAL 4)

### Copy binary libraries to the build directory
# if(WIN32)
# CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/../libs/Win${TARGET}/glew${TARGET}.dll ${CMAKE_CURRENT_BINARY_DIR}/glew${TARGET}.dll COPYONLY)
# CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/../libs/Win32/glut32.dll ${CMAKE_CURRENT_BINARY_DIR}/glut32.dll COPYONLY)
# endif(WIN32)

# Add include directory
INCLUDE_DIRECTORIES(${GLEW_INCLUDE_PATH} 
					${REG_TOOLS_INCLUDE_DIR}
					${GLUT_INCLUDE_DIR}
					../Sample_Common
					)

# Add link directory
LINK_DIRECTORIES(${REG_TOOLS_LIBRARY_DIR})

# set the library name
ADD_EXECUTABLE( Sample01_OpenGL ${FILES} )

# Link libraries
target_link_libraries(Sample01_OpenGL
  RegToolsLIB
  ${GLEW_LIBRARY}
  ${OPENGL_LIBRARIES}
  ${GLUT_LIBRARIES}
  )
