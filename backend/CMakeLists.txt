cmake_minimum_required(VERSION 3.10)
project(backend)

find_package(Boost 1.45.0 COMPONENTS range) 

message(STATUS "Found BOOST ${BOOST_VERSION}")
message(STATUS "BOOST_INCLUDE_DIRS : ${Boost_INCLUDE_DIRS}")
message(STATUS "BOOST_LIBRARIES : ${Boost_LIBRARIES}")
include_directories(${Boost_INCLUDE_DIRS}) 

find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
add_definitions(${LLVM_DEFINITIONS})
include_directories( ${LLVM_INCLUDE_DIRS} )
link_directories( ${LLVM_LIBRARY_DIRS} )
include_directories(3rdparty/json/single_include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++14 -fno-rtti")


include_directories(
  src/
)

set(source
  src/driver.cpp
  src/bir/Module.cpp
  src/bir/Function.cpp
  src/bir/BasicBlock.cpp
  src/bir/Instruction.cpp
  src/bir/MemoryLocation.cpp
  src/pass/pass.cpp
  src/pass/ChannelEnhance.cpp
  src/pass/LowerComplex.cpp
  src/pass/LowerConv2d.cpp
  src/pass/Allocator.cpp
  src/pass/Scheduler.cpp
  src/pass/Codegen.cpp
  src/pass/MCBase.cpp
  src/pass/LegacyMC.cpp
  src/pass/DRAMLayoutGen.cpp
  src/EngineType.cpp
)

add_executable(backend ${source})

# Find the libraries that correspond to the LLVM components
# that we wish to use
llvm_map_components_to_libnames(llvm_libs option support)

# Link against LLVM libraries
target_link_libraries(backend ${llvm_libs})
target_link_libraries(backend ${Boost_LIBRARIES})

