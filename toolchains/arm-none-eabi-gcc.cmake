#
# Copyright (c) 2020-2021 Arm Limited. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

set(TARGET_CPU "cortex-m4" CACHE STRING "Target CPU")

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_C_COMPILER "arm-none-eabi-gcc")
set(CMAKE_ASM_COMPILER "arm-none-eabi-gcc")
set(CMAKE_CXX_COMPILER "arm-none-eabi-g++")

# Convert TARGET_CPU=Cortex-M33+nofp+nodsp into
#   - CMAKE_SYSTEM_PROCESSOR=cortex-m33
#   - TARGET_CPU_FEATURES=no-fp;no-dsp
string(REPLACE "+" ";" TARGET_CPU_FEATURES ${TARGET_CPU})
list(POP_FRONT TARGET_CPU_FEATURES CMAKE_SYSTEM_PROCESSOR)
string(TOLOWER ${CMAKE_SYSTEM_PROCESSOR} CMAKE_SYSTEM_PROCESSOR)

set(CMAKE_EXECUTABLE_SUFFIX ".elf")
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# Select C/C++ version
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)

# Compile options
add_compile_options(
    -mcpu=${TARGET_CPU}
    -mthumb
    "$<$<CONFIG:DEBUG>:-gdwarf-3>"
    "$<$<COMPILE_LANGUAGE:CXX>:-fno-unwind-tables;-fno-rtti;-fno-exceptions>")

# Compile defines
add_compile_definitions(
    "$<$<NOT:$<CONFIG:DEBUG>>:NDEBUG>")

# Link options
add_link_options(
    -mcpu=${TARGET_CPU}
    -mthumb
    --specs=nosys.specs)

# Set floating point unit
if("${TARGET_CPU}" MATCHES "\\+fp")
    set(FLOAT hard)
elseif("${TARGET_CPU}" MATCHES "\\+nofp")
    set(FLOAT soft)
elseif("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "cortex-m33" OR
       "${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "cortex-m55")
    set(FLOAT hard)
elseif("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "cortex-m4" OR
        "${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "cortex-m7")
    set(FLOAT hard)
    set(FPU_CONFIG "fpv4-sp-d16")
    add_compile_options(-mfpu=${FPU_CONFIG})
    add_link_options(-mfpu=${FPU_CONFIG})
else()
    set(FLOAT soft)
endif()

if (FLOAT)
    add_compile_options(-mfloat-abi=${FLOAT})
    add_link_options(-mfloat-abi=${FLOAT})
endif()

# Compilation warnings
add_compile_options(
    -Wall
    -Wextra

    -Wcast-align
    -Wdouble-promotion
    -Wformat
    -Wmissing-field-initializers
    -Wnull-dereference
    -Wredundant-decls
    -Wshadow
    -Wswitch
    -Wswitch-default
    -Wunused

    -Wno-redundant-decls
)
