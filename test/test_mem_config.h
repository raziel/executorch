#pragma once

// mimic a typical memory pool pre-allocation in an embedded system.
// It's defined by the user and included in the application when linking to
// lib executorch

// Number of pools used. Pool zero is defaulted to constant data in flatbuffer
#define NUM_MEMORY_POOLS 2

// ACTIVATION_POOL_SZ is set at compilation
#define ACTIVATION_POOL_SZ 0x00200000

// The associated macro, ACTIVATION_POOL_ATTRIBUTE is a linker attribute,
// e.g. __attribute__(section(.bss.sram)).
static uint8_t  activation_pool[ACTIVATION_POOL_SZ] /*ACTIVATION_POOL_ATTRIBUTE*/;

