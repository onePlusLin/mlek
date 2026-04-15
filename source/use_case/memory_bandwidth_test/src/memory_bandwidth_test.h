
#ifndef TEST_INPUT_DATA_H
#define TEST_INPUT_DATA_H

#include <stdint.h>
#include <stddef.h>

// Test data for Ethos-U NPU benchmarking
// Generated automatically by test pattern generator

// Array shape: (1, 224, 224, 3)
#define TEST_INPUT_DATA_BATCH_SIZE 1
#define TEST_INPUT_DATA_HEIGHT 224
#define TEST_INPUT_DATA_WIDTH 224
#define TEST_INPUT_DATA_CHANNELS 3
#define TEST_INPUT_DATA_SIZE 150528

// Test input data array
extern const int8_t test_input_data[TEST_INPUT_DATA_SIZE];

// Array dimensions
extern const size_t test_input_data_dims[4];

#endif // TEST_INPUT_DATA_H
