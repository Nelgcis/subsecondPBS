# Makefile for RTD Integration Project

# Compiler settings
NVCC = nvcc
CXX = g++
# Allow overriding CUDA_HOME; default to /usr/local/cuda
CUDA_HOME ?= /usr/local/cuda
CUDA_INC = $(CUDA_HOME)/include
CUDA_LIB = $(CUDA_HOME)/lib64
INCLUDE_DIRS = -I./include -I./include/algorithms -I./include/core -I./include/utils -I$(CUDA_INC)
NVCC_FLAGS = -std=c++14 -O2 $(INCLUDE_DIRS) --compiler-options -fPIC -arch=sm_86
CXX_FLAGS = -std=c++14 -O2 $(INCLUDE_DIRS) -fPIC
LDFLAGS = -L$(CUDA_LIB) -L$(CUDA_HOME)/targets/x86_64-linux/lib -lcudart
# Ensure runtime loader can find shared libs next to executables (rpath using $ORIGIN)
RPATH_FLAGS = -Wl,-rpath,'$$ORIGIN'

# Directories
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build
BIN_DIR = bin

# Source files (recursive)
CUDA_SOURCES := $(shell find $(SRC_DIR) -name '*.cu' -print)
CPP_SOURCES := $(shell find $(SRC_DIR) -name '*.cpp' -print)
# Test CUDA sources (explicit from tests/)
TEST_CUDA_SOURCES := $(wildcard $(SRC_DIR)/tests/*.cu)
# Library CUDA sources = all CUDA sources minus explicit test/debug sources
LIBRARY_CUDA_SOURCES := $(filter-out $(TEST_CUDA_SOURCES) $(SRC_DIR)/debug_main.cu $(SRC_DIR)/simple_debug.cu $(SRC_DIR)/complete_dose_debug.cu $(SRC_DIR)/dose_debug_detailed.cu $(SRC_DIR)/simple_dose_test.cu $(SRC_DIR)/table_loader.cu, $(CUDA_SOURCES))
# Exclude test/debug C++ sources from library objects to avoid multiple main() definitions
LIBRARY_CPP_SOURCES := $(filter-out $(SRC_DIR)/test_%.cpp $(SRC_DIR)/debug_%.cpp $(SRC_DIR)/test_library.cpp $(SRC_DIR)/test_simple.cpp, $(CPP_SOURCES))
# Object lists (preserve subdir relative paths)
OBJECTS := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(LIBRARY_CUDA_SOURCES))
CPP_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(LIBRARY_CPP_SOURCES))
# Test targets (one binary per tests/*.cu)
TEST_TARGETS := $(patsubst $(SRC_DIR)/tests/%.cu,$(BIN_DIR)/%,$(TEST_CUDA_SOURCES))

# Target shared library
TARGET = $(BIN_DIR)/libraytracedicom.so

# Debug executable
DEBUG_TARGET = $(BIN_DIR)/debug_raytracedicom
SIMPLE_DEBUG_TARGET = $(BIN_DIR)/simple_debug
COMPLETE_DEBUG_TARGET = $(BIN_DIR)/complete_dose_debug
DETAILED_DEBUG_TARGET = $(BIN_DIR)/dose_debug_detailed
SIMPLE_DOSE_TARGET = $(BIN_DIR)/simple_dose_test
TABLE_LOADER_TARGET = $(BIN_DIR)/table_loader

# Default target
EXTRA_TARGETS :=
# Legacy explicit debug targets (if present)
ifneq ($(wildcard $(SRC_DIR)/debug_main.cu),)
EXTRA_TARGETS += $(DEBUG_TARGET)
endif
ifneq ($(wildcard $(SRC_DIR)/simple_debug.cu),)
EXTRA_TARGETS += $(SIMPLE_DEBUG_TARGET)
endif
ifneq ($(wildcard $(SRC_DIR)/complete_dose_debug.cu),)
EXTRA_TARGETS += $(COMPLETE_DEBUG_TARGET)
endif
# Add detailed debug target if source exists in either src/ or src/tests/
DETAILED_SRC := $(firstword $(wildcard $(SRC_DIR)/dose_debug_detailed.cu $(SRC_DIR)/tests/dose_debug_detailed.cu))
ifneq ($(DETAILED_SRC),)
EXTRA_TARGETS += $(DETAILED_DEBUG_TARGET)
endif

ifneq ($(wildcard $(SRC_DIR)/simple_dose_test.cu),)
EXTRA_TARGETS += $(SIMPLE_DOSE_TARGET)
endif
ifneq ($(wildcard $(SRC_DIR)/table_loader.cu),)
EXTRA_TARGETS += $(TABLE_LOADER_TARGET)
endif

# Add any tests/ CUDA files as targets
EXTRA_TARGETS += $(TEST_TARGETS)

all: $(TARGET) $(EXTRA_TARGETS)

# Create directories
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Compile CUDA source files
# Compile CUDA source files (preserve subdir structure)
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile C++ source files (preserve subdir structure)
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

# Build test binaries (one per src/tests/*.cu). Link with library if available
$(BIN_DIR)/%: $(SRC_DIR)/tests/%.cu $(TARGET) | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $< -o $@ -L$(BIN_DIR) -lraytracedicom $(RPATH_FLAGS) $(LDFLAGS)

# Link shared library
ifneq ($(OBJECTS)$(CPP_OBJECTS),)
$(TARGET): $(OBJECTS) $(CPP_OBJECTS) | $(BIN_DIR)
	$(NVCC) -shared $(OBJECTS) $(CPP_OBJECTS) -o $@ $(LDFLAGS)
else
$(TARGET):
	@echo "No library sources found; skipping $(TARGET)"
endif

# Compile debug executable
$(DEBUG_TARGET): $(SRC_DIR)/debug_main.cu $(TARGET) | $(BIN_DIR)
	$(NVCC) -std=c++14 -O2 -I./include -arch=sm_86 $(SRC_DIR)/debug_main.cu -o $@ -L$(BIN_DIR) -lraytracedicom $(RPATH_FLAGS) $(LDFLAGS)

# Compile simple debug executable
$(SIMPLE_DEBUG_TARGET): $(SRC_DIR)/simple_debug.cu | $(BIN_DIR)
	$(NVCC) -std=c++14 -O2 -I./include $(SRC_DIR)/simple_debug.cu -o $@ $(LDFLAGS)

# Compile complete dose debug executable
$(COMPLETE_DEBUG_TARGET): $(SRC_DIR)/complete_dose_debug.cu $(TARGET) | $(BIN_DIR)
	$(NVCC) -std=c++14 -O2 -I./include -arch=sm_86 $(SRC_DIR)/complete_dose_debug.cu -o $@ -L$(BIN_DIR) -lraytracedicom $(RPATH_FLAGS) $(LDFLAGS)

# Compile detailed dose debug executable (source may live in src/ or src/tests/)
$(DETAILED_DEBUG_TARGET): $(DETAILED_SRC) $(TARGET) | $(BIN_DIR)
	$(NVCC) -std=c++14 -O2 $(NVCC_FLAGS) $(DETAILED_SRC) -o $@ -L$(BIN_DIR) -lraytracedicom $(RPATH_FLAGS) $(LDFLAGS)

# Compile simple dose test executable
$(SIMPLE_DOSE_TARGET): $(SRC_DIR)/simple_dose_test.cu $(TARGET) | $(BIN_DIR)
	$(NVCC) -std=c++14 -O2 -I./include -arch=sm_86 $(SRC_DIR)/simple_dose_test.cu -o $@ -L$(BIN_DIR) -lraytracedicom $(RPATH_FLAGS) $(LDFLAGS)

# Clean build files
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

# Install dependencies (CentOS)
install-deps:
	sudo yum update -y
	sudo yum groupinstall -y "Development Tools"
	sudo yum install -y cuda-toolkit

# Test compilation
test: $(TARGET)
	@echo "Running RTD integration test..."
	./$(TARGET)

# Run debug program
debug: $(DEBUG_TARGET)
	@echo "Running debug program..."
	LD_LIBRARY_PATH=$(BIN_DIR):/usr/local/cuda/targets/x86_64-linux/lib ./$(DEBUG_TARGET)

# Run simple debug program
simple-debug: $(SIMPLE_DEBUG_TARGET)
	@echo "Running simple debug program..."
	./$(SIMPLE_DEBUG_TARGET)

# Run complete dose debug program
complete-debug: $(COMPLETE_DEBUG_TARGET)
	@echo "Running complete dose debug program..."
	LD_LIBRARY_PATH=$(BIN_DIR):/usr/local/cuda/targets/x86_64-linux/lib ./$(COMPLETE_DEBUG_TARGET)

# Run detailed dose debug program
detailed-debug: $(DETAILED_DEBUG_TARGET)
	@echo "Running detailed dose debug program..."
	LD_LIBRARY_PATH=$(BIN_DIR):/usr/local/cuda/targets/x86_64-linux/lib ./$(DETAILED_DEBUG_TARGET)

# Run simple dose test program
simple-dose-test: $(SIMPLE_DOSE_TARGET)
	@echo "Running simple dose test program..."
	LD_LIBRARY_PATH=$(BIN_DIR):/usr/local/cuda/targets/x86_64-linux/lib ./$(SIMPLE_DOSE_TARGET)

# Show help
help:
	@echo "Available targets:"
	@echo "  all        - Build the complete project (library + debug executable)"
	@echo "  clean      - Remove build files"
	@echo "  install-deps - Install dependencies (CentOS)"
	@echo "  test       - Build and run the test"
	@echo "  debug      - Build and run the debug program"
	@echo "  simple-debug - Build and run the simple debug program"
	@echo "  complete-debug - Build and run the complete dose calculation debug program"
	@echo "  detailed-debug - Build and run the detailed dose debugging program"
	@echo "  simple-dose-test - Build and run the simple dose calculation test"
	@echo "  help       - Show this help message"

.PHONY: all clean install-deps test debug simple-debug complete-debug detailed-debug simple-dose-test help
