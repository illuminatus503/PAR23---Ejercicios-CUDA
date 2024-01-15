## Project compiler options ##
# CUDA directory:
CUDA_ROOT_DIR = /usr/local/cuda
# CC compiler options:
CC = g++
CC_FLAGS = -Wall -g -O2 -I$(INC_DIR)
CC_LIBS = -lm
# NVCC compiler options:
NVCC = nvcc
# NVCC_FLAGS = -O2 -arch=sm_50 -gencode=arch=compute_50,code=sm_50 -Wno-deprecated-gpu-targets -I$(INC_DIR)
NVCC_FLAGS = -O2 -I$(INC_DIR)
NVCC_LIBS = -lm
# CUDA library directory:
CUDA_LIB_DIR = -L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR = -I$(CUDA_ROOT_DIR)/include
# CUDA linking libraries:
CUDA_LINK_LIBS = -lcudart

## Project file structure ##
# Source file directory:
SRC_DIR = src
# Object file directory:
OBJ_DIR = obj
# Include header file directory:
INC_DIR = include
# Library directory:
LIB_DIR = lib
# Test directory:
TEST_DIR = test
# Bin directory (for the final executable):
BIN_DIR = bin

## Make variables ##
# Target executable name:
BIN = $(BIN_DIR)/program
# C and CUDA source files:
SRC_C = $(wildcard $(SRC_DIR)/*.c)
SRC_CU = $(wildcard $(SRC_DIR)/*.cu)
# Object files:
OBJS = $(SRC_C:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o) $(SRC_CU:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)

## Compile ##
# Link C and CUDA compiled object files to target executable:
$(BIN): $(OBJS) | $(BIN_DIR)
	$(CC) $(CC_FLAGS) $(OBJS) -o $@ $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS) $(CC_LIBS)

# Compile C source files to object files:
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CC_FLAGS) -c $< -o $@ $(CC_LIBS)

# Compile CUDA source files to object files:
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

# Handle directories
$(OBJ_DIR):
	mkdir -p $@

$(BIN_DIR):
	mkdir -p $@

# Clean objects in object directory and executable in bin directory.
clean:
	$(RM) -rv $(BIN_DIR)/* $(OBJ_DIR)/*

.PHONY: clean
