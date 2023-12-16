## Project compiler options ##
# CUDA directory:
CUDA_ROOT_DIR = /usr/local/cuda
# CC compiler options:
CC = g++
CC_FLAGS = -Wall -g -O2 
CC_LIBS = -lm
# NVCC compiler options:
NVCC = nvcc
NVCC_FLAGS = -O2 -arch=sm_50 # For GTX750Ti
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
OBJ_DIR = bin
# Include header file directory:
INC_DIR = include

## Make variables ##
# Target executable name:
BIN = program
# C and CUDA source files:
SRC_C = $(wildcard $(SRC_DIR)/*.c)
SRC_CU = $(wildcard $(SRC_DIR)/*.cu)
# Object files:
OBJS = $(SRC_C:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o) $(SRC_CU:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)

## Compile ##
# Link C and CUDA compiled object files to target executable:
$(BIN): $(OBJS)
	$(CC) $(CC_FLAGS) $(OBJS) -o $@ $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS) $(CC_LIBS)

# Compile C source files to object files:
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CC_FLAGS) -c $< -o $@ $(CC_LIBS)

# Compile CUDA source files to object files:
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

# Handle directories
$(OBJ_DIR):
	mkdir -p $@

# Clean objects in object directory.
clean:
	$(RM) -rv $(BIN) $(OBJ_DIR)/*
