## Project compiler options ##
# CUDA directory:
CUDA_ROOT_DIR = /usr/local/cuda
# CC compiler options:
CC = g++
CC_FLAGS = -Wall -g -O2 -I$(INC_DIR)
CC_LIBS = -lm
# NVCC compiler options:
NVCC = nvcc
NVCC_FLAGS = -O2 -I$(INC_DIR) -arch sm_75
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
CUDA_SRC_DIR = $(SRC_DIR)/cuda
# Object file directory:
OBJ_DIR = obj
CUDA_OBJ_DIR = $(OBJ_DIR)/cuda
# Include header file directory:
INC_DIR = include
# Additional include directories:
CUDA_INC_DIR = $(INC_DIR)/cuda
# Library directory:
LIB_DIR = lib
# Test directory:
TEST_DIR = test
TEST_BIN_DIR = $(TEST_DIR)/bin
# Bin directory (for the final executable):
BIN_DIR = bin

## Make variables ##
# Target executable name:
BIN = $(BIN_DIR)/program
# C and CUDA source files:
SRC_C = $(wildcard $(SRC_DIR)/*.c)
SRC_CU = $(wildcard $(CUDA_SRC_DIR)/*.cu)
# Object files:
OBJS = $(filter-out $(OBJ_DIR)/main.o, $(SRC_C:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o) $(SRC_CU:$(CUDA_SRC_DIR)/%.cu=$(CUDA_OBJ_DIR)/%.o))

## Test compilation and execution ##
# Test source files:
TEST_SRC = $(wildcard $(TEST_DIR)/*.c)
# Test object files:
TEST_OBJS = $(TEST_SRC:$(TEST_DIR)/%.c=$(TEST_DIR)/%.o)
# Test binaries:
TEST_BINS = $(TEST_SRC:$(TEST_DIR)/%.c=$(TEST_BIN_DIR)/%)

## Compile ##
# Compile all project files
all: $(BIN)

# Test rule:
test: $(TEST_BINS)
	@echo "";
	@for test_bin in $(TEST_BINS); do \
		echo Running $$test_bin; \
		$$test_bin; \
		echo ""; \
	done

# Rule to compile test binaries:
$(TEST_BIN_DIR)/%: $(TEST_DIR)/%.o $(OBJS) | $(TEST_BIN_DIR) 
	$(CC) $(CC_FLAGS) $^ -o $@ $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS) $(CC_LIBS)

# Link C and CUDA compiled object files to target executable:
$(BIN): $(OBJS) | $(BIN_DIR)
	$(CC) $(CC_FLAGS) $(OBJS) $(SRC_DIR)/main.c -o $@ $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS) $(CC_LIBS)

# Compile test source files to object files:
$(TEST_DIR)/%.o: $(TEST_DIR)/%.c | $(TEST_BIN_DIR) 
	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile C source files to object files:
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CC_FLAGS) -c $< -o $@ $(CC_LIBS)

# Compile CUDA source files to object files:
$(CUDA_OBJ_DIR)/%.o: $(CUDA_SRC_DIR)/%.cu | $(CUDA_OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

# Handle directories
$(OBJ_DIR):
	@mkdir -p $@

$(CUDA_OBJ_DIR):
	@mkdir -p $@

$(BIN_DIR):
	@mkdir -p $@

$(TEST_BIN_DIR):
	@mkdir -p $@

# Clean objects in object directory and executable in bin directory.
clean:
	$(RM) -rv $(BIN_DIR)/* $(OBJ_DIR)/* $(CUDA_OBJ_DIR)/* $(TEST_BIN_DIR)/*


.PHONY: all test clean $(BIN_DIR) $(OBJ_DIR) $(CUDA_OBJ_DIR) $(TEST_DIR)
