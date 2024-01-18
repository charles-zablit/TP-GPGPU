CC=nvcc
CFLAGS=-I.

SRC_DIR := src
OBJ_DIR := obj
BIN_DIR := ../bin

EXE := $(BIN_DIR)/ann_par
SRC := $(wildcard $(SRC_DIR)/*.cu)

OBJ := $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(SRC))

CPPFLAGS := -Iinclude
LDFLAGS  := -Llib
LDLIBS   := -lm -lcublas -lcudart
ARCH := -arch=sm_60
EXTRA_FLAGS :=

.PHONY: all profile clean

all: $(EXE)

$(EXE): $(OBJ) | $(BIN_DIR)
	$(CC) $(EXTRA_FLAGS) $(LDFLAGS) $(ARCH) $^ $(LDLIBS) -o $@

# With profiling
profile:
	$(MAKE) all EXTRA_FLAGS=-pg

# With debug
debug:
	$(MAKE) all EXTRA_FLAGS="-g -G -Xcompiler -rdynamic"

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(CC) $(EXTRA_FLAGS) $(CPPFLAGS) $(ARCH) $(CFLAGS) -c $< -o $@

$(BIN_DIR) $(OBJ_DIR):
	mkdir -p $@

clean:
	@$(RM) -rv $(EXE) $(OBJ_DIR)

-include $(OBJ:.o=.d)