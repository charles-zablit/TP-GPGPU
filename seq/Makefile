CC=g++
CFLAGS=-I.

SRC_DIR := src
OBJ_DIR := obj
BIN_DIR := ../bin

EXE := $(BIN_DIR)/ann_seq
SRC := $(wildcard $(SRC_DIR)/*.c)

OBJ := $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC))

CPPFLAGS := -Iinclude -MMD -MP
CFLAGS   := -Wall
LDFLAGS  := -Llib
LDLIBS   := -lm
EXTRA_FLAGS :=

.PHONY: all profile clean

all: $(EXE)

$(EXE): $(OBJ) | $(BIN_DIR)
	$(CC) $(EXTRA_FLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

# With profiling
profile:
	$(MAKE) all EXTRA_FLAGS=-pg

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BIN_DIR) $(OBJ_DIR):
	mkdir -p $@

clean:
	@$(RM) -rv $(BIN_DIR) $(OBJ_DIR)

-include $(OBJ:.o=.d)