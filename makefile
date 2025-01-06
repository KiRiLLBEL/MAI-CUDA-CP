# Compiler
COMPILER = /usr/local/cuda/bin/nvcc
COMPILER_FLAGS = -G -g -std=c++11 -Werror cross-execution-space-call,all-warnings,deprecated-declarations -I$(INCLUDE_DIR)

# Directories
INCLUDE_DIR = include
SOURCE_DIR = src
BUILD_DIR = build
OUT_DIR = out
RESOURCE_DIR = res
TEST_DIR = test

SOURCES = $(wildcard $(SOURCE_DIR)/*.cu)
OBJECTS = $(patsubst $(SOURCE_DIR)/%.cu, $(BUILD_DIR)/%.o, $(SOURCES))
BINARIES = course_project
DATA_FILES = $(wildcard $(OUT_DIR)/img_*.data)
PNG_FILES = $(DATA_FILES:.data=.png)

$(BINARIES): $(OBJECTS)
	$(COMPILER) $(COMPILER_FLAGS) -o $@ $^

$(BUILD_DIR)/%.o: $(SOURCE_DIR)/%.cu | $(BUILD_DIR)
	$(COMPILER) $(COMPILER_FLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

run: $(BINARIES)
	./$(BINARIES)

run_test:
	./$(BINARIES) < $(TEST_DIR)/test.txt

run_test_cpu:
	./$(BINARIES) --cpu < $(TEST_DIR)/test.txt

convert_floor:
	python ./scripts/conv.py ./res/floor.jpg ./res/floor.data

images_create: $(PNG_FILES)

$(OUT_DIR)/%.png: $(OUT_DIR)/%.data
	python ./scripts/conv.py $< $@

memcheck:
	/usr/local/cuda/bin/compute-sanitizer --tool memcheck $(BINARIES)

all: $(BINARIES) convert_floor
	mkdir $(OUT_DIR)
clean_out:
	rm -rf $(OUT_DIR)/*
clean:
	rm -rf $(BUILD_DIR) $(BINARIES) $(OUT_DIR) $(RESOURCE_DIR)/floor.data $(PNG_FILES) $(DATA_FILES)