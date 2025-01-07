# Compiler
COMPILER = /usr/local/cuda/bin/nvcc
COMPILER_FLAGS = -O3 -std=c++11 -Xcompiler -Wall -Xcompiler -Werror -Werror cross-execution-space-call -I$(INCLUDE_DIR)

# Directories
INCLUDE_DIR = include
SOURCE_DIR = src
BUILD_DIR = .
RESOURCE_DIR = res
TEST_DIR = test

SOURCES = $(wildcard $(SOURCE_DIR)/*.cu)
OBJECTS = $(patsubst $(SOURCE_DIR)/%.cu, $(BUILD_DIR)/%.o, $(SOURCES))
BINARIES = kp-cuda
DATA_FILES = $(wildcard $(RESOURCE_DIR)/img_*.data)
PNG_FILES = $(DATA_FILES:.data=.png)

$(BINARIES): $(OBJECTS)
	$(COMPILER) $(COMPILER_FLAGS) -o $@ $^

$(BUILD_DIR)/%.o: $(SOURCE_DIR)/%.cu | $(BUILD_DIR)
	$(COMPILER) $(COMPILER_FLAGS) -c $< -o $@

$(BUILD_DIR):

run: $(BINARIES)
	./$(BINARIES)

run_test:
	./$(BINARIES) < $(TEST_DIR)/test.txt

run_test_cpu:
	./$(BINARIES) --cpu < $(TEST_DIR)/test.txt

convert_floor:
	python ./scripts/conv.py ./res/floor.jpg ./res/floor.data

images_create: $(PNG_FILES)

$(RESOURCE_DIR)/%.png: $(RESOURCE_DIR)/%.data
	python ./scripts/conv.py $< $@

memcheck:
	/usr/local/cuda/bin/compute-sanitizer --tool memcheck $(BINARIES)

benchmark_cpu:
	./$(BINARIES) --cpu < $(TEST_DIR)/test1.txt
	./$(BINARIES) --cpu < $(TEST_DIR)/test2.txt
	./$(BINARIES) --cpu < $(TEST_DIR)/test3.txt
	./$(BINARIES) --cpu < $(TEST_DIR)/test4.txt

benchmark_gpu:
	./$(BINARIES) < $(TEST_DIR)/test1.txt
	./$(BINARIES) < $(TEST_DIR)/test2.txt
	./$(BINARIES) < $(TEST_DIR)/test3.txt
	./$(BINARIES) < $(TEST_DIR)/test4.txt

all: $(BINARIES) convert_floor
	rm -rf ./*.o
clean_out:
	rm -rf $(RESOURCE_DIR)/*
clean:
	rm -rf $(BINARIES) $(RESOURCE_DIR)/floor.data $(PNG_FILES) $(DATA_FILES)