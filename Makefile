.PHONY: all configure build test run clean

BUILD_DIR ?= build
CMAKE ?= cmake
JOBS ?= $(shell nproc 2>/dev/null || echo 4)

all: build

configure:
	@$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release

build: configure
	@$(CMAKE) --build $(BUILD_DIR) -j $(JOBS)

test: build
	@ctest --test-dir $(BUILD_DIR) --output-on-failure

run: build
	@./$(BUILD_DIR)/incin_example

clean:
	@$(CMAKE) -E rm -rf $(BUILD_DIR)
