# =============================================================================
#  Makefile — spmv_benchmark
#  CS-3006 Parallel & Distributed Computing · Spring 2026
#  Muhammad Umair (23i-0662) · Awais Basheer (23i-0506)
# =============================================================================

CXX      := g++
TARGET   := spmv_benchmark
SRC      := spmv_benchmark.cpp

# --------------------------------------------------------------------------
# Core flags
# --------------------------------------------------------------------------
CXXFLAGS := \
    -std=c++17       \
    -O3              \
    -march=native    \
    -mavx2           \
    -mfma            \
    -fopenmp         \
    -funroll-loops   \
    -fno-math-errno  \
    -Wall -Wextra    \
    -Wno-unused-result

LDFLAGS  := -fopenmp -lm

# --------------------------------------------------------------------------
# Debug build (add `make debug`)
# --------------------------------------------------------------------------
DBGFLAGS := -std=c++17 -O0 -g -fopenmp -mavx2 -mfma -march=native \
            -Wall -Wextra -fsanitize=address,undefined

# =============================================================================
.PHONY: all clean debug run help

all: $(TARGET)
	@echo ""
	@echo "  Build complete: ./$(TARGET)"
	@echo "  Usage examples:"
	@echo "    ./$(TARGET) --mtx=thermomech_dK.mtx"
	@echo "    ./$(TARGET) --mtx=ASIC_320k.mtx --threads=8 --iters=200"
	@echo ""

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

debug: $(SRC)
	$(CXX) $(DBGFLAGS) -o $(TARGET)_debug $< -lm
	@echo "  Debug build: ./$(TARGET)_debug"

# Quick smoke test on the smaller of the two downloaded matrices
run: $(TARGET)
	@if [ -f thermomech_dK.mtx ]; then \
	    ./$(TARGET) --mtx=thermomech_dK.mtx --iters=50; \
	elif [ -f ASIC_320k.mtx ]; then \
	    ./$(TARGET) --mtx=ASIC_320k.mtx --iters=50; \
	else \
	    echo "No .mtx file found – run ./fetch_data.sh first"; \
	fi

clean:
	rm -f $(TARGET) $(TARGET)_debug results.csv
	@echo "  Cleaned."

help:
	@echo "Targets:"
	@echo "  all      Build optimised binary (default)"
	@echo "  debug    Build with -O0 -g -fsanitize=address"
	@echo "  run      Build + quick smoke test"
	@echo "  clean    Remove binaries and results.csv"
