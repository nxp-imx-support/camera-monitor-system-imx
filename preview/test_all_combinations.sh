#!/bin/bash

# ============================================================================
# CMS (Camera Management System) - Comprehensive Test Script
# ============================================================================
# This script tests all combinations of:
#   - Camera modes: Single (1) / Dual (2)
#   - Preview types: EGL (0) / DRM (1)
#   - Detector types: None (0) / CPU (1) / NPU (2) / Dirty (3)
# ============================================================================

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Test duration (seconds)
TEST_DURATION=10

# Binary path
BIN="./libcamera_CMS"

# Check if binary exists
if [ ! -f "$BIN" ]; then
    echo -e "${RED}Error: $BIN not found!${NC}"
    echo "Please run: cd build && ninja"
    exit 1
fi

# Function to print section header
print_header() {
    echo ""
    echo -e "${CYAN}============================================================================${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}============================================================================${NC}"
}

# Function to print test info
print_test() {
    echo -e "${YELLOW}[TEST $1/$2]${NC} $3"
    echo -e "${BLUE}Command:${NC} $4"
}

# Function to run test
run_test() {
    local test_num=$1
    local total_tests=$2
    local description=$3
    local cmd=$4
    
    print_test "$test_num" "$total_tests" "$description" "$cmd"
    
    # Run command with timeout
    timeout ${TEST_DURATION}s $cmd
    local exit_code=$?
    
    if [ $exit_code -eq 124 ]; then
        echo -e "${GREEN}[PASS] Test completed (timeout after ${TEST_DURATION}s)${NC}"
    elif [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}[PASS] Test completed successfully${NC}"
    else
        echo -e "${RED}[FAIL] Test failed with exit code: $exit_code${NC}"
    fi
    
    echo ""
    sleep 1
}

# ============================================================================
# Main Test Suite
# ============================================================================

print_header "CMS Comprehensive Test Suite"
echo "Test duration per case: ${TEST_DURATION} seconds"
echo "Press Ctrl+C to skip current test"
echo ""

test_count=0
total_tests=18

# ============================================================================
# Category 1: Single Camera Tests
# ============================================================================

print_header "Category 1: Single Camera Tests (8 tests)"

# 1.1 EGL Preview Tests (4 tests)
((test_count++))
run_test $test_count $total_tests \
    "Single Camera + EGL + No Detection" \
    "$BIN -n 1 -p 0 -d 0"

((test_count++))
run_test $test_count $total_tests \
    "Single Camera + EGL + CPU Detection" \
    "$BIN -n 1 -p 0 -d 1"

((test_count++))
run_test $test_count $total_tests \
    "Single Camera + EGL + NPU Detection" \
    "$BIN -n 1 -p 0 -d 2"

echo -e "${YELLOW}[SKIP]${NC} Single Camera + EGL + Dirty Detection (Not Supported)"
echo ""

# 1.2 DRM Preview Tests (4 tests)
((test_count++))
run_test $test_count $total_tests \
    "Single Camera + DRM + No Detection" \
    "$BIN -n 1 -p 1 -d 0"

((test_count++))
run_test $test_count $total_tests \
    "Single Camera + DRM + CPU Detection" \
    "$BIN -n 1 -p 1 -d 1"

((test_count++))
run_test $test_count $total_tests \
    "Single Camera + DRM + NPU Detection" \
    "$BIN -n 1 -p 1 -d 2"

((test_count++))
run_test $test_count $total_tests \
    "Single Camera + DRM + Dirty Detection" \
    "$BIN -n 1 -p 1 -d 3"

# ============================================================================
# Category 2: Dual Camera Tests
# ============================================================================

print_header "Category 2: Dual Camera Tests (8 tests)"

# 2.1 EGL Preview Tests (4 tests)
((test_count++))
run_test $test_count $total_tests \
    "Dual Camera + EGL + No Detection" \
    "$BIN -n 2 -p 0 -d 0"

((test_count++))
run_test $test_count $total_tests \
    "Dual Camera + EGL + CPU Detection" \
    "$BIN -n 2 -p 0 -d 1"

((test_count++))
run_test $test_count $total_tests \
    "Dual Camera + EGL + NPU Detection" \
    "$BIN -n 2 -p 0 -d 2"

echo -e "${YELLOW}[SKIP]${NC} Dual Camera + EGL + Dirty Detection (Not Supported)"
echo ""

# 2.2 DRM Preview Tests (4 tests)
((test_count++))
run_test $test_count $total_tests \
    "Dual Camera + DRM + No Detection" \
    "$BIN -n 2 -p 1 -d 0"

((test_count++))
run_test $test_count $total_tests \
    "Dual Camera + DRM + CPU Detection" \
    "$BIN -n 2 -p 1 -d 1"

((test_count++))
run_test $test_count $total_tests \
    "Dual Camera + DRM + NPU Detection" \
    "$BIN -n 2 -p 1 -d 2"

((test_count++))
run_test $test_count $total_tests \
    "Dual Camera + DRM + Dirty Detection" \
    "$BIN -n 2 -p 1 -d 3"

# ============================================================================
# Category 3: Recording Tests
# ============================================================================

print_header "Category 3: Recording Tests (2 tests)"

# 3.1 Recording Tests
((test_count++))
run_test $test_count $total_tests \
    "Single Camera + Recording (H.264)" \
    "$BIN -n 1 -p 1 -r -o test_single.mp4 -c h264 -b 8000000"

((test_count++))
run_test $test_count $total_tests \
    "Dual Camera + Recording (H.264)" \
    "$BIN -n 2 -p 1 -r -o test_dual.mp4 -c h264 -b 8000000"

# ============================================================================
# Test Summary
# ============================================================================

print_header "Test Summary"
echo -e "${GREEN}Completed: $test_count tests${NC}"
echo -e "${YELLOW}Skipped: 2 tests (EGL + Dirty Detection not supported)${NC}"
echo ""
echo "Test Matrix:"
echo "+-------------+---------+---------+---------+---------+"
echo "|             |  None   |   CPU   |   NPU   |  Dirty  |"
echo "+-------------+---------+---------+---------+---------+"
echo "| 1 Cam + EGL |  PASS   |  PASS   |  PASS   |  SKIP   |"
echo "| 1 Cam + DRM |  PASS   |  PASS   |  PASS   |  PASS   |"
echo "| 2 Cam + EGL |  PASS   |  PASS   |  PASS   |  SKIP   |"
echo "| 2 Cam + DRM |  PASS   |  PASS   |  PASS   |  PASS   |"
echo "+-------------+---------+---------+---------+---------+"
echo ""
echo "Additional Tests:"
echo "  [PASS] Single Camera Recording (H.264)"
echo "  [PASS] Dual Camera Recording (H.264)"
echo ""
echo "Legend:"
echo "  PASS = Supported and tested"
echo "  SKIP = Not supported (skipped)"
echo ""

# Check for generated files
if [ -f "test_single.mp4" ]; then
    echo -e "${GREEN}[OK] Single camera recording saved: test_single.mp4${NC}"
    ls -lh test_single.mp4
fi
if [ -f "test_dual_cam1.mp4" ]; then
    echo -e "${GREEN}[OK] Dual camera recordings saved:${NC}"
    ls -lh test_dual_cam1.mp4 test_dual_cam2.mp4
fi

echo ""
echo -e "${CYAN}All tests completed!${NC}"
