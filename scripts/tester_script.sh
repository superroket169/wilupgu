#!/usr/bin/env bash
# Runs wilupgu's tests and the parity example on this machine's GPUs.
# A step is SKIPPED when it found no device for a backend; that is not a pass.
cd "$(dirname "$0")/.." || exit 1

export PATH="$PATH:$HOME/.cargo/bin:/usr/local/cuda/bin"

LOGFILE=test_results.log
STEPLOG=$(mktemp)
trap 'rm -f "$STEPLOG"' EXIT

{
    echo "[$(date)] wilupgu test run"
    echo "--- rustc: $(rustc --version 2>&1)"
    echo "--- nvidia-smi -L:"
    nvidia-smi -L 2>&1
    echo "--- nvcc --version:"
    nvcc --version 2>&1
} > "$LOGFILE"

NAMES=()
RESULTS=()
ANY_FAILED=0
ANY_SKIPPED=0

step() {
    local name=$1
    shift
    echo "=== $name: $* ==="
    echo "=== $name: $* ===" >> "$LOGFILE"
    "$@" > "$STEPLOG" 2>&1
    local code=$?
    cat "$STEPLOG" >> "$LOGFILE"
    local result=OK
    if [ $code -ne 0 ]; then
        result=FAILED
        ANY_FAILED=1
    elif grep -q "skipped" "$STEPLOG"; then
        result=SKIPPED
        ANY_SKIPPED=1
    fi
    echo "$result"
    NAMES+=("$name")
    RESULTS+=("$result")
}

step "cpu tests " cargo test --features cpu -- --test-threads=1
step "cuda tests" cargo test --features cpu,cuda,compile-nvrtc,cudarc/cuda-version-from-build-system -- --test-threads=1 --nocapture
step "wgpu tests" cargo test --features cpu,wgpu -- --test-threads=1 --nocapture
step "parity    " cargo run --example parity --features cpu,cuda,compile-nvrtc,wgpu,tool-parity,cudarc/cuda-version-from-build-system

echo
echo "============================================"
for i in "${!NAMES[@]}"; do
    echo "  ${NAMES[$i]} ... ${RESULTS[$i]}"
done
echo "============================================"

if [ "$ANY_FAILED" = "1" ] || [ "$ANY_SKIPPED" = "1" ]; then
    echo "Not all passed -- send back the whole $LOGFILE"
    read -n 1 -s -r -p "Press any key to continue..."
    echo
    exit 1
fi

echo "ALL PASSED. Send back $LOGFILE anyway: it names the GPU."
read -n 1 -s -r -p "Press any key to continue..."
echo
exit 0
