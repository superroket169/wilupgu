#!/usr/bin/env bash
# Runs the CPU tests under Miri. Miri runs the program in an interpreter
# that checks every memory access, so undefined behavior in unsafe code is
# reported where it happens: use after free, out-of-bounds access, a pointer
# used after a `&mut` invalidated it (Stacked Borrows), data races. It sees
# only the code a test runs, and can't run CUDA or wgpu.
#
#   scripts/miri.sh             every test in its own process, one line each:
#                               ok, UB (with where) or FAILED; then a count
#   scripts/miri.sh <filter>    the tests whose name contains <filter>, with
#                               Miri's whole report
#   SEEDS=N scripts/miri.sh <filter>
#                               each test N times, under N thread schedules
#                               and allocator choices (-Zmiri-many-seeds)
#
# The flags match CI's Miri job: isolation off (some tests use files), and
# every freed address reused when it can be, so address-reuse bugs show.
cd "$(dirname "$0")/.." || exit 1

if ! cargo +nightly miri --version > /dev/null 2>&1; then
    rustup component add miri --toolchain nightly || exit 1
fi

export MIRIFLAGS="-Zmiri-disable-isolation -Zmiri-address-reuse-rate=1.0"
if [ -n "$SEEDS" ]; then
    MIRIFLAGS="$MIRIFLAGS -Zmiri-many-seeds=0..$SEEDS"
fi

if [ -n "$1" ]; then
    exec cargo +nightly miri test --features cpu --lib -- "$1"
fi

tests=$(cargo test -q --features cpu --lib -- --list 2> /dev/null | sed -n 's/: test$//p')
ok=0
ub=0
failed=0
for t in $tests; do
    log=$(cargo +nightly miri test -q --features cpu --lib -- --exact "$t" 2>&1)
    if grep -q "Undefined Behavior" <<< "$log"; then
        where=$(grep -m1 -A1 "Undefined Behavior" <<< "$log" | tail -1 | sed 's/^ *--> //')
        echo "UB      $t  @ $where"
        ub=$((ub + 1))
    elif grep -q "test result: ok" <<< "$log"; then
        echo "ok      $t"
        ok=$((ok + 1))
    else
        echo "FAILED  $t"
        failed=$((failed + 1))
    fi
done
echo
echo "$ok ok, $ub UB, $failed failed"
[ "$ub" = 0 ] && [ "$failed" = 0 ]
