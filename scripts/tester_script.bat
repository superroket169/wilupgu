@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0.."

set PATH=%PATH%;%USERPROFILE%\.cargo\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64

set LOGFILE=test_results.log
echo [%date% %time%] wilupgu test run > "%LOGFILE%"

set ANY_FAILED=0

echo.
echo === [1/5] cargo check --features cuda ===
cargo check --features cuda >> "%LOGFILE%" 2>&1
if errorlevel 1 (set S1=FAILED& set ANY_FAILED=1& echo FAILED) else (set S1=OK& echo OK)

echo.
echo === [2/5] cargo test --features cuda -- --test-threads=1 ===
echo (single-threaded on purpose -- multiple GPU contexts in parallel segfault)
cargo test --features cuda -- --test-threads=1 >> "%LOGFILE%" 2>&1
if errorlevel 1 (set S2=FAILED& set ANY_FAILED=1& echo FAILED) else (set S2=OK& echo OK)

echo.
echo === [3/5] cargo check --features cpu ===
cargo check --features cpu >> "%LOGFILE%" 2>&1
if errorlevel 1 (set S3=FAILED& set ANY_FAILED=1& echo FAILED) else (set S3=OK& echo OK)

echo.
echo === [4/5] cargo test --features cpu -- --test-threads=1 ===
cargo test --features cpu -- --test-threads=1 >> "%LOGFILE%" 2>&1
if errorlevel 1 (set S4=FAILED& set ANY_FAILED=1& echo FAILED) else (set S4=OK& echo OK)

echo.
echo === [5/5] f16/bf16 GEMM check (isolated re-run, real GPU math) ===
echo (already part of step 2 -- re-run alone with output visible for a quick look)
cargo test --features cuda gemm_dtype_validation -- --test-threads=1 --nocapture >> "%LOGFILE%" 2>&1
if errorlevel 1 (set S5=FAILED& set ANY_FAILED=1& echo FAILED) else (set S5=OK& echo OK)

echo.
echo ============================================
echo SUMMARY
echo   [1/5] cargo check --features cuda ........ !S1!
echo   [2/5] cargo test  --features cuda ........ !S2!
echo   [3/5] cargo check --features cpu  ........ !S3!
echo   [4/5] cargo test  --features cpu  ........ !S4!
echo   [5/5] f16/bf16 GEMM isolated re-run ........ !S5!
echo ============================================

if "!ANY_FAILED!"=="1" (
    echo ONE OR MORE STEPS FAILED. Full log: %LOGFILE%
    echo Send the whole log back -- each step's own output is still in there
    echo even though the script kept going past the failure.
    pause
    exit /b 1
)

echo ALL WILUPGU TESTS PASSED.
echo Full log: %LOGFILE%
pause
exit /b 0
