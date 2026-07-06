@echo off
setlocal
cd /d "%~dp0.."

set PATH=%PATH%;%USERPROFILE%\.cargo\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64

set LOGFILE=test_results.log
echo [%date% %time%] wilupgu test run > "%LOGFILE%"

echo.
echo === [1/5] cargo check --features cuda ===
cargo check --features cuda >> "%LOGFILE%" 2>&1
if errorlevel 1 goto :fail_check_cuda
echo OK

echo.
echo === [2/5] cargo test --features cuda -- --test-threads=1 ===
echo (single-threaded on purpose -- multiple GPU contexts in parallel segfault)
cargo test --features cuda -- --test-threads=1 >> "%LOGFILE%" 2>&1
if errorlevel 1 goto :fail_test_cuda
echo OK

echo.
echo === [3/5] cargo check --features cpu ===
cargo check --features cpu >> "%LOGFILE%" 2>&1
if errorlevel 1 goto :fail_check_cpu
echo OK

echo.
echo === [4/5] cargo test --features cpu -- --test-threads=1 ===
cargo test --features cpu -- --test-threads=1 >> "%LOGFILE%" 2>&1
if errorlevel 1 goto :fail_test_cpu
echo OK

echo.
echo === [5/5] f16 GEMM check (isolated re-run, real GPU math) ===
echo (already part of step 2 -- re-run alone with output visible for a quick look)
cargo test --features cuda f16_matmul_matches_f32_matmul -- --test-threads=1 --nocapture >> "%LOGFILE%" 2>&1
if errorlevel 1 goto :fail_f16
echo OK

echo.
echo ============================================
echo ALL WILUPGU TESTS PASSED.
echo Full log: %LOGFILE%
echo ============================================
pause
exit /b 0

:fail_check_cuda
echo.
echo FAILED at step 1: cargo check --features cuda
echo This is the CUDA backend compiling for the very first time ever --
echo Dtype/CudaBuffer enum, the define_launch! macro move, gemm dtype
echo dispatch. Open %LOGFILE% and look at the last ~50 lines for the error.
pause
exit /b 1

:fail_test_cuda
echo.
echo FAILED at step 2: cargo test --features cuda
echo Open %LOGFILE% and look at the last ~80 lines.
pause
exit /b 1

:fail_check_cpu
echo.
echo FAILED at step 3: cargo check --features cpu
echo This one is surprising -- it already passed on the dev machine. Open
echo %LOGFILE% and check whether it's actually a cpu-feature issue or
echo something environmental (missing toolchain component, etc).
pause
exit /b 1

:fail_test_cpu
echo.
echo FAILED at step 4: cargo test --features cpu
echo Open %LOGFILE% and look at the last ~80 lines.
pause
exit /b 1

:fail_f16
echo.
echo FAILED at step 5: f16 GEMM test specifically.
echo This is the newest, riskiest piece -- real half::f16 conversion plus
echo cuBLAS f16 GEMM through the new dtype-dispatching gemm_matmul. Open
echo %LOGFILE% and look at the last ~40 lines.
pause
exit /b 1
