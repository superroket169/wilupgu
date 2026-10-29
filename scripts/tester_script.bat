@echo off
rem Runs wilupgu's tests and the parity example on this machine's GPUs.
rem A step is SKIPPED when it found no device for a backend; that is not a pass.
setlocal enabledelayedexpansion
cd /d "%~dp0.."

set PATH=%PATH%;%USERPROFILE%\.cargo\bin;%CUDA_PATH%\bin;%CUDA_PATH%\bin\x64

set LOGFILE=test_results.log
set STEPLOG=%TEMP%\wilupgu_step.log

echo [%date% %time%] wilupgu test run > "%LOGFILE%"
echo --- rustc: >> "%LOGFILE%"
rustc --version >> "%LOGFILE%" 2>&1
echo --- nvidia-smi -L: >> "%LOGFILE%"
nvidia-smi -L >> "%LOGFILE%" 2>&1
echo --- nvcc --version: >> "%LOGFILE%"
nvcc --version >> "%LOGFILE%" 2>&1

set ANY_FAILED=0
set ANY_SKIPPED=0

call :step S1 "cpu tests " "cargo test --features cpu -- --test-threads=1"
call :step S2 "cuda tests" "cargo test --features cpu,cuda,compile-nvrtc -- --test-threads=1 --nocapture"
call :step S3 "wgpu tests" "cargo test --features cpu,wgpu -- --test-threads=1 --nocapture"
call :step S4 "parity    " "cargo run --example parity --features cpu,cuda,compile-nvrtc,wgpu,tool-parity"

del "%STEPLOG%" 2>nul

echo.
echo ============================================
echo   cpu tests  ... !S1!
echo   cuda tests ... !S2!
echo   wgpu tests ... !S3!
echo   parity     ... !S4!
echo ============================================

if "!ANY_FAILED!!ANY_SKIPPED!" neq "00" (
    echo Not all passed -- send back the whole %LOGFILE%
    pause
    exit /b 1
)

echo ALL PASSED. Send back %LOGFILE% anyway: it names the GPU.
pause
exit /b 0

rem :step VAR "name" "command" -- runs the command, sets VAR to OK, FAILED or SKIPPED.
:step
set NAME=%~2
set CMD=%~3
echo === %NAME%: !CMD! ===
echo === %NAME%: !CMD! === >> "%LOGFILE%"
!CMD! > "%STEPLOG%" 2>&1
set CODE=!errorlevel!
type "%STEPLOG%" >> "%LOGFILE%"
set RESULT=OK
if not "!CODE!"=="0" (
    set RESULT=FAILED
    set ANY_FAILED=1
) else (
    findstr /c:"skipped" "%STEPLOG%" >nul && (
        set RESULT=SKIPPED
        set ANY_SKIPPED=1
    )
)
echo !RESULT!
set %1=!RESULT!
exit /b 0
