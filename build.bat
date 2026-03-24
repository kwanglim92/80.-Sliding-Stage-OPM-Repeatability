@echo off
chcp 65001 >nul 2>&1
REM ====================================================
REM  Sliding Stage OPM Repeatability - PyInstaller Build
REM  Usage: build.bat
REM ====================================================

set PROJECT_DIR=%~dp0
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

echo.
echo ============================================
echo   Sliding Stage OPM - PyInstaller Build
echo ============================================
echo.

python -m PyInstaller ^
    --noconfirm ^
    --onefile ^
    --windowed ^
    --name "SlidingStageOPM" ^
    --distpath dist ^
    --workpath build ^
    --specpath build ^
    --hidden-import=src.core ^
    --hidden-import=src.ui ^
    --hidden-import=src.visualization ^
    main.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] PyInstaller build failed!
    pause
    exit /b 1
)

echo.
echo ============================================
echo   Build Complete!
echo ============================================
echo   EXE: dist\SlidingStageOPM.exe
echo.
pause
