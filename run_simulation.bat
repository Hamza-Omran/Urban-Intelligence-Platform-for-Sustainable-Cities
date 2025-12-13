@echo off
echo ====================================
echo Monte Carlo Simulation Runner
echo ====================================
echo.

echo Step 1: Testing setup...
python Scripts/test_simulation.py
if errorlevel 1 (
    echo.
    echo ERROR: Setup test failed!
    pause
    exit /b 1
)

echo.
echo ====================================
echo Step 2: Running simulation...
echo ====================================
echo.
python Scripts/monte_carlo_simulation.py

echo.
echo ====================================
echo Simulation Complete!
echo ====================================
echo.
echo Check results in: Data/gold/
echo.
pause