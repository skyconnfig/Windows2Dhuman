@echo off
REM Digital Human Web Application - Python Server Startup Script
REM ========================================
echo Digital Human Web Application - Python Server
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not detected. Please install Python 3.6 or higher
    echo Download: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Display Python version
echo [INFO] Python version detected:
python --version
echo.

REM Set project directory
set PROJECT_DIR=%~dp0
echo [INFO] Project directory: %PROJECT_DIR%
echo.

REM Set server port
set PORT=8080
echo [INFO] Server port: %PORT%
echo.

REM Check if port is in use
netstat -an | findstr :%PORT% >nul 2>&1
if not errorlevel 1 (
    echo [WARNING] Port %PORT% is in use, trying port 8081
    set PORT=8082
)

echo [INFO] Starting Python HTTP server...
echo [INFO] Access URL: http://localhost:%PORT%
echo [INFO] Press Ctrl+C to stop server
echo ========================================
echo.

REM Change to project directory
cd /d "%PROJECT_DIR%"

REM Start Python HTTP server
REM Python 3.x uses http.server module
python -m http.server %PORT% --bind 127.0.0.1

REM If above command fails, try Python 2.x way
if errorlevel 1 (
    echo [INFO] Trying Python 2.x method...
    python -m SimpleHTTPServer %PORT%
)

echo.
echo [INFO] Server stopped
pause