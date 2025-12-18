@echo off
chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion

echo FrozenLake Q-Learning Lab
echo ================================
echo.

:: Check if Python is installed
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python n'est pas installe ou n'est pas dans le PATH!
    echo Telechargez Python depuis https://www.python.org/downloads/
    echo Assurez-vous de cocher "Add Python to PATH" lors de l'installation.
    pause
    exit /b 1
)

:: Check Python version
for /f "tokens=2 delims= " %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Python %PYTHON_VERSION% detecte

:: Check if tkinter is available
echo Verification de tkinter...
python -c "import tkinter" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Tkinter n'est pas installe!
    echo.
    echo Sur Windows, tkinter est normalement inclus avec Python.
    echo Reinstallez Python depuis https://www.python.org/downloads/
    echo et assurez-vous de cocher "tcl/tk and IDLE" dans les options.
    pause
    exit /b 1
)
echo [OK] Tkinter est disponible!

echo.

:: Check if venv exists
if not exist "venv" (
    echo [INFO] Virtual environment non trouve!
    echo Creation du virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Echec de la creation du virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment cree!
    echo.
    echo Installation des dependances...
    call venv\Scripts\activate.bat
    echo Mise a jour de pip et setuptools...
    python -m pip install --upgrade pip setuptools wheel -q
    pip install --only-binary pygame-ce -q -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Echec de l'installation des dependances.
        pause
        exit /b 1
    )
    echo [OK] Dependances installees!
) else (
    echo [OK] Virtual environment trouve!
    call venv\Scripts\activate.bat
)

echo.
echo Lancement de l'interface graphique...
echo.

python frozenlake_gui.py

pause
endlocal
