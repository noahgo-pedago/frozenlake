@echo off
REM Script de lancement rapide pour FrozenLake GUI (Windows)

echo 🧊 FrozenLake Q-Learning Lab 🤖
echo ================================
echo.

REM Vérifier si le venv existe
if not exist "venv\" (
    echo ❌ Virtual environment non trouvé!
    echo Création du virtual environment...
    python -m venv venv
    echo ✅ Virtual environment créé!
    echo.
    echo Installation des dépendances...
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
    echo ✅ Dépendances installées!
) else (
    echo ✅ Virtual environment trouvé!
    call venv\Scripts\activate.bat
)

echo.
echo 🚀 Lancement de l'interface graphique...
echo.

python frozenlake_gui.py

pause
