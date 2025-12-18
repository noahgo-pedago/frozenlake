# FrozenLake Q-Learning Lab - PowerShell Launcher
# Requires: PowerShell 5.1+ (included in Windows 10/11)

$Host.UI.RawUI.WindowTitle = "FrozenLake Q-Learning Lab"

Write-Host "FrozenLake Q-Learning Lab" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "[ERROR] Python n'est pas installe ou n'est pas dans le PATH!" -ForegroundColor Red
    Write-Host "Telechargez Python depuis https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host "Assurez-vous de cocher 'Add Python to PATH' lors de l'installation." -ForegroundColor Yellow
    Read-Host "Appuyez sur Entree pour quitter"
    exit 1
}

# Get Python version
$pythonVersion = & python --version 2>&1
Write-Host "[OK] $pythonVersion detecte" -ForegroundColor Green

# Check if tkinter is available
Write-Host "Verification de tkinter..."
$tkinterCheck = & python -c "import tkinter" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Tkinter n'est pas installe!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Sur Windows, tkinter est normalement inclus avec Python." -ForegroundColor Yellow
    Write-Host "Reinstallez Python depuis https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host "et assurez-vous de cocher 'tcl/tk and IDLE' dans les options." -ForegroundColor Yellow
    Read-Host "Appuyez sur Entree pour quitter"
    exit 1
}
Write-Host "[OK] Tkinter est disponible!" -ForegroundColor Green

Write-Host ""

# Check if venv exists
if (-not (Test-Path "venv")) {
    Write-Host "[INFO] Virtual environment non trouve!" -ForegroundColor Yellow
    Write-Host "Creation du virtual environment..."

    & python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Echec de la creation du virtual environment." -ForegroundColor Red
        Read-Host "Appuyez sur Entree pour quitter"
        exit 1
    }
    Write-Host "[OK] Virtual environment cree!" -ForegroundColor Green

    Write-Host ""
    Write-Host "Installation des dependances..."

    # Activate venv
    & ".\venv\Scripts\Activate.ps1"

    & pip install -q -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Echec de l'installation des dependances." -ForegroundColor Red
        Read-Host "Appuyez sur Entree pour quitter"
        exit 1
    }
    Write-Host "[OK] Dependances installees!" -ForegroundColor Green
} else {
    Write-Host "[OK] Virtual environment trouve!" -ForegroundColor Green
    & ".\venv\Scripts\Activate.ps1"
}

Write-Host ""
Write-Host "Lancement de l'interface graphique..." -ForegroundColor Cyan
Write-Host ""

& python frozenlake_gui.py
