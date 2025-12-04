#!/bin/bash
# Script de lancement rapide pour FrozenLake GUI

echo "🧊 FrozenLake Q-Learning Lab 🤖"
echo "================================"
echo ""

# Vérifier si tkinter est installé
echo "Vérification de tkinter..."
if ! python3 -c "import tkinter" 2>/dev/null; then
    echo "❌ Tkinter n'est pas installé!"
    echo ""
    echo "Installation de python3-tk..."

    # Détecter la distribution Linux
    if [ -f /etc/debian_version ]; then
        echo "Distribution Debian/Ubuntu détectée"
        sudo apt-get update
        sudo apt-get install -y python3-tk
    elif [ -f /etc/fedora-release ]; then
        echo "Distribution Fedora détectée"
        sudo dnf install -y python3-tkinter
    elif [ -f /etc/arch-release ]; then
        echo "Distribution Arch détectée"
        sudo pacman -S --noconfirm tk
    else
        echo "⚠️  Distribution non reconnue."
        echo "Veuillez installer python3-tk manuellement:"
        echo "  - Ubuntu/Debian: sudo apt-get install python3-tk"
        echo "  - Fedora: sudo dnf install python3-tkinter"
        echo "  - Arch: sudo pacman -S tk"
        exit 1
    fi

    # Vérifier à nouveau
    if ! python3 -c "import tkinter" 2>/dev/null; then
        echo "❌ L'installation de tkinter a échoué."
        exit 1
    fi
    echo "✅ Tkinter installé avec succès!"
else
    echo "✅ Tkinter est disponible!"
fi

echo ""

# Vérifier si le venv existe
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment non trouvé!"
    echo "Création du virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment créé!"
    echo ""
    echo "Installation des dépendances..."
    source venv/bin/activate
    pip install -q -r requirements.txt
    echo "✅ Dépendances installées!"
else
    echo "✅ Virtual environment trouvé!"
    source venv/bin/activate
fi

echo ""
echo "🚀 Lancement de l'interface graphique..."
echo ""

python frozenlake_gui.py
