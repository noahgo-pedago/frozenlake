#!/bin/bash
# Script de lancement rapide pour FrozenLake GUI

echo "🧊 FrozenLake Q-Learning Lab 🤖"
echo "================================"
echo ""

# Sur macOS, préférer Python de Homebrew pour avoir Tk 8.6+
PYTHON_CMD="python3"
if [[ "$OSTYPE" == "darwin"* ]]; then
    if command -v brew &> /dev/null; then
        BREW_PYTHON=$(brew --prefix)/bin/python3
        if [ -x "$BREW_PYTHON" ]; then
            echo "ℹ️  Utilisation de Python Homebrew pour une meilleure compatibilité Tk"
            PYTHON_CMD="$BREW_PYTHON"
        fi
    fi
fi

# Vérifier si tkinter est installé
echo "Vérification de tkinter..."
if ! $PYTHON_CMD -c "import tkinter" 2>/dev/null; then
    echo "❌ Tkinter n'est pas installé!"
    echo ""
    echo "Installation de python3-tk..."

    # Détecter le système d'exploitation
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macOS détecté"
        echo ""
        echo "Sur macOS, tkinter nécessite une installation spéciale."
        echo "Options:"
        echo "  1. Installer Python via Homebrew: brew install python python-tk"
        echo "  2. Utiliser le Python système (si disponible)"
        echo ""
        if command -v brew &> /dev/null; then
            read -p "Voulez-vous installer python-tk via Homebrew? (o/N) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[OoYy]$ ]]; then
                brew install python python-tk
            else
                echo "Installation annulée."
                exit 1
            fi
        else
            echo "❌ Homebrew n'est pas installé."
            echo "Installez Homebrew depuis https://brew.sh puis exécutez:"
            echo "  brew install python python-tk"
            exit 1
        fi
    elif [ -f /etc/debian_version ]; then
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
        echo "⚠️  Système d'exploitation non reconnu."
        echo "Veuillez installer python3-tk manuellement:"
        echo "  - macOS: brew install python python-tk"
        echo "  - Ubuntu/Debian: sudo apt-get install python3-tk"
        echo "  - Fedora: sudo dnf install python3-tkinter"
        echo "  - Arch: sudo pacman -S tk"
        exit 1
    fi

    # Vérifier à nouveau
    if ! $PYTHON_CMD -c "import tkinter" 2>/dev/null; then
        echo "❌ L'installation de tkinter a échoué."
        exit 1
    fi
    echo "✅ Tkinter installé avec succès!"
else
    echo "✅ Tkinter est disponible!"

    # Sur macOS, vérifier la version de Tk et installer SDL2 pour pygame
    if [[ "$OSTYPE" == "darwin"* ]]; then
        TK_VERSION=$($PYTHON_CMD -c "import tkinter; print(tkinter.TkVersion)" 2>/dev/null)
        echo "ℹ️  Version Tk: $TK_VERSION"

        if (( $(echo "$TK_VERSION < 8.6" | bc -l) )); then
            echo "⚠️  Tk version $TK_VERSION est trop ancienne (8.6+ requis)"
            echo ""
            echo "Sur macOS, il est recommandé d'installer Python via Homebrew:"
            echo "  1. Installez Homebrew: https://brew.sh"
            echo "  2. Installez Python: brew install python python-tk"
            echo "  3. Relancez ce script"
            echo ""
            read -p "Continuer quand même? (o/N) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[OoYy]$ ]]; then
                exit 1
            fi
        fi

        # Vérifier et installer SDL2 pour pygame
        if command -v brew &> /dev/null; then
            if ! brew list sdl2 &> /dev/null; then
                echo ""
                echo "ℹ️  SDL2 requis pour pygame"
                read -p "Installer SDL2 via Homebrew? (o/N) " -n 1 -r
                echo
                if [[ $REPLY =~ ^[OoYy]$ ]]; then
                    echo "Installation de SDL2..."
                    brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf
                else
                    echo "⚠️  pygame pourrait ne pas s'installer correctement sans SDL2"
                fi
            fi
        fi
    fi
fi

echo ""

# Vérifier si le venv existe
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment non trouvé!"
    echo "Création du virtual environment..."
    $PYTHON_CMD -m venv venv
    echo "✅ Virtual environment créé!"
    echo ""
    echo "Installation des dépendances..."
    source venv/bin/activate
    echo "ℹ️  Mise à jour de pip et setuptools..."
    python -m pip install --upgrade pip setuptools wheel -q
    pip install --only-binary pygame-ce -q -r requirements.txt
    echo "✅ Dépendances installées!"
else
    echo "✅ Virtual environment trouvé!"
    source venv/bin/activate
fi

echo ""
echo "🚀 Lancement de l'interface graphique..."
echo ""

python frozenlake_gui.py
