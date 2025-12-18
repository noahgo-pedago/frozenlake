# FrozenLake Gymnasium Project

A Python project implementing reinforcement learning agents for the Gymnasium FrozenLake environment.

## Overview

FrozenLake is a classic grid-world environment where an agent must navigate across a frozen lake from the starting position to the goal while avoiding holes in the ice. The lake is slippery, so the agent doesn't always move in the intended direction.

**Environment Details:**
- **Grid Size:** 4x4 (default) or 8x8
- **Actions:** 4 discrete actions (Left, Down, Right, Up)
- **States:** 16 positions (4x4 grid)
- **Objective:** Navigate from Start (S) to Goal (G) while avoiding Holes (H)

## Project Structure

```
frozenlake/
├── requirements.txt           # Project dependencies
├── run.sh                    # 🚀 Script de lancement rapide (Linux/Mac)
├── run.bat                   # 🚀 Script de lancement rapide (Windows CMD)
├── run.ps1                   # 🚀 Script de lancement rapide (Windows PowerShell)
├── frozenlake_gui.py         # 🎓 Interface graphique interactive (RECOMMANDÉ!)
├── frozenlake_qlearning.py   # Q-learning agent implementation
├── frozenlake_random.py      # Random baseline agent
├── frozenlake_visual_demo.py # Visual demo with graphical rendering
├── venv/                     # Virtual environment
└── README.md                 # This file
```

## 🚀 Lancement Rapide (Recommandé!)

### Linux / Mac:
```bash
./run.sh
```

### Windows:

**Option 1 - Command Prompt (CMD):**
```cmd
run.bat
```
Double-cliquez sur `run.bat` ou exécutez-le depuis CMD.

**Option 2 - PowerShell (recommandé):**
```powershell
.\run.ps1
```
Clic-droit sur `run.ps1` → "Exécuter avec PowerShell"

> **Note PowerShell:** Si vous obtenez une erreur d'exécution de script, exécutez d'abord:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

Ces scripts vont automatiquement:
- ✅ Créer le virtual environment (si nécessaire)
- ✅ Installer les dépendances
- ✅ Lancer l'interface graphique

**C'est la méthode la plus simple pour démarrer!**

## Installation Manuelle

Si vous préférez installer manuellement:

1. **Sur Linux, installer tkinter si nécessaire:**
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# Fedora
sudo dnf install python3-tkinter

# Arch
sudo pacman -S tk
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note:** Tkinter est inclus par défaut avec Python sur Windows et Mac. Sur Linux, il peut nécessiter une installation système.

## Usage

### 🎓 Interface Graphique Interactive (RECOMMANDÉ pour les Étudiants!)

Lancez l'interface graphique complète pour expérimenter avec les paramètres d'apprentissage:

```bash
# Avec le script de lancement (plus simple)
./run.sh  # ou run.bat sur Windows

# Ou manuellement
source venv/bin/activate
python frozenlake_gui.py
```

**Fonctionnalités de l'Interface:**

- ⚙️ **Contrôles des Hyperparamètres** - Ajustez en temps réel:

  | Paramètre | Plage | Description |
  |-----------|-------|-------------|
  | Taux d'apprentissage (α) | 0.01 - 1.0 | Vitesse d'apprentissage de l'agent |
  | Facteur de discount (γ) | 0.0 - 1.0 | Importance des récompenses futures |
  | Décroissance epsilon | 0.9 - 0.999 | Vitesse de transition exploration → exploitation |
  | Nombre d'épisodes | 1000 - 50000 | Durée de l'entraînement |
  | Taille de carte | 4x4 / 8x8 / Personnalisée | Complexité de l'environnement |
  | Glace glissante | On/Off | Stochasticité des mouvements |

- 📊 **Statistiques en Temps Réel:**
  - Progression de l'entraînement (barre de progression)
  - Taux de réussite (%)
  - Epsilon actuel (exploration vs exploitation)
  - Récompense moyenne glissante
  - Temps écoulé
  - Graphique de progression en direct

- 🎮 **Préréglages Prêts à l'Emploi** (ne modifient pas la carte):

  | Préréglage | α | γ | ε decay | Épisodes |
  |------------|---|---|---------|----------|
  | **Débutant** | 0.2 | 0.95 | 0.997 | 5000 |
  | **Standard** | 0.15 | 0.98 | 0.996 | 10000 |
  | **Optimal** | 0.1 | 0.99 | 0.9965 | 15000 |

- 🗺️ **Éditeur de Carte Personnalisée** - Créez vos propres environnements
- 👁️ **Démo Visuelle Intégrée** - Regardez l'agent entraîné jouer

Cette interface est parfaite pour comprendre l'impact de chaque hyperparamètre sur l'apprentissage!

### Visual Demo (Command Line)

Watch the agent learn and play with graphical rendering:

```bash
python frozenlake_visual_demo.py
```

This interactive demo offers:
1. **Random Agent** - Watch an agent move randomly (no learning)
2. **Trained Agent** - See a Q-learning agent navigate successfully
3. **Compare Both** - See random vs trained side-by-side
4. **Quick Demo** - Fast 3-episode demonstration

The visual demo shows the grid with:
- **S** (Start) - Blue square where the agent begins
- **F** (Frozen) - White ice tiles (safe to walk on)
- **H** (Hole) - Black holes (game over if you fall in)
- **G** (Goal) - Green target to reach
- **Agent** - Red circle showing current position

### Train Q-Learning Agent

Train an agent using Q-learning algorithm:

```bash
python frozenlake_qlearning.py
```

This will:
- Train the agent for 10,000 episodes
- Print progress every 1,000 episodes
- Evaluate the trained agent over 100 test episodes
- Display final win rate and average reward

### Run Random Agent Baseline

See how a random agent performs:

```bash
python frozenlake_random.py
```

This demonstrates the baseline performance without learning.

## How It Works

### Q-Learning Algorithm

The Q-learning agent learns by:
1. **Exploration vs Exploitation:** Using epsilon-greedy policy to balance exploring new actions and exploiting learned knowledge
2. **Q-Table Updates:** Learning optimal action values using the Q-learning formula:
   ```
   Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
   ```
   where:
   - α (alpha) = learning rate
   - γ (gamma) = discount factor
   - r = reward
   - s = current state
   - a = action taken
   - s' = next state

3. **Epsilon Decay:** Gradually reducing exploration as the agent learns

### Hyperparameters

Default parameters in `frozenlake_qlearning.py`:
- **Learning Rate (α):** 0.1
- **Discount Factor (γ):** 0.99
- **Initial Epsilon:** 1.0 (100% exploration)
- **Epsilon Decay:** 0.995
- **Minimum Epsilon:** 0.01
- **Training Episodes:** 10,000

**Comprendre les Hyperparamètres:**

| Paramètre | Effet si trop bas | Effet si trop haut |
|-----------|-------------------|-------------------|
| **Learning Rate (α)** | Apprentissage très lent | Apprentissage instable, oscillations |
| **Discount Factor (γ)** | Agent myope, ignore le futur | Peut survaloriser des chemins longs |
| **Epsilon Decay** | Reste en exploration trop longtemps | Exploite trop tôt, manque de solutions |

## Customization

### Modify Hyperparameters

Edit the agent initialization in `frozenlake_qlearning.py`:

```python
agent = QLearningAgent(
    env=env,
    learning_rate=0.1,      # Adjust learning rate
    discount_factor=0.95,   # Adjust discount factor
    epsilon=1.0,            # Initial exploration rate
    epsilon_decay=0.99,     # Adjust decay rate
    epsilon_min=0.01        # Minimum exploration
)
```

### Use 8x8 Map

Change the environment creation to use a larger map:

```python
env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)
```

### Disable Slippery Ice

For deterministic movement:

```python
env = gym.make("FrozenLake-v1", is_slippery=False)
```

## Expected Results

- **Random Agent:** ~1-2% win rate
- **Q-Learning Agent:** ~70-80% win rate (4x4 slippery map)

The Q-learning agent significantly outperforms random actions by learning optimal paths.

## Resources

- [Gymnasium FrozenLake Documentation](https://gymnasium.farama.org/environments/toy_text/frozen_lake/)
- [Q-Learning Algorithm](https://en.wikipedia.org/wiki/Q-learning)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

## License

This project is open source and available for educational purposes.
