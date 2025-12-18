"""
FrozenLake Interactive GUI Application

A user-friendly interface for students to experiment with Q-learning
and observe how different hyperparameters affect agent performance.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import gymnasium as gym
import numpy as np
import threading
import time
from queue import Queue
import random
import sys


class ToolTip:
    """Create a tooltip for a given widget."""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")

        # Better colors for both light and dark mode on macOS
        label = tk.Label(self.tooltip, text=self.text, justify='left',
                        background="#2b2b2b",  # Dark gray (works in both modes)
                        foreground="#ffffff",  # White text (readable on dark bg)
                        relief='solid', borderwidth=2,
                        font=("Arial", 10),  # Slightly larger font
                        wraplength=350,  # More space for text
                        padx=12, pady=8)  # Padding so text doesn't touch edges
        label.pack()

    def hide_tooltip(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None


def has_valid_path(grid):
    """Check if there's a valid path from start (0,0) to goal using BFS."""
    size = len(grid)
    start = (0, 0)
    goal = (size - 1, size - 1)

    # BFS to find path
    from collections import deque
    queue = deque([start])
    visited = {start}

    # Directions: left, down, right, up
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]

    while queue:
        row, col = queue.popleft()

        # Check if we reached the goal
        if (row, col) == goal:
            return True

        # Explore neighbors
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc

            # Check bounds
            if 0 <= new_row < size and 0 <= new_col < size:
                if (new_row, new_col) not in visited:
                    # Check if it's not a hole
                    if grid[new_row][new_col] != 'H':
                        visited.add((new_row, new_col))
                        queue.append((new_row, new_col))

    return False


def generate_random_map(size=8, hole_probability=0.2, max_attempts=100):
    """Generate a random valid FrozenLake map with guaranteed path to goal."""
    if size < 4:
        size = 4

    attempts = 0
    while attempts < max_attempts:
        attempts += 1

        # Start with all frozen
        grid = [['F' for _ in range(size)] for _ in range(size)]

        # Set start and goal
        grid[0][0] = 'S'
        grid[size-1][size-1] = 'G'

        # Add holes randomly
        for i in range(size):
            for j in range(size):
                if grid[i][j] == 'F' and random.random() < hole_probability:
                    grid[i][j] = 'H'

        # Check if there's a valid path
        if has_valid_path(grid):
            return [''.join(row) for row in grid]

    # Fallback: create a simple guaranteed path if max attempts reached
    grid = [['F' for _ in range(size)] for _ in range(size)]
    grid[0][0] = 'S'
    grid[size-1][size-1] = 'G'

    # Create a safe path along the edge
    for i in range(size):
        grid[i][0] = 'F' if grid[i][0] == 'F' else grid[i][0]
        grid[size-1][i] = 'F' if grid[size-1][i] == 'F' else grid[size-1][i]

    # Add some random holes away from the safe path
    for i in range(1, size - 1):
        for j in range(2, size - 1):
            if random.random() < hole_probability * 0.5:
                grid[i][j] = 'H'

    return [''.join(row) for row in grid]


class QLearningAgent:
    """Q-Learning agent implementation with customizable action biases and advanced penalties."""

    def __init__(self, env, learning_rate=0.1, discount_factor=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 action_biases=None, reward_shaping=None, advanced_penalties=None):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = np.zeros([env.observation_space.n, env.action_space.n])

        # Action biases: [left, down, right, up] - higher = more likely
        self.action_biases = action_biases if action_biases is not None else [1.0, 1.0, 1.0, 1.0]

        # Reward shaping: custom rewards
        self.reward_shaping = reward_shaping if reward_shaping is not None else {
            'goal': 1.0,
            'hole': 0.0,
            'step': 0.0  # Penalty per step
        }

        # Advanced penalties: custom penalties for specific behaviors
        self.advanced_penalties = advanced_penalties if advanced_penalties is not None else {
            'wall_hit': 0.0,       # Penalty for hitting wall (staying in same position)
            'revisit': 0.0,        # Penalty for revisiting a position
            'loop': 0.0,           # Penalty for looping (visiting same position twice in short time)
            'distance': 0.0        # Bonus/penalty based on distance to goal
        }

        # Tracking for episode
        self.visited_states = set()
        self.recent_states = []  # For loop detection
        self.last_state = None

    def choose_action(self, state, greedy=False):
        """Choose action using epsilon-greedy policy with action biases."""
        if greedy or np.random.random() >= self.epsilon:
            # Exploitation: choose best action with bias
            q_values = self.q_table[state].copy()
            # Apply biases to Q-values
            biased_q = q_values * self.action_biases
            return np.argmax(biased_q)
        else:
            # Exploration: random with bias
            biases = np.array(self.action_biases)
            probs = biases / biases.sum()
            return np.random.choice(4, p=probs)

    def update_q_table(self, state, action, reward, next_state, done):
        """Update Q-table using Q-learning formula with reward shaping and advanced penalties."""
        # Apply reward shaping
        shaped_reward = reward
        if reward > 0:  # Goal reached
            shaped_reward = self.reward_shaping['goal']
        elif done and reward == 0:  # Hole
            shaped_reward = self.reward_shaping['hole']
        shaped_reward += self.reward_shaping['step']  # Step penalty/bonus

        # Apply advanced penalties
        # Wall hit detection (stayed in same position)
        if self.last_state is not None and state == next_state and not done:
            shaped_reward += self.advanced_penalties['wall_hit']

        # Revisit penalty
        if next_state in self.visited_states:
            shaped_reward += self.advanced_penalties['revisit']

        # Loop detection (visited same state in last 5 steps)
        if len(self.recent_states) >= 5 and next_state in self.recent_states[-5:]:
            shaped_reward += self.advanced_penalties['loop']

        # Distance-based reward (Manhattan distance to goal)
        if self.advanced_penalties['distance'] != 0.0:
            grid_size = int(np.sqrt(self.env.observation_space.n))
            goal_state = self.env.observation_space.n - 1

            next_row, next_col = divmod(next_state, grid_size)
            goal_row, goal_col = divmod(goal_state, grid_size)
            distance = abs(next_row - goal_row) + abs(next_col - goal_col)

            # Negative distance means closer to goal = positive reward
            shaped_reward += self.advanced_penalties['distance'] * (-distance / (grid_size * 2))

        # Update tracking
        self.visited_states.add(next_state)
        self.recent_states.append(next_state)
        if len(self.recent_states) > 10:
            self.recent_states.pop(0)
        self.last_state = next_state

        current_q = self.q_table[state, action]
        if done:
            target_q = shaped_reward
        else:
            target_q = shaped_reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state, action] = current_q + self.lr * (target_q - current_q)

    def reset_episode_tracking(self):
        """Reset tracking variables for new episode."""
        self.visited_states = set()
        self.recent_states = []
        self.last_state = None

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class FrozenLakeGUI:
    """Main GUI application for FrozenLake training and visualization."""

    def __init__(self, root):
        self.root = root
        self.root.title("FrozenLake Q-Learning Lab - Interface Pédagogique")
        self.root.geometry("1500x900")

        # Training state
        self.agent = None
        self.env = None
        self.is_training = False
        self.is_demo_running = False
        self.training_thread = None
        self.message_queue = Queue()
        self.custom_map = None
        self.rewards_history = []
        self.last_graph_update = 0  # Track last graph update to avoid frequent updates

        # UI Mode
        self.advanced_mode = False  # Start in simple mode

        # Setup GUI
        self.setup_gui()
        self.check_queue()

    def setup_gui(self):
        """Setup all GUI components with resizable panes."""

        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Title with help button
        title_frame = ttk.Frame(main_frame)
        title_frame.grid(row=0, column=0, pady=10, sticky=(tk.W, tk.E))

        title_label = ttk.Label(title_frame, text="🧊 FrozenLake Q-Learning Lab 🤖",
                                font=('Arial', 20, 'bold'))
        title_label.pack(side=tk.LEFT, padx=10)

        help_button = ttk.Button(title_frame, text="❓ Guide d'Utilisation",
                                command=self.show_help)
        help_button.pack(side=tk.LEFT)

        # Main horizontal PanedWindow (left/right split)
        self.main_paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        self.main_paned.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Left panel - Parameters (in a container for the paned window)
        left_container = ttk.Frame(self.main_paned)
        self.setup_parameters_panel(left_container)
        self.main_paned.add(left_container, weight=1)

        # Right side - Vertical PanedWindow (top: visualization, bottom: console)
        self.right_paned = ttk.PanedWindow(self.main_paned, orient=tk.VERTICAL)
        self.main_paned.add(self.right_paned, weight=3)

        # Top right - Visualization
        viz_container = ttk.Frame(self.right_paned)
        self.setup_visualization_panel(viz_container)
        self.right_paned.add(viz_container, weight=2)

        # Bottom right - Console
        console_container = ttk.Frame(self.right_paned)
        self.setup_console_panel(console_container)
        self.right_paned.add(console_container, weight=1)

    def show_help(self):
        """Show help window with explanations."""
        help_window = tk.Toplevel(self.root)
        help_window.title("Guide d'Utilisation - Q-Learning")
        help_window.geometry("800x600")
        help_window.transient(self.root)

        text = scrolledtext.ScrolledText(help_window, wrap=tk.WORD, font=('Arial', 10))
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        help_text = """
╔══════════════════════════════════════════════════════════════════════╗
║           GUIDE D'UTILISATION - FROZENLAKE Q-LEARNING LAB             ║
╚══════════════════════════════════════════════════════════════════════╝

📖 QU'EST-CE QUE LE Q-LEARNING?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Le Q-Learning est une technique d'apprentissage par renforcement où un agent
apprend à naviguer dans un environnement en essayant différentes actions et
en apprenant de leurs résultats (récompenses).

🎮 LE JEU FROZENLAKE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- S (Start) : Point de départ
- F (Frozen): Glace sûre, on peut marcher dessus
- H (Hole)  : Trou dans la glace, GAME OVER!
- G (Goal)  : Objectif à atteindre = +1 point

L'agent doit apprendre à aller de S à G sans tomber dans les trous (H).

⚙️ COMPRENDRE LES HYPERPARAMÈTRES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 TAUX D'APPRENTISSAGE (α - Alpha)
   Valeur: 0.01 à 1.0 | Recommandé: 0.1-0.3

   • Plus il est ÉLEVÉ (0.5-1.0):
     → Apprend vite mais peut être instable
     → Bonne pour des tests rapides
     → Peut "oublier" ce qu'il a appris avant

   • Plus il est BAS (0.01-0.2):
     → Apprend lentement mais de façon stable
     → Bon pour l'apprentissage final
     → Mémorise mieux les expériences

   🎯 CONSEIL: Commencez avec 0.2 pour voir des résultats rapidement!

💰 FACTEUR DE DISCOUNT (γ - Gamma)
   Valeur: 0.0 à 1.0 | Recommandé: 0.95-0.99

   • ÉLEVÉ (0.95-0.99):
     → L'agent planifie à long terme
     → Pense aux récompenses futures
     → Meilleur pour trouver le chemin optimal

   • BAS (0.5-0.8):
     → L'agent est "myope"
     → Ne regarde que les récompenses immédiates
     → Peut rater le meilleur chemin

   🎯 CONSEIL: Utilisez 0.95 ou plus pour FrozenLake!

🎲 DÉCROISSANCE EPSILON (Exploration vs Exploitation)
   Valeur: 0.90 à 0.999 | Recommandé: 0.995-0.999

   L'epsilon contrôle l'exploration:
   • Début: epsilon = 1.0 → 100% d'exploration (actions aléatoires)
   • Fin: epsilon → 0.01 → 99% exploitation (utilise ce qu'il a appris)

   • DÉCROISSANCE RAPIDE (0.95-0.98):
     → Arrête d'explorer rapidement
     → Converge vite mais peut rater de meilleures solutions

   • DÉCROISSANCE LENTE (0.995-0.999):
     → Continue d'explorer longtemps
     → Trouve souvent de meilleurs chemins
     → Nécessite plus d'épisodes

   🎯 CONSEIL: Utilisez 0.995 pour un bon équilibre!

📈 NOMBRE D'ÉPISODES
   Recommandé: 5000-15000 pour 4x4, 20000+ pour 8x8

   Un épisode = une partie complète (de S à G ou à un trou)

   • TROP PEU (<2000):
     → L'agent n'a pas assez appris
     → Taux de réussite faible

   • OPTIMAL (5000-15000):
     → Bon équilibre temps/performance
     → L'agent a le temps d'apprendre

   🎯 CONSEIL: 10000 épisodes est un bon point de départ!

🌊 GLACE GLISSANTE

   • ACTIVÉE (Stochastique):
     → L'agent ne va pas toujours où il veut (33% de chance)
     → Plus réaliste et difficile
     → Nécessite plus d'entraînement

   • DÉSACTIVÉE (Déterministe):
     → L'agent va exactement où il veut
     → Plus facile à apprendre
     → Bon pour débuter et comprendre

   🎯 CONSEIL: Commencez SANS glace glissante pour voir l'apprentissage!

📊 INTERPRÉTATION DES RÉSULTATS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ BON APPRENTISSAGE:
   • Taux de réussite: >70% (sans glace), >60% (avec glace)
   • Epsilon final: <0.1 (a fini d'explorer)
   • Courbe: monte progressivement

❌ APPRENTISSAGE INSUFFISANT:
   • Taux de réussite: <30%
   • Possible raisons:
     → Pas assez d'épisodes
     → Taux d'apprentissage trop bas
     → Décroissance epsilon trop rapide

🎯 PRÉRÉGLAGES RECOMMANDÉS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎓 DÉBUTANT (Pour comprendre):
   • Sans glace glissante
   • 5000 épisodes
   • α=0.2, γ=0.95
   • Résultat attendu: >90%

⚡ STANDARD (Équilibré):
   • Avec glace glissante
   • 10000 épisodes
   • α=0.15, γ=0.98
   • Résultat attendu: 65-75%

🎯 OPTIMAL (Meilleure performance):
   • Avec glace glissante
   • 15000 épisodes
   • α=0.1, γ=0.99
   • Résultat attendu: 75-85%

💡 CONSEILS PRATIQUES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Commencez avec le préréglage DÉBUTANT pour voir que ça marche!
2. Activez ensuite la glace glissante pour le vrai challenge
3. Augmentez les épisodes si le taux de réussite est trop bas
4. Regardez la courbe d'apprentissage dans le graphique
5. La démo visuelle montre comment l'agent a appris à jouer

🐛 PROBLÈMES COURANTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❓ L'agent n'apprend pas (taux <10%)?
   → Augmentez le nombre d'épisodes à 15000+
   → Augmentez le taux d'apprentissage à 0.2-0.3
   → Ralentissez la décroissance epsilon (0.997+)

❓ L'apprentissage est instable?
   → Réduisez le taux d'apprentissage (0.1)
   → Augmentez le facteur de discount (0.99)

❓ Ça prend trop de temps?
   → Réduisez le nombre d'épisodes
   → Augmentez la fréquence de mise à jour UI
   → Utilisez le préréglage RAPIDE

════════════════════════════════════════════════════════════════════════
"""
        text.insert('1.0', help_text)
        text.config(state='disabled')

    def setup_parameters_panel(self, parent):
        """Setup hyperparameter controls panel with scrollbar."""

        # Container frame
        container = ttk.LabelFrame(parent, text="⚙️ Paramètres d'Apprentissage", padding="5")
        container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        # Canvas with scrollbar
        canvas = tk.Canvas(container, highlightthickness=0, width=380, height=600)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Enable mouse wheel scrolling - bind to canvas and container
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        def bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", on_mousewheel)

        def unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")

        # Bind mouse wheel only when hovering over the panel
        canvas.bind("<Enter>", bind_mousewheel)
        canvas.bind("<Leave>", unbind_mousewheel)
        container.bind("<Enter>", bind_mousewheel)
        container.bind("<Leave>", unbind_mousewheel)

        # Now use scrollable_frame as the parent for all parameters
        params_frame = scrollable_frame
        row = 0

        # Environment settings
        env_label = ttk.Label(params_frame, text="🎮 Environnement:",
                             font=('Arial', 10, 'bold'))
        env_label.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        ToolTip(env_label, "Configuration de la carte de jeu")
        row += 1

        # Map size
        map_label = ttk.Label(params_frame, text="Taille de la carte:")
        map_label.grid(row=row, column=0, sticky=tk.W, pady=5)
        ToolTip(map_label, "4x4 = facile et rapide\n8x8 = plus difficile\nPersonnalisée = créez votre propre carte")

        self.map_size = tk.StringVar(value="4x4")
        map_combo = ttk.Combobox(params_frame, textvariable=self.map_size,
                                values=["4x4", "8x8", "Personnalisée"], state="readonly", width=15)
        map_combo.grid(row=row, column=1, sticky=tk.W, pady=5)
        map_combo.bind('<<ComboboxSelected>>', self.on_map_size_change)
        row += 1

        # Custom map button - ALWAYS ENABLED NOW
        self.random_map_button = ttk.Button(params_frame, text="✏️ Créer ma propre carte",
                                           command=self.generate_random_map)
        self.random_map_button.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        ToolTip(self.random_map_button, "Créez une carte personnalisée de n'importe quelle taille (4x4 à 12x12)")
        row += 1

        # Show status of custom map
        self.custom_map_status = ttk.Label(params_frame, text="", foreground="green", font=('Arial', 8))
        self.custom_map_status.grid(row=row, column=0, columnspan=2, sticky=tk.W)
        row += 1

        # Slippery
        slippery_frame = ttk.Frame(params_frame)
        slippery_frame.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=5)
        self.is_slippery = tk.BooleanVar(value=False)  # Changed default to False
        slippery_check = ttk.Checkbutton(slippery_frame, text="Glace glissante",
                                        variable=self.is_slippery)
        slippery_check.pack(side=tk.LEFT)
        ToolTip(slippery_check, "❄️ DÉSACTIVÉ (recommandé pour débuter):\n"
                "L'agent va exactement où vous voulez\n\n"
                "❄️ ACTIVÉ (plus difficile):\n"
                "33% de chance d'aller dans une direction aléatoire")
        row += 1

        # Separator
        ttk.Separator(params_frame, orient='horizontal').grid(row=row, column=0, columnspan=2,
                                                              sticky=(tk.W, tk.E), pady=10)
        row += 1

        # === SIMPLE TRAINING OPTIONS (no technical sliders) ===
        training_label = ttk.Label(params_frame, text="⚙️ Options d'entraînement:",
                                   font=('Arial', 10, 'bold'))
        training_label.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        row += 1

        # Episodes (keep this simple)
        ep_label = ttk.Label(params_frame, text="Nombre d'essais:")
        ep_label.grid(row=row, column=0, sticky=tk.W, pady=5)
        ToolTip(ep_label, "Plus d'essais = meilleur apprentissage mais plus long")

        self.episodes = tk.IntVar(value=10000)
        episodes_spinbox = ttk.Spinbox(params_frame, from_=1000, to=50000, increment=1000,
                                       textvariable=self.episodes, width=15)
        episodes_spinbox.grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1

        # Update frequency (keep but simplify)
        update_label = ttk.Label(params_frame, text="Vitesse d'affichage:")
        update_label.grid(row=row, column=0, sticky=tk.W, pady=5)
        ToolTip(update_label, "Fréquence de mise à jour de l'interface")

        self.update_frequency = tk.IntVar(value=100)
        update_combo = ttk.Combobox(params_frame, textvariable=self.update_frequency,
                                   values=[50, 100, 250, 500], state="readonly", width=15)
        update_combo.grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1

        # Initialize hidden technical parameters (will be set by presets)
        self.learning_rate = tk.DoubleVar(value=0.15)
        self.discount_factor = tk.DoubleVar(value=0.98)
        self.epsilon_decay = tk.DoubleVar(value=0.996)

        # Separator
        ttk.Separator(params_frame, orient='horizontal').grid(row=row, column=0, columnspan=2,
                                                              sticky=(tk.W, tk.E), pady=10)
        row += 1

        # Preset buttons - NOW WITH CLEAR DESCRIPTIONS
        preset_label = ttk.Label(params_frame, text="🎯 Comment voulez-vous qu'il apprenne?",
                                font=('Arial', 10, 'bold'))
        preset_label.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        row += 1

        beginner_btn = ttk.Button(params_frame, text="🐰 Apprend VITE mais oublie vite",
                  command=self.preset_beginner)
        beginner_btn.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        ToolTip(beginner_btn, "Taux d'apprentissage élevé (0.2)\nBon pour: tests rapides, débutants\nPlus instable mais résultat rapide")
        row += 1

        standard_btn = ttk.Button(params_frame, text="⚖️ Apprend à vitesse NORMALE",
                  command=self.preset_standard)
        standard_btn.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        ToolTip(standard_btn, "Taux d'apprentissage moyen (0.15)\nBon pour: la plupart des cas\nÉquilibre vitesse et qualité")
        row += 1

        optimal_btn = ttk.Button(params_frame, text="🐢 Apprend LENTEMENT mais retient mieux",
                  command=self.preset_optimal)
        optimal_btn.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        ToolTip(optimal_btn, "Taux d'apprentissage bas (0.1)\nBon pour: meilleure qualité finale\nPlus lent mais plus stable")
        row += 1

        # Advanced ludic controls
        ttk.Separator(params_frame, orient=tk.HORIZONTAL).grid(row=row, column=0, columnspan=2,
                                                                sticky=(tk.W, tk.E), pady=10)
        row += 1

        # === UNIFIED REWARDS/PENALTIES SECTION ===
        # No more action weights - too complex
        # User chooses positive (reward) or negative (penalty) for each behavior

        behavior_frame = ttk.LabelFrame(params_frame, text="🎯 Récompenses & Pénalités (Optionnel)", padding="15")
        behavior_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5, padx=5)
        behavior_frame.columnconfigure(1, weight=1)
        row += 1

        ttk.Label(behavior_frame, text="Positif = récompense | Négatif = pénalité | 0 = neutre",
                 font=('Arial', 8, 'italic'), foreground='gray').grid(
            row=0, column=0, columnspan=2, pady=(0, 10))

        # Initialize all behavior variables
        self.behaviors = {
            'goal': tk.DoubleVar(value=1.0),
            'hole': tk.DoubleVar(value=0.0),
            'step': tk.DoubleVar(value=0.0),
            'wall_hit': tk.DoubleVar(value=0.0),
            'revisit': tk.DoubleVar(value=0.0),
            'loop': tk.DoubleVar(value=0.0),
            'move_closer': tk.DoubleVar(value=0.0),  # NEW: separate control
            'move_away': tk.DoubleVar(value=0.0)      # NEW: separate control
        }

        behaviors_config = [
            ('goal', "🏆 Atteindre le but:", -5.0, 10.0, 1.0, "Positif = récompense | Négatif = pénalité"),
            ('hole', "💀 Tomber dans un trou:", -5.0, 5.0, 0.0, "Positif/Négatif à votre choix"),
            ('step', "👟 Chaque étape:", -1.0, 1.0, 0.0, "Négatif = encourage rapidité | Positif = encourage exploration"),
            ('wall_hit', "🧱 Frapper un mur:", -1.0, 1.0, 0.0, "Positif/Négatif à votre choix"),
            ('revisit', "🔁 Revisiter une case:", -1.0, 1.0, 0.0, "Positif/Négatif à votre choix"),
            ('loop', "🔄 Tourner en boucle:", -1.0, 1.0, 0.0, "Positif/Négatif à votre choix"),
            ('move_closer', "📍 Se rapprocher du but:", -2.0, 2.0, 0.0, "Positif = récompense | Négatif = pénalité | Contrôle indépendant!"),
            ('move_away', "↩️ S'éloigner du but:", -2.0, 2.0, 0.0, "Positif = récompense (!) | Négatif = pénalité | Contrôle indépendant!"),
        ]

        for i, (key, label, min_val, max_val, default, tooltip) in enumerate(behaviors_config):
            ttk.Label(behavior_frame, text=label, font=('Arial', 9)).grid(
                row=i+1, column=0, sticky=tk.W, pady=3, padx=(0, 10))
            spinbox = ttk.Spinbox(behavior_frame, from_=min_val, to=max_val, increment=0.1,
                                 textvariable=self.behaviors[key], width=10, format="%.1f")
            spinbox.grid(row=i+1, column=1, sticky=(tk.W, tk.E), pady=3)
            ToolTip(spinbox, tooltip)

        # Initialize action_weights to neutral (no longer used but needed for compatibility)
        self.action_weights = {
            'left': tk.DoubleVar(value=1.0),
            'down': tk.DoubleVar(value=1.0),
            'right': tk.DoubleVar(value=1.0),
            'up': tk.DoubleVar(value=1.0)
        }

        # Reset button
        reset_btn = ttk.Button(params_frame, text="🔄 Réinitialiser Mode Ludique",
                               command=self.reset_ludic_mode)
        reset_btn.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 20), padx=5)
        ToolTip(reset_btn, "Remet tous les poids et récompenses à 1.0")

    def update_action_probabilities(self, *args):
        """Update the visual display of action probabilities."""
        weights = [self.action_weights['left'].get(),
                   self.action_weights['down'].get(),
                   self.action_weights['right'].get(),
                   self.action_weights['up'].get()]
        total = sum(weights)
        if total > 0:
            probs = [w / total * 100 for w in weights]
            self.action_prob_label.config(
                text=f"Probabilités: ← {probs[0]:.0f}% | ↓ {probs[1]:.0f}% | → {probs[2]:.0f}% | ↑ {probs[3]:.0f}%"
            )

    def reset_ludic_mode(self):
        """Reset all behavior parameters to default."""
        # Reset all behaviors to neutral (except goal = 1.0)
        self.behaviors['goal'].set(1.0)
        self.behaviors['hole'].set(0.0)
        self.behaviors['step'].set(0.0)
        self.behaviors['wall_hit'].set(0.0)
        self.behaviors['revisit'].set(0.0)
        self.behaviors['loop'].set(0.0)
        self.behaviors['move_closer'].set(0.0)
        self.behaviors['move_away'].set(0.0)
        self.log("✅ Récompenses et pénalités réinitialisées aux valeurs par défaut")

    def show_learning_curve(self):
        """Display learning curve in the embedded graph panel."""
        # Don't update if there's no data or too little data
        if not self.rewards_history or len(self.rewards_history) < 10:
            return

        try:
            import matplotlib
            matplotlib.use('TkAgg')  # Ensure TkAgg backend is used
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            plt.ioff()  # Turn off interactive mode to avoid threading issues

            # Calculate moving average
            window_size = 100
            if len(self.rewards_history) < window_size:
                window_size = max(10, len(self.rewards_history) // 10)

            moving_avg = []
            for i in range(len(self.rewards_history)):
                if i < window_size:
                    moving_avg.append(np.mean(self.rewards_history[:i+1]) * 100)
                else:
                    moving_avg.append(np.mean(self.rewards_history[i-window_size+1:i+1]) * 100)

            episodes = list(range(1, len(moving_avg) + 1))

            # Create figure and canvas only once
            if self.graph_canvas is None:
                # Create figure - ONLY ONE GRAPH NOW (simpler and clearer)
                self.graph_figure = plt.Figure(figsize=(6, 4), dpi=90)
                self.graph_ax = self.graph_figure.add_subplot(1, 1, 1)

                # Initial setup
                self.graph_ax.set_xlabel('Épisode', fontsize=11)
                self.graph_ax.set_ylabel('Taux de Réussite (%)', fontsize=11)
                self.graph_ax.grid(True, alpha=0.3, linestyle='--')
                self.graph_ax.set_ylim(0, 100)
                self.graph_ax.tick_params(labelsize=9)

                self.graph_figure.tight_layout(pad=1.5)

                # Embed in tkinter frame with fixed size
                self.graph_canvas = FigureCanvasTkAgg(self.graph_figure, master=self.graph_frame_widget)
                canvas_widget = self.graph_canvas.get_tk_widget()
                canvas_widget.pack(fill=tk.BOTH, expand=True)
                # Set minimum size to prevent jiggling
                canvas_widget.configure(width=600, height=400)

                # Initialize line and fill objects
                self.graph_line, = self.graph_ax.plot([], [], color='#2ecc71', linewidth=2.5)
                self.graph_fill = None

            # Update data (much faster than recreating everything)
            self.graph_line.set_data(episodes, moving_avg)

            # Update fill
            if self.graph_fill:
                self.graph_fill.remove()
            self.graph_fill = self.graph_ax.fill_between(episodes, 0, moving_avg, alpha=0.3, color='#2ecc71')

            # Update title and limits
            self.graph_ax.set_title(f'Évolution du Taux de Réussite (moyenne sur {window_size} épisodes)',
                        fontsize=12, fontweight='bold')
            self.graph_ax.relim()
            self.graph_ax.autoscale_view(scalex=True, scaley=False)  # Only autoscale X, keep Y at 0-100

            # Redraw only the changed parts (faster)
            self.graph_canvas.draw_idle()

            # Only log every 5 updates to reduce console spam
            if len(self.rewards_history) % 500 == 0:
                self.log("📊 Graphique d'apprentissage mis à jour!")

        except ImportError:
            self.log("⚠️  Matplotlib n'est pas installé. Installez-le avec: pip install matplotlib")
        except Exception as e:
            self.log(f"❌ Erreur lors de la création du graphique: {str(e)}")
            import traceback
            self.log(traceback.format_exc())

    def setup_visualization_panel(self, parent):
        """Setup visualization and control panel."""

        viz_frame = ttk.Frame(parent)
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        viz_frame.rowconfigure(1, weight=1)
        viz_frame.rowconfigure(2, weight=1)
        viz_frame.columnconfigure(0, weight=1)
        viz_frame.columnconfigure(1, weight=1)  # Add second column for graph

        # Control buttons (span 2 columns)
        control_frame = ttk.LabelFrame(viz_frame, text="🎮 Contrôles", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        self.train_button = ttk.Button(control_frame, text="🚀 Démarrer l'Entraînement",
                                       command=self.start_training)
        self.train_button.grid(row=0, column=0, padx=5, pady=5)

        self.stop_button = ttk.Button(control_frame, text="⏹️ Arrêter",
                                      command=self.stop_training, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=5, pady=5)

        self.demo_button = ttk.Button(control_frame, text="👁️ Démo dans la Grille",
                                      command=self.show_grid_demo, state=tk.DISABLED)
        self.demo_button.grid(row=0, column=2, padx=5, pady=5)
        ToolTip(self.demo_button, "Animation de l'agent dans la grille ci-dessous")

        self.stop_demo_button = ttk.Button(control_frame, text="⏹️ Stop Démo",
                                           command=self.stop_demo, state=tk.DISABLED)
        self.stop_demo_button.grid(row=0, column=3, padx=5, pady=5)
        ToolTip(self.stop_demo_button, "Arrêter la démo en cours")

        self.pygame_button = ttk.Button(control_frame, text="🎮 Démo Pygame",
                                        command=self.show_pygame_demo, state=tk.DISABLED)
        self.pygame_button.grid(row=0, column=4, padx=5, pady=5)
        ToolTip(self.pygame_button, "Ouvre une fenêtre de jeu pygame")

        # Demo speed control (real-time)
        speed_label = ttk.Label(control_frame, text="Vitesse:")
        speed_label.grid(row=0, column=5, padx=(20, 5))
        ToolTip(speed_label, "Vitesse de la démo (modifiable en temps réel)\n0.05 = très rapide\n1.0 = lent")

        self.demo_speed = tk.DoubleVar(value=0.3)
        speed_scale = ttk.Scale(control_frame, from_=0.05, to=1.0, variable=self.demo_speed,
                               orient=tk.HORIZONTAL, length=100)
        speed_scale.grid(row=0, column=6, padx=5)

        self.speed_label_value = ttk.Label(control_frame, text="0.3s")
        self.speed_label_value.grid(row=0, column=7, padx=(0, 10))
        self.demo_speed.trace_add("write", lambda *args: self.speed_label_value.config(text=f"{self.demo_speed.get():.2f}s"))

        # Max steps control
        steps_label = ttk.Label(control_frame, text="Max steps:")
        steps_label.grid(row=0, column=8, padx=(10, 5))
        ToolTip(steps_label, "Nombre maximum de mouvements avant échec")

        self.demo_max_steps = tk.IntVar(value=50)
        steps_spinbox = ttk.Spinbox(control_frame, from_=10, to=200, increment=10,
                                    textvariable=self.demo_max_steps, width=5)
        steps_spinbox.grid(row=0, column=9, padx=5)

        # Live visualization always enabled (no toggle)
        self.show_live_grid = tk.BooleanVar(value=True)

        # Statistics panel
        stats_frame = ttk.LabelFrame(viz_frame, text="📊 Statistiques en Temps Réel", padding="10")
        stats_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        stats_frame.columnconfigure(0, weight=1)
        stats_frame.columnconfigure(1, weight=1)
        stats_frame.columnconfigure(2, weight=1)

        # Stats display in 3 columns
        stats_container = ttk.Frame(stats_frame)
        stats_container.pack(fill=tk.BOTH, expand=True)

        # Configure 3 columns with equal weight
        for i in range(6):
            stats_container.columnconfigure(i, weight=1)

        # COLUMN 1 - Progression & Epsilon
        # Progress
        ttk.Label(stats_container, text="Progression:", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, sticky=tk.W, pady=5, padx=(0, 5))
        self.progress_var = tk.StringVar(value="0 / 0 épisodes")
        ttk.Label(stats_container, textvariable=self.progress_var).grid(
            row=0, column=1, sticky=tk.W, padx=5, pady=5)

        # Epsilon
        eps_label = ttk.Label(stats_container, text="Epsilon:", font=('Arial', 10, 'bold'))
        eps_label.grid(row=1, column=0, sticky=tk.W, pady=5, padx=(0, 5))
        ToolTip(eps_label, "Taux d'exploration actuel\n1.0 = 100% exploration\n0.01 = 99% exploitation")
        self.epsilon_var = tk.StringVar(value="1.000")
        ttk.Label(stats_container, textvariable=self.epsilon_var, font=('Arial', 12)).grid(
            row=1, column=1, sticky=tk.W, padx=5, pady=5)

        # COLUMN 2 - Win rate & Reward
        # Win rate
        win_label = ttk.Label(stats_container, text="Taux réussite:", font=('Arial', 10, 'bold'))
        win_label.grid(row=0, column=2, sticky=tk.W, pady=5, padx=(15, 5))
        ToolTip(win_label, "% de victoires sur les 100 derniers épisodes\n>70% = très bon (sans glace)\n>60% = très bon (avec glace)")
        self.winrate_var = tk.StringVar(value="0.0%")
        self.winrate_label = ttk.Label(stats_container, textvariable=self.winrate_var,
                 font=('Arial', 14, 'bold'), foreground='green')
        self.winrate_label.grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)

        # Average reward
        ttk.Label(stats_container, text="Récompense moy.:",
                 font=('Arial', 10, 'bold')).grid(row=1, column=2, sticky=tk.W, pady=5, padx=(15, 5))
        self.reward_var = tk.StringVar(value="0.000")
        ttk.Label(stats_container, textvariable=self.reward_var, font=('Arial', 12)).grid(
            row=1, column=3, sticky=tk.W, padx=5, pady=5)

        # COLUMN 3 - Time
        # Time elapsed
        ttk.Label(stats_container, text="Temps écoulé:", font=('Arial', 10, 'bold')).grid(
            row=0, column=4, sticky=tk.W, pady=5, padx=(15, 5))
        self.time_var = tk.StringVar(value="0s")
        ttk.Label(stats_container, textvariable=self.time_var).grid(
            row=0, column=5, sticky=tk.W, padx=5, pady=5)

        # ETA
        ttk.Label(stats_container, text="Temps restant:", font=('Arial', 10, 'bold')).grid(
            row=1, column=4, sticky=tk.W, pady=5, padx=(15, 5))
        self.eta_var = tk.StringVar(value="--")
        ttk.Label(stats_container, textvariable=self.eta_var).grid(
            row=1, column=5, sticky=tk.W, padx=5, pady=5)

        # Progress bar (full width below stats)
        self.progress_bar = ttk.Progressbar(stats_container, mode='determinate', length=400)
        self.progress_bar.grid(row=2, column=0, columnspan=6, sticky=(tk.W, tk.E), pady=(10, 5), padx=5)

        # Grid visualization (left side, smaller)
        grid_frame = ttk.LabelFrame(viz_frame, text="🎮 Visualisation de la Grille", padding="10")
        grid_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))

        self.grid_canvas = tk.Canvas(grid_frame, bg='#2c3e50')
        self.grid_canvas.pack(fill=tk.BOTH, expand=True)

        # Redraw grid when canvas is resized
        self.grid_canvas.bind("<Configure>", self.on_canvas_resize)

        # Graph visualization (right side)
        graph_frame = ttk.LabelFrame(viz_frame, text="📈 Courbe d'Apprentissage", padding="10")
        graph_frame.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))

        # Create matplotlib canvas
        self.graph_canvas = None
        self.graph_figure = None
        self.graph_frame_widget = graph_frame  # Store reference

        # Legend
        legend_frame = ttk.Frame(grid_frame)
        legend_frame.pack(fill=tk.X, pady=5)

        ttk.Label(legend_frame, text="🟦 Start", foreground='blue').pack(side=tk.LEFT, padx=5)
        ttk.Label(legend_frame, text="🟩 Goal", foreground='green').pack(side=tk.LEFT, padx=5)
        ttk.Label(legend_frame, text="⬜ Safe", foreground='gray').pack(side=tk.LEFT, padx=5)
        ttk.Label(legend_frame, text="⬛ Hole", foreground='black').pack(side=tk.LEFT, padx=5)
        ttk.Label(legend_frame, text="🔴 Agent", foreground='red').pack(side=tk.LEFT, padx=5)

    def setup_console_panel(self, parent):
        """Setup console output panel."""

        console_frame = ttk.LabelFrame(parent, text="📝 Journal d'Entraînement", padding="10")
        console_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        console_frame.rowconfigure(0, weight=1)
        console_frame.columnconfigure(0, weight=1)

        self.console = scrolledtext.ScrolledText(console_frame, height=8, wrap=tk.WORD,
                                                 font=('Courier', 9))
        self.console.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.log("╔════════════════════════════════════════════════════════════════════╗")
        self.log("║  Bienvenue dans le FrozenLake Q-Learning Lab! 🧊🤖                 ║")
        self.log("╚════════════════════════════════════════════════════════════════════╝")
        self.log("")
        self.log("💡 DÉMARRAGE RAPIDE:")
        self.log("   1. Cliquez sur '🎓 Débutant' pour charger des paramètres qui marchent")
        self.log("   2. Cliquez sur '🚀 Démarrer l'Entraînement'")
        self.log("   3. Regardez le taux de réussite monter!")
        self.log("   4. Cliquez sur '👁️ Voir Démo Visuelle' pour voir l'agent jouer")
        self.log("")
        self.log("❓ Cliquez sur '❓ Guide d'Utilisation' pour comprendre les paramètres!")
        self.log("-" * 72)

    def log(self, message):
        """Add message to console."""
        self.console.insert(tk.END, message + "\n")
        self.console.see(tk.END)

    def on_canvas_resize(self, event):
        """Handle canvas resize event - redraw grid to fit new size."""
        # Only redraw if we have a valid environment and the size is reasonable
        if self.env and event.width > 50 and event.height > 50:
            # Small delay to avoid excessive redraws during resizing
            if hasattr(self, '_resize_job'):
                self.root.after_cancel(self._resize_job)
            self._resize_job = self.root.after(100, lambda: self.draw_grid())

    def draw_grid(self, agent_pos=None, path=None, show_values=False):
        """Draw the FrozenLake grid with agent position."""
        if not self.env:
            # Draw placeholder
            self.grid_canvas.delete("all")
            width = self.grid_canvas.winfo_width()
            height = self.grid_canvas.winfo_height()
            if width > 10 and height > 10:
                self.grid_canvas.create_text(width/2, height/2,
                    text="La grille apparaîtra après l'entraînement",
                    fill='white', font=('Arial', 12))
            return

        self.grid_canvas.delete("all")
        width = self.grid_canvas.winfo_width()
        height = self.grid_canvas.winfo_height()

        if width < 50 or height < 50:
            return

        # Get grid description
        if self.custom_map:
            desc = self.custom_map
        else:
            map_name = self.map_size.get()
            if map_name == "8x8":
                desc = ["SFFFFFFF", "FFFFFFFF", "FFFHFFFF", "FFFFFHFF",
                        "FFFHFFFF", "FHHFFFHF", "FHFFHFHF", "FFFHFFFG"]
            else:  # 4x4
                desc = ["SFFF", "FHFH", "FFFH", "HFFG"]

        grid_size = len(desc)
        margin = 20
        available_width = width - 2 * margin
        available_height = height - 2 * margin
        cell_size = min(available_width / grid_size, available_height / grid_size)

        # Center the grid
        start_x = margin + (available_width - cell_size * grid_size) / 2
        start_y = margin + (available_height - cell_size * grid_size) / 2

        # Draw cells
        for row in range(grid_size):
            for col in range(grid_size):
                x1 = start_x + col * cell_size
                y1 = start_y + row * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size

                cell = desc[row][col]

                # Cell colors
                if cell == 'S':
                    color = '#3498db'  # Blue
                    text_color = 'white'
                    label = 'S'
                elif cell == 'G':
                    color = '#2ecc71'  # Green
                    text_color = 'white'
                    label = 'G'
                elif cell == 'H':
                    color = '#34495e'  # Dark gray
                    text_color = 'white'
                    label = 'H'
                else:  # F
                    color = '#ecf0f1'  # Light gray
                    text_color = 'black'
                    label = ''

                # Draw cell
                self.grid_canvas.create_rectangle(x1, y1, x2, y2,
                    fill=color, outline='#2c3e50', width=2)

                # Draw label
                if label:
                    self.grid_canvas.create_text(x1 + cell_size/2, y1 + cell_size/2,
                        text=label, font=('Arial', int(cell_size/3), 'bold'),
                        fill=text_color)

                # Show Q-values if requested
                if show_values and self.agent and cell == 'F':
                    state = row * grid_size + col
                    max_q = np.max(self.agent.q_table[state])
                    if max_q > 0:
                        # Draw best action arrow
                        best_action = np.argmax(self.agent.q_table[state])
                        cx = x1 + cell_size/2
                        cy = y1 + cell_size/2
                        arrow_len = cell_size/4

                        arrows = {
                            0: (-arrow_len, 0),  # Left
                            1: (0, arrow_len),   # Down
                            2: (arrow_len, 0),   # Right
                            3: (0, -arrow_len)   # Up
                        }
                        dx, dy = arrows[best_action]
                        self.grid_canvas.create_line(cx, cy, cx+dx, cy+dy,
                            arrow=tk.LAST, fill='#e74c3c', width=2)

        # Draw agent
        if agent_pos is not None:
            row = agent_pos // grid_size
            col = agent_pos % grid_size
            x = start_x + col * cell_size + cell_size/2
            y = start_y + row * cell_size + cell_size/2
            radius = cell_size/3

            # Agent as a red circle
            self.grid_canvas.create_oval(x-radius, y-radius, x+radius, y+radius,
                fill='#e74c3c', outline='#c0392b', width=3)

        # Draw path if provided
        if path:
            for i in range(len(path)-1):
                row1 = path[i] // grid_size
                col1 = path[i] % grid_size
                row2 = path[i+1] // grid_size
                col2 = path[i+1] % grid_size

                x1 = start_x + col1 * cell_size + cell_size/2
                y1 = start_y + row1 * cell_size + cell_size/2
                x2 = start_x + col2 * cell_size + cell_size/2
                y2 = start_y + row2 * cell_size + cell_size/2

                self.grid_canvas.create_line(x1, y1, x2, y2,
                    fill='#f39c12', width=3, arrow=tk.LAST)

    def on_map_size_change(self, event=None):
        """Handle map size change."""
        # Clear custom map if switching away from custom
        if self.map_size.get() != "Personnalisée" and self.custom_map is not None:
            self.custom_map = None
            self.custom_map_status.config(text="")
            self.log("ℹ️ Carte personnalisée désactivée")

    def generate_random_map(self):
        """Generate a random map."""
        try:
            dialog = tk.Toplevel(self.root)
            dialog.title("Générer Carte Aléatoire")
            dialog.geometry("400x320")
            dialog.resizable(True, True)  # Allow resizing
            dialog.transient(self.root)
            dialog.grab_set()

            ttk.Label(dialog, text="Taille de la carte:", font=('Arial', 10, 'bold')).pack(pady=10)
            size_var = tk.IntVar(value=8)
            size_scale = ttk.Scale(dialog, from_=4, to=12, variable=size_var, orient=tk.HORIZONTAL, length=250)
            size_scale.pack(pady=5)
            size_label = ttk.Label(dialog, text="8x8")
            size_label.pack()
            size_var.trace_add('write', lambda *args: size_label.config(text=f"{size_var.get()}x{size_var.get()}"))

            ttk.Label(dialog, text="Probabilité de trous:", font=('Arial', 10, 'bold')).pack(pady=10)
            prob_var = tk.DoubleVar(value=0.2)
            prob_scale = ttk.Scale(dialog, from_=0.1, to=0.4, variable=prob_var, orient=tk.HORIZONTAL, length=250)
            prob_scale.pack(pady=5)
            prob_label = ttk.Label(dialog, text="0.20")
            prob_label.pack()
            prob_var.trace_add('write', lambda *args: prob_label.config(text=f"{prob_var.get():.2f}"))

            def on_generate():
                size = size_var.get()
                prob = prob_var.get()
                self.log(f"\n✏️ Génération d'une carte {size}x{size} (difficulté: {int(prob*100)}% de trous)...")

                # Track attempts for logging
                attempts = 0
                max_attempts = 100

                while attempts < max_attempts:
                    attempts += 1
                    candidate_map = generate_random_map(size, prob, max_attempts=1)

                    # Verify path exists
                    grid = [list(row) for row in candidate_map]
                    if has_valid_path(grid):
                        self.custom_map = candidate_map
                        self.log(f"✅ Carte valide générée en {attempts} tentative{'s' if attempts > 1 else ''}")
                        self.log(f"   Chemin garanti de S (départ) à G (arrivée)")
                        for i, row in enumerate(self.custom_map):
                            self.log(f"  {row}")

                        # Update UI - SET TO CUSTOM AND SHOW STATUS
                        self.map_size.set("Personnalisée")
                        self.custom_map_status.config(text=f"✓ Carte {size}x{size} créée!")

                        dialog.destroy()
                        return

                # Fallback if max attempts reached (shouldn't happen often)
                self.log(f"⚠️  Difficulté à générer avec ces paramètres")
                self.log(f"   Génération d'une carte garantie avec chemin...")
                self.custom_map = generate_random_map(size, prob, max_attempts=100)
                self.log(f"✅ Carte de secours générée (chemin garanti)")
                for i, row in enumerate(self.custom_map):
                    self.log(f"  {row}")

                # Update UI
                self.map_size.set("Personnalisée")
                self.custom_map_status.config(text=f"✓ Carte {size}x{size} créée!")

                dialog.destroy()

            ttk.Button(dialog, text="Générer", command=on_generate).pack(pady=15)

        except Exception as e:
            self.log(f"❌ Erreur génération carte: {e}")

    def preset_beginner(self):
        """Load beginner preset - GUARANTEED TO WORK!"""
        self.learning_rate.set(0.2)
        self.discount_factor.set(0.95)
        self.epsilon_decay.set(0.997)
        self.episodes.set(5000)
        self.update_frequency.set(50)
        self.log("\n🎓 PRÉRÉGLAGE DÉBUTANT chargé")
        self.log("   • α=0.2, γ=0.95, ε decay=0.997")
        self.log("   • 5000 épisodes (rapide)")
        self.log("   ➡️  Cliquez sur '🚀 Démarrer l'Entraînement'")

    def preset_standard(self):
        """Load standard preset."""
        self.learning_rate.set(0.15)
        self.discount_factor.set(0.98)
        self.epsilon_decay.set(0.996)
        self.episodes.set(10000)
        self.update_frequency.set(100)
        self.log("\n⚡ PRÉRÉGLAGE STANDARD chargé")
        self.log("   • α=0.15, γ=0.98, ε decay=0.996")
        self.log("   • 10000 épisodes")

    def preset_optimal(self):
        """Load optimal preset."""
        self.learning_rate.set(0.1)
        self.discount_factor.set(0.99)
        self.epsilon_decay.set(0.9965)
        self.episodes.set(15000)
        self.update_frequency.set(100)
        self.log("\n🎯 PRÉRÉGLAGE OPTIMAL chargé")
        self.log("   • α=0.1, γ=0.99, ε decay=0.9965")
        self.log("   • 15000 épisodes (plus long)")

    def start_training(self):
        """Start training in a separate thread."""
        if self.is_training:
            return

        self.is_training = True
        self.rewards_history = []
        self.last_graph_update = 0
        self.train_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.demo_button.config(state=tk.DISABLED)

        self.training_thread = threading.Thread(target=self.train_agent, daemon=True)
        self.training_thread.start()

    def stop_training(self):
        """Stop training."""
        self.is_training = False
        self.log("⏹️  Arrêt de l'entraînement demandé...")

    def train_agent(self):
        """Train the agent (runs in separate thread - optimized for speed)."""
        try:
            # Get parameters
            map_name = self.map_size.get()
            is_slippery = self.is_slippery.get()
            lr = self.learning_rate.get()
            gamma = self.discount_factor.get()
            decay = self.epsilon_decay.get()
            episodes = self.episodes.get()
            update_freq = self.update_frequency.get()

            # Create environment
            if map_name == "Personnalisée":
                if self.custom_map:
                    self.env = gym.make("FrozenLake-v1", desc=self.custom_map, is_slippery=is_slippery)
                else:
                    raise ValueError("Vous devez d'abord créer une carte personnalisée avec le bouton '✏️ Créer ma propre carte'")
            else:
                self.env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery)

            # Get ludic parameters
            action_biases = [
                self.action_weights['left'].get(),
                self.action_weights['down'].get(),
                self.action_weights['right'].get(),
                self.action_weights['up'].get()
            ]

            # Get unified behaviors (can be positive or negative)
            reward_shaping = {
                'goal': self.behaviors['goal'].get(),
                'hole': self.behaviors['hole'].get(),
                'step': self.behaviors['step'].get()
            }
            advanced_penalties = {
                'wall_hit': self.behaviors['wall_hit'].get(),
                'revisit': self.behaviors['revisit'].get(),
                'loop': self.behaviors['loop'].get(),
                # Combine move_closer and move_away into single distance parameter
                # Positive move_closer + negative move_away = standard behavior
                'distance': self.behaviors['move_closer'].get() + self.behaviors['move_away'].get()
            }

            # Create agent
            self.agent = QLearningAgent(
                self.env,
                learning_rate=lr,
                discount_factor=gamma,
                epsilon_decay=decay,
                action_biases=action_biases,
                reward_shaping=reward_shaping,
                advanced_penalties=advanced_penalties
            )

            self.message_queue.put(("log", f"\n{'='*72}"))
            self.message_queue.put(("log", f"🚀 DÉMARRAGE DE L'ENTRAÎNEMENT (Mode optimisé)"))
            map_info = "Personnalisée" if self.custom_map else map_name
            self.message_queue.put(("log", f"   Carte: {map_info} | Glissant: {'Oui' if is_slippery else 'Non'}"))
            self.message_queue.put(("log", f"   α={lr:.2f} | γ={gamma:.2f} | decay={decay:.3f} | épisodes={episodes}"))

            # Show ludic parameters if non-default
            if action_biases != [1.0, 1.0, 1.0, 1.0]:
                self.message_queue.put(("log", f"   🎮 Poids actions: ←{action_biases[0]:.1f} ↓{action_biases[1]:.1f} →{action_biases[2]:.1f} ↑{action_biases[3]:.1f}"))
            if reward_shaping != {'goal': 1.0, 'hole': 0.0, 'step': 0.0}:
                self.message_queue.put(("log", f"   🎁 Récompenses: Goal={reward_shaping['goal']:.1f} Hole={reward_shaping['hole']:.1f} Step={reward_shaping['step']:.2f}"))
            if advanced_penalties != {'wall_hit': 0.0, 'revisit': 0.0, 'loop': 0.0, 'distance': 0.0}:
                penalties_str = []
                if advanced_penalties['wall_hit'] != 0.0:
                    penalties_str.append(f"Mur={advanced_penalties['wall_hit']:.2f}")
                if advanced_penalties['revisit'] != 0.0:
                    penalties_str.append(f"Revisite={advanced_penalties['revisit']:.2f}")
                if advanced_penalties['loop'] != 0.0:
                    penalties_str.append(f"Boucle={advanced_penalties['loop']:.2f}")
                if advanced_penalties['distance'] != 0.0:
                    penalties_str.append(f"Distance={advanced_penalties['distance']:.2f}")
                if penalties_str:
                    self.message_queue.put(("log", f"   ⚙️ Pénalités: {' | '.join(penalties_str)}"))

            self.message_queue.put(("log", f"{'='*72}"))

            # Draw initial grid
            self.message_queue.put(("draw_grid", 0))

            start_time = time.time()
            last_update_time = start_time
            last_grid_update = start_time

            # Batch episodes for better performance
            batch_size = min(update_freq, 100)

            for episode in range(episodes):
                if not self.is_training:
                    break

                # Run episode
                state, info = self.env.reset()
                self.agent.reset_episode_tracking()  # Reset tracking for new episode
                done = False
                episode_reward = 0
                steps = 0
                max_steps = 100  # Prevent infinite loops

                while not done and steps < max_steps:
                    action = self.agent.choose_action(state)
                    next_state, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated

                    self.agent.update_q_table(state, action, reward, next_state, done)

                    # Update grid visualization during training (throttled for better visibility)
                    current_time = time.time()
                    if self.show_live_grid.get() and (current_time - last_grid_update) > 0.15:
                        # Send position, action taken, and reward for better visualization
                        self.message_queue.put(("draw_grid_training", {
                            'position': next_state,
                            'action': action,
                            'reward': reward,
                            'episode': episode + 1
                        }))
                        last_grid_update = current_time

                    state = next_state
                    episode_reward += reward
                    steps += 1

                self.rewards_history.append(episode_reward)
                self.agent.decay_epsilon()

                # Update UI less frequently and non-blocking
                current_time = time.time()
                if (episode + 1) % update_freq == 0 or (current_time - last_update_time) > 0.5:
                    last_update_time = current_time

                    # Calculate stats
                    recent_rewards = self.rewards_history[-100:] if len(self.rewards_history) >= 100 else self.rewards_history
                    avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                    win_rate = (np.sum(recent_rewards) / len(recent_rewards) * 100) if recent_rewards else 0
                    elapsed = current_time - start_time

                    # Calculate ETA
                    episodes_per_sec = (episode + 1) / elapsed if elapsed > 0 else 0
                    remaining_episodes = episodes - (episode + 1)
                    eta_seconds = remaining_episodes / episodes_per_sec if episodes_per_sec > 0 else 0

                    # Send all updates at once (more efficient)
                    self.message_queue.put(("batch_update", {
                        "progress": (episode + 1, episodes),
                        "epsilon": self.agent.epsilon,
                        "winrate": win_rate,
                        "reward": avg_reward,
                        "time": elapsed,
                        "eta": eta_seconds
                    }))

                    if (episode + 1) % 1000 == 0:
                        self.message_queue.put(("log",
                            f"Épisode {episode + 1:5d}/{episodes} | "
                            f"Réussite: {win_rate:5.1f}% | "
                            f"ε={self.agent.epsilon:.3f} | "
                            f"Vitesse: {episodes_per_sec:.1f} ep/s"))

            # Training complete
            elapsed = time.time() - start_time
            final_rewards = self.rewards_history[-100:] if len(self.rewards_history) >= 100 else self.rewards_history
            final_winrate = (np.sum(final_rewards) / len(final_rewards) * 100) if final_rewards else 0
            avg_speed = episodes / elapsed if elapsed > 0 else 0

            # Calculate detailed statistics
            total_successes = int(np.sum(self.rewards_history))
            total_episodes = len(self.rewards_history)
            overall_winrate = (total_successes / total_episodes * 100) if total_episodes > 0 else 0

            # Calculate best performance window
            best_winrate = 0
            best_episode = 0
            window_size = min(100, total_episodes)
            for i in range(window_size, total_episodes + 1):
                window = self.rewards_history[i-window_size:i]
                winrate = (np.sum(window) / len(window) * 100)
                if winrate > best_winrate:
                    best_winrate = winrate
                    best_episode = i

            self.message_queue.put(("log", f"\n{'='*72}"))
            if self.is_training:
                self.message_queue.put(("log", f"✅ ENTRAÎNEMENT TERMINÉ!"))
                self.message_queue.put(("log", f"{'='*72}"))
                self.message_queue.put(("log", f"📊 STATISTIQUES FINALES:"))
                self.message_queue.put(("log", f""))
                self.message_queue.put(("log", f"   ⏱️  Temps total: {elapsed:.1f}s ({avg_speed:.1f} épisodes/sec)"))
                self.message_queue.put(("log", f"   🎯 Épisodes: {total_episodes}"))
                self.message_queue.put(("log", f"   ✅ Victoires totales: {total_successes}/{total_episodes}"))
                self.message_queue.put(("log", f"   📈 Taux global: {overall_winrate:.1f}%"))
                self.message_queue.put(("log", f"   📊 Taux final (100 derniers): {final_winrate:.1f}%"))
                self.message_queue.put(("log", f"   🏆 Meilleur taux atteint: {best_winrate:.1f}% (épisode {best_episode})"))
                self.message_queue.put(("log", f"   🎲 Epsilon final: {self.agent.epsilon:.4f}"))
                self.message_queue.put(("log", f""))

                # Evaluation message
                if final_winrate > 80:
                    self.message_queue.put(("log", f"   🌟 EXCELLENT! L'agent a très bien appris!"))
                elif final_winrate > 60:
                    self.message_queue.put(("log", f"   ✅ TRÈS BIEN! L'agent performe bien!"))
                elif final_winrate > 40:
                    self.message_queue.put(("log", f"   👍 BIEN! Peut-être essayer plus d'épisodes?"))
                else:
                    self.message_queue.put(("log", f"   ⚠️  Apprentissage insuffisant. Essayez:"))
                    self.message_queue.put(("log", f"      - Augmenter les épisodes à 15000+"))
                    self.message_queue.put(("log", f"      - Augmenter le taux d'apprentissage"))
                    self.message_queue.put(("log", f"      - Ou utilisez le préréglage DÉBUTANT"))

                self.message_queue.put(("log", f"\n   ➡️  Cliquez sur '👁️ Démo dans la Grille' pour voir l'agent!"))

                # Generate learning curve graph
                self.message_queue.put(("show_graph", None))
            else:
                self.message_queue.put(("log", f"⏹️  Entraînement arrêté par l'utilisateur"))
                self.message_queue.put(("log", f"   Temps: {elapsed:.1f}s | Réussite: {final_winrate:.1f}%"))

            self.message_queue.put(("log", f"{'='*72}\n"))
            self.message_queue.put(("complete", None))

        except Exception as e:
            self.message_queue.put(("log", f"❌ Erreur: {str(e)}"))
            import traceback
            self.message_queue.put(("log", traceback.format_exc()))
            self.message_queue.put(("complete", None))

    def show_grid_demo(self):
        """Show grid demo with the trained agent (animated in GUI)."""
        if self.agent is None:
            self.log("❌ Aucun agent entraîné! Lancez d'abord l'entraînement.")
            return

        self.log("\n🎬 Lancement de la démo dans la grille...")
        self.log("   (Ajustez la vitesse en temps réel avec le curseur)")
        self.is_demo_running = True
        self.demo_button.config(state=tk.DISABLED)
        self.stop_demo_button.config(state=tk.NORMAL)
        threading.Thread(target=self.run_grid_demo, daemon=True).start()

    def stop_demo(self):
        """Stop the running demo."""
        self.is_demo_running = False
        self.log("⏹️ Arrêt de la démo demandé...")

    def show_pygame_demo(self):
        """Show pygame demo with the trained agent (game window)."""
        if self.agent is None:
            self.log("❌ Aucun agent entraîné! Lancez d'abord l'entraînement.")
            return

        self.log("\n🎮 Lancement de la démo Pygame (fenêtre séparée)...")
        self.pygame_button.config(state=tk.DISABLED)

        # Use subprocess instead of threading to avoid macOS crash
        threading.Thread(target=self.run_pygame_demo_subprocess, daemon=True).start()

    def run_pygame_demo_subprocess(self):
        """Run pygame demo in a separate process (macOS compatible)."""
        import subprocess
        import pickle
        import tempfile
        import os

        try:
            # Create temporary files for Q-table and custom map
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
                q_table_path = f.name
                pickle.dump(self.agent.q_table, f)

            custom_map_path = None
            if self.custom_map:
                with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
                    custom_map_path = f.name
                    pickle.dump(self.custom_map, f)

            # Get parameters
            map_name = self.map_size.get()
            # If using custom map, pass a dummy map_name (it will be ignored anyway)
            if map_name == "Personnalisée":
                map_name = "4x4"  # Dummy value, custom_map_path will be used instead
            is_slippery = str(self.is_slippery.get())
            delay = str(self.demo_speed.get())

            # Build command
            python_cmd = sys.executable
            script_path = os.path.join(os.path.dirname(__file__), 'pygame_demo.py')

            cmd = [python_cmd, script_path, q_table_path, map_name, is_slippery, delay]
            if custom_map_path:
                cmd.append(custom_map_path)

            # Run subprocess
            self.message_queue.put(("log", "🎮 Ouverture de la fenêtre pygame..."))
            result = subprocess.run(cmd, capture_output=True, text=True)

            # Log output
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    self.message_queue.put(("log", line))

            if result.returncode != 0 and result.stderr:
                self.message_queue.put(("log", f"❌ Erreur: {result.stderr}"))

            # Clean up temp files
            os.unlink(q_table_path)
            if custom_map_path:
                os.unlink(custom_map_path)

            self.message_queue.put(("log", "🏁 Démo pygame terminée!"))
            self.message_queue.put(("pygame_complete", None))

        except Exception as e:
            self.message_queue.put(("log", f"❌ Erreur pygame: {str(e)}"))
            self.message_queue.put(("pygame_complete", None))

    def show_result_screen(self, screen, success, episode, total_episodes, steps):
        """Display victory/defeat overlay on pygame window."""
        import pygame

        # Get screen dimensions
        width, height = screen.get_size()

        # Create semi-transparent overlay
        overlay = pygame.Surface((width, height))
        overlay.set_alpha(220)
        overlay.fill((0, 0, 0))
        screen.blit(overlay, (0, 0))

        # Fonts
        try:
            title_font = pygame.font.SysFont('arial', 72, bold=True)
            subtitle_font = pygame.font.SysFont('arial', 36, bold=True)
            info_font = pygame.font.SysFont('arial', 28)
            small_font = pygame.font.SysFont('arial', 24)
        except:
            title_font = pygame.font.Font(None, 72)
            subtitle_font = pygame.font.Font(None, 36)
            info_font = pygame.font.Font(None, 28)
            small_font = pygame.font.Font(None, 24)

        if success:
            # Victory screen - bright and celebratory
            bg_color = (34, 139, 34)  # Forest green
            border_color = (50, 205, 50)  # Lime green
            title_color = (255, 255, 100)  # Bright yellow

            title_text = "VICTOIRE!"
            emoji = "🎉"
            subtitle_text = f"But atteint en {steps} étape{'s' if steps > 1 else ''}!"
        else:
            # Defeat screen - clear but not too aggressive
            bg_color = (139, 0, 0)  # Dark red
            border_color = (220, 20, 60)  # Crimson
            title_color = (255, 200, 200)  # Light pink

            title_text = "DÉFAITE"
            emoji = "💀"
            subtitle_text = "Tombé dans un trou!"

        # Calculate box dimensions
        box_width = width - 120
        box_height = 320
        box_x = 60
        box_y = height // 2 - box_height // 2

        # Draw outer glow effect
        for i in range(5):
            glow_alpha = 30 - (i * 5)
            glow_surface = pygame.Surface((width, height), pygame.SRCALPHA)
            pygame.draw.rect(glow_surface, (*border_color, glow_alpha),
                           (box_x - i*2, box_y - i*2, box_width + i*4, box_height + i*4),
                           border_radius=25)
            screen.blit(glow_surface, (0, 0))

        # Draw main background box with gradient effect
        pygame.draw.rect(screen, bg_color, (box_x, box_y, box_width, box_height), border_radius=20)

        # Draw decorative border (thick)
        pygame.draw.rect(screen, border_color, (box_x, box_y, box_width, box_height), 5, border_radius=20)

        # Draw inner highlight
        pygame.draw.rect(screen, (255, 255, 255, 60),
                        (box_x + 10, box_y + 10, box_width - 20, box_height - 20),
                        2, border_radius=15)

        # Render emoji (larger)
        emoji_font = pygame.font.SysFont('arial', 100)
        emoji_render = emoji_font.render(emoji, True, (255, 255, 255))
        emoji_rect = emoji_render.get_rect(center=(width // 2, box_y + 80))
        screen.blit(emoji_render, emoji_rect)

        # Render title with shadow
        title_render = title_font.render(title_text, True, (0, 0, 0))  # Shadow
        title_rect = title_render.get_rect(center=(width // 2 + 2, box_y + 162))
        screen.blit(title_render, title_rect)

        title_render = title_font.render(title_text, True, title_color)  # Main text
        title_rect = title_render.get_rect(center=(width // 2, box_y + 160))
        screen.blit(title_render, title_rect)

        # Render subtitle
        subtitle_render = subtitle_font.render(subtitle_text, True, (255, 255, 255))
        subtitle_rect = subtitle_render.get_rect(center=(width // 2, box_y + 220))
        screen.blit(subtitle_render, subtitle_rect)

        # Draw separator line
        line_y = box_y + 260
        pygame.draw.line(screen, (255, 255, 255, 100),
                        (box_x + 40, line_y), (box_x + box_width - 40, line_y), 2)

        # Episode info at bottom
        episode_render = info_font.render(f"Épisode {episode}/{total_episodes}", True, (220, 220, 220))
        episode_rect = episode_render.get_rect(center=(width // 2, box_y + 285))
        screen.blit(episode_render, episode_rect)

        pygame.display.flip()

    def run_pygame_demo(self):
        """Run pygame demo in separate window."""
        import pygame

        try:
            map_name = self.map_size.get()
            is_slippery = self.is_slippery.get()
            delay = self.demo_speed.get()

            if self.custom_map:
                demo_env = gym.make("FrozenLake-v1", desc=self.custom_map,
                                  is_slippery=is_slippery, render_mode="human")
            else:
                demo_env = gym.make("FrozenLake-v1", map_name=map_name,
                                  is_slippery=is_slippery, render_mode="human")

            action_names = {0: "←GAUCHE", 1: "↓BAS", 2: "→DROITE", 3: "↑HAUT"}
            successes = 0
            quit_requested = False

            # Trigger initial render to create window
            demo_env.reset()
            demo_env.render()

            # Get pygame window - try different attributes
            pygame_window = None
            if hasattr(demo_env.unwrapped, 'window'):
                pygame_window = demo_env.unwrapped.window
            elif hasattr(demo_env.unwrapped, 'screen'):
                pygame_window = demo_env.unwrapped.screen
            else:
                # Get display surface directly
                pygame_window = pygame.display.get_surface()

            for episode in range(5):
                if quit_requested:
                    break

                state, info = demo_env.reset()
                self.message_queue.put(("log", f"\n🎮 Pygame Épisode {episode + 1}/5"))
                done = False
                step = 0

                while not done and step < 100:
                    # Check for pygame events (window close, etc.)
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            quit_requested = True
                            self.message_queue.put(("log", "🛑 Fermeture demandée par l'utilisateur"))
                            break

                    if quit_requested:
                        break

                    time.sleep(delay)
                    action = np.argmax(self.agent.q_table[state])
                    self.message_queue.put(("log", f"   {action_names[action]}"))

                    next_state, reward, terminated, truncated, info = demo_env.step(action)
                    done = terminated or truncated

                    if done:
                        # Show result screen
                        if reward > 0:
                            self.message_queue.put(("log", f"   ✅ SUCCÈS en {step + 1} étapes!"))
                            if pygame_window:
                                self.show_result_screen(pygame_window, True, episode + 1, 5, step + 1)
                            successes += 1
                        else:
                            self.message_queue.put(("log", f"   ❌ DÉFAITE après {step + 1} étapes"))
                            if pygame_window:
                                self.show_result_screen(pygame_window, False, episode + 1, 5, step + 1)

                        # Wait to show result (2 seconds or user close)
                        start_wait = time.time()
                        while time.time() - start_wait < 2.0:
                            for event in pygame.event.get():
                                if event.type == pygame.QUIT:
                                    quit_requested = True
                                    break
                            if quit_requested:
                                break
                            time.sleep(0.1)

                    state = next_state
                    step += 1

                if quit_requested:
                    break

            demo_env.close()

            if quit_requested:
                self.message_queue.put(("log", f"\n🛑 Pygame fermé par l'utilisateur. Succès: {successes}/5"))
            else:
                self.message_queue.put(("log", f"\n🏁 Pygame terminé! Succès: {successes}/5"))
            self.message_queue.put(("pygame_complete", None))

        except Exception as e:
            self.message_queue.put(("log", f"❌ Erreur pygame: {str(e)}"))
            import traceback
            self.message_queue.put(("log", traceback.format_exc()))
            self.message_queue.put(("pygame_complete", None))

    def run_grid_demo(self):
        """Run visual demo with animation in the grid (in separate thread)."""
        try:
            action_names = {0: "←GAUCHE", 1: "↓BAS", 2: "→DROITE", 3: "↑HAUT"}
            successes = 0
            max_steps = self.demo_max_steps.get()

            for episode in range(5):
                if not self.is_demo_running:
                    self.message_queue.put(("log", "⏹️ Démo arrêtée par l'utilisateur"))
                    break

                state, info = self.env.reset()
                self.message_queue.put(("log", f"\n🎮 Démo Épisode {episode + 1}/5"))
                self.message_queue.put(("draw_grid", state))  # Show initial position

                done = False
                step = 0
                path = [state]

                while not done and step < max_steps and self.is_demo_running:
                    # Read speed in real-time
                    delay = self.demo_speed.get()
                    time.sleep(delay)

                    if not self.is_demo_running:
                        break

                    action = np.argmax(self.agent.q_table[state])
                    self.message_queue.put(("log", f"   Step {step + 1}: {action_names[action]}"))

                    next_state, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated

                    path.append(next_state)
                    self.message_queue.put(("draw_grid", next_state))  # Animate movement

                    if done:
                        if reward > 0:
                            self.message_queue.put(("log", f"   ✅ SUCCÈS en {step + 1} étapes!"))
                            self.message_queue.put(("draw_grid_success", (next_state, path)))
                            successes += 1
                        else:
                            self.message_queue.put(("log", f"   ❌ Échec après {step + 1} étapes"))
                            self.message_queue.put(("draw_grid_fail", next_state))
                        time.sleep(delay * 2)  # Pause to see result

                    state = next_state
                    step += 1

            if self.is_demo_running:
                self.message_queue.put(("log", f"\n🏁 Démo terminée! Succès: {successes}/5 ({successes*20}%)"))
            self.is_demo_running = False
            self.message_queue.put(("demo_complete", None))

        except Exception as e:
            self.message_queue.put(("log", f"❌ Erreur démo: {str(e)}"))
            self.is_demo_running = False
            self.message_queue.put(("demo_complete", None))

    def check_queue(self):
        """Check message queue and update UI (optimized)."""
        try:
            # Process multiple messages at once for better performance
            messages_processed = 0
            max_messages_per_cycle = 10  # Limit to avoid UI lag

            while not self.message_queue.empty() and messages_processed < max_messages_per_cycle:
                msg_type, data = self.message_queue.get_nowait()
                messages_processed += 1

                if msg_type == "log":
                    self.log(data)

                elif msg_type == "batch_update":
                    # Handle all updates at once (much faster!)
                    if "progress" in data:
                        episode, total = data["progress"]
                        self.progress_var.set(f"{episode} / {total} épisodes")
                        self.progress_bar['maximum'] = total
                        self.progress_bar['value'] = episode

                    if "epsilon" in data:
                        self.epsilon_var.set(f"{data['epsilon']:.3f}")

                    if "winrate" in data:
                        winrate = data["winrate"]
                        self.winrate_var.set(f"{winrate:.1f}%")
                        # Color code based on performance
                        if winrate > 70:
                            self.winrate_label.config(foreground='darkgreen')
                        elif winrate > 50:
                            self.winrate_label.config(foreground='green')
                        elif winrate > 30:
                            self.winrate_label.config(foreground='orange')
                        else:
                            self.winrate_label.config(foreground='red')

                    if "reward" in data:
                        self.reward_var.set(f"{data['reward']:.3f}")

                    if "time" in data:
                        elapsed = data["time"]
                        mins = int(elapsed // 60)
                        secs = int(elapsed % 60)
                        self.time_var.set(f"{mins}m {secs}s" if mins > 0 else f"{secs}s")

                    if "eta" in data:
                        eta = data["eta"]
                        if eta > 0:
                            mins = int(eta // 60)
                            secs = int(eta % 60)
                            self.eta_var.set(f"{mins}m {secs}s" if mins > 0 else f"{secs}s")
                        else:
                            self.eta_var.set("--")

                    # Draw curve only every 50 episodes to avoid lag and crashes on macOS
                    if len(self.rewards_history) > 0 and len(self.rewards_history) - self.last_graph_update >= 50:
                        try:
                            self.show_learning_curve()
                            self.last_graph_update = len(self.rewards_history)
                        except Exception as e:
                            # Silently ignore graph errors to prevent crashes
                            pass

                elif msg_type == "progress":
                    episode, total = data
                    self.progress_var.set(f"{episode} / {total} épisodes")
                    self.progress_bar['maximum'] = total
                    self.progress_bar['value'] = episode

                elif msg_type == "epsilon":
                    self.epsilon_var.set(f"{data:.3f}")

                elif msg_type == "winrate":
                    self.winrate_var.set(f"{data:.1f}%")
                    # Color code based on performance
                    if data > 70:
                        self.winrate_label.config(foreground='darkgreen')
                    elif data > 50:
                        self.winrate_label.config(foreground='green')
                    elif data > 30:
                        self.winrate_label.config(foreground='orange')
                    else:
                        self.winrate_label.config(foreground='red')

                elif msg_type == "reward":
                    self.reward_var.set(f"{data:.3f}")

                elif msg_type == "time":
                    mins = int(data // 60)
                    secs = int(data % 60)
                    self.time_var.set(f"{mins}m {secs}s" if mins > 0 else f"{secs}s")

                elif msg_type == "eta":
                    if data > 0:
                        mins = int(data // 60)
                        secs = int(data % 60)
                        self.eta_var.set(f"{mins}m {secs}s" if mins > 0 else f"{secs}s")
                    else:
                        self.eta_var.set("--")

                elif msg_type == "draw_curve":
                    pass  # Removed learning curve

                elif msg_type == "draw_grid_training":
                    # Draw grid during training with enhanced info
                    if isinstance(data, dict):
                        # New format with action, reward, etc.
                        self.draw_grid(agent_pos=data['position'])
                    else:
                        # Old format - just position
                        self.draw_grid(agent_pos=data)

                elif msg_type == "draw_grid":
                    # Draw grid with agent at position
                    self.draw_grid(agent_pos=data)

                elif msg_type == "draw_grid_success":
                    # Draw grid with path showing success
                    agent_pos, path = data
                    self.draw_grid(agent_pos=agent_pos, path=path)

                elif msg_type == "draw_grid_fail":
                    # Draw grid showing failure
                    self.draw_grid(agent_pos=data)

                elif msg_type == "demo_complete":
                    self.demo_button.config(state=tk.NORMAL)
                    self.stop_demo_button.config(state=tk.DISABLED)
                    # Show final grid with Q-values
                    if self.agent:
                        self.draw_grid(show_values=True)

                elif msg_type == "pygame_complete":
                    self.pygame_button.config(state=tk.NORMAL)

                elif msg_type == "show_graph":
                    self.show_learning_curve()

                elif msg_type == "complete":
                    self.is_training = False
                    self.train_button.config(state=tk.NORMAL)
                    self.stop_button.config(state=tk.DISABLED)
                    self.demo_button.config(state=tk.NORMAL)
                    self.pygame_button.config(state=tk.NORMAL)
                    # Show grid with learned policy
                    if self.env:
                        self.draw_grid(show_values=True)

        except Exception as e:
            print(f"Queue error: {e}")

        self.root.after(100, self.check_queue)


def main():
    """Launch the GUI application."""
    root = tk.Tk()
    app = FrozenLakeGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
