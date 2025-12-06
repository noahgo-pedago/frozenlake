"""
New UI Methods for FrozenLake GUI - Refactored Version
These methods will replace the old setup_parameters_panel
"""

# Method 1: Toggle between Simple and Advanced mode
def toggle_mode(self):
    """Toggle between simple and advanced mode."""
    self.advanced_mode = not self.advanced_mode

    # Rebuild the parameters panel
    # We need to destroy and recreate the panel
    for widget in self.params_container.winfo_children():
        widget.destroy()

    self.build_parameters_content(self.params_container)

    # Update button text
    mode_text = "Mode Simple" if self.advanced_mode else "Mode Avancé"
    self.mode_toggle_button.config(text=f"📊 Passer en {mode_text}")


# Method 2: Build simple mode parameters
def build_simple_mode(self, parent):
    """Build simplified parameter interface for beginners."""
    row = 0

    # === PRESETS AS PRIMARY CHOICE ===
    presets_frame = ttk.LabelFrame(parent, text="🎯 Choisissez votre stratégie", padding="10")
    presets_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
    row += 1

    # Preset buttons with clear descriptions
    preset_configs = [
        ("🐢 Apprend lentement\nmais retient bien", "slow",
         "Idéal pour: bien comprendre\nProcessus lent mais stable", self.preset_slow_learner),
        ("⚖️ Équilibré\n(recommandé)", "balanced",
         "Idéal pour: la plupart des cas\nBon compromis vitesse/qualité", self.preset_balanced),
        ("🐰 Apprend vite\nmais oublie facilement", "fast",
         "Idéal pour: tests rapides\nProcessus rapide mais instable", self.preset_fast_learner),
    ]

    for i, (text, name, tooltip, command) in enumerate(preset_configs):
        btn = ttk.Button(presets_frame, text=text, command=command, width=25)
        btn.grid(row=0, column=i, padx=5)
        ToolTip(btn, tooltip)

    # === MAP CONFIGURATION ===
    map_frame = ttk.LabelFrame(parent, text="🗺️ Carte de jeu", padding="10")
    map_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
    row += 1

    # Map size
    ttk.Label(map_frame, text="Taille:").grid(row=0, column=0, sticky=tk.W, pady=5, padx=(0, 10))
    self.map_size = tk.StringVar(value="4x4")
    map_combo = ttk.Combobox(map_frame, textvariable=self.map_size,
                            values=["4x4 (Facile)", "8x8 (Difficile)"],
                            state="readonly", width=15)
    map_combo.grid(row=0, column=1, sticky=tk.W, pady=5)
    ToolTip(map_combo, "Taille de la carte: Plus grande = plus difficile")

    # Create custom map button (always visible now)
    create_map_btn = ttk.Button(map_frame, text="✏️ Créer ma propre carte",
                               command=self.open_map_creator)
    create_map_btn.grid(row=1, column=0, columnspan=2, pady=(10, 5))
    ToolTip(create_map_btn, "Créez une carte personnalisée avec vos propres obstacles")

    # Show current custom map status
    self.custom_map_label = ttk.Label(map_frame, text="", foreground="green")
    self.custom_map_label.grid(row=2, column=0, columnspan=2)

    # Slippery option
    self.is_slippery = tk.BooleanVar(value=False)
    slip_check = ttk.Checkbutton(map_frame, text="🧊 Glace glissante (plus difficile)",
                                 variable=self.is_slippery)
    slip_check.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=5)
    ToolTip(slip_check, "L'agent peut glisser et aller dans une direction différente")

    # === TRAINING SPEED ===
    speed_frame = ttk.LabelFrame(parent, text="⏱️ Vitesse d'entraînement", padding="10")
    speed_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
    row += 1

    ttk.Label(speed_frame, text="Nombre d'essais:").grid(row=0, column=0, sticky=tk.W, pady=5)
    self.episodes = tk.IntVar(value=5000)
    episodes_spinbox = ttk.Spinbox(speed_frame, from_=1000, to=50000, increment=1000,
                                   textvariable=self.episodes, width=10)
    episodes_spinbox.grid(row=0, column=1, sticky=tk.W, pady=5, padx=(10, 0))
    ToolTip(episodes_spinbox, "Plus d'essais = meilleur apprentissage mais plus lent")

    ttk.Label(speed_frame, text="Affichage:").grid(row=1, column=0, sticky=tk.W, pady=5)
    self.update_frequency = tk.IntVar(value=100)
    update_combo = ttk.Combobox(speed_frame, textvariable=self.update_frequency,
                               values=[50, 100, 250, 500], state="readonly", width=8)
    update_combo.grid(row=1, column=1, sticky=tk.W, pady=5, padx=(10, 0))
    ToolTip(update_combo, "Fréquence de mise à jour de l'affichage (plus bas = plus fluide)")

    # Visualization toggle
    self.show_training_viz = tk.BooleanVar(value=False)
    viz_check = ttk.Checkbutton(speed_frame, text="👁️ Voir l'animation pendant l'entraînement",
                               variable=self.show_training_viz)
    viz_check.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=5)
    ToolTip(viz_check, "Affiche l'agent dans la grille pendant l'entraînement (plus lent)")


# Method 3: Build advanced mode parameters
def build_advanced_mode(self, parent):
    """Build complete parameter interface for advanced users."""
    row = 0

    # === HYPERPARAMETERS ===
    hyper_frame = ttk.LabelFrame(parent, text="🎓 Hyperparamètres", padding="10")
    hyper_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
    row += 1

    # Learning rate
    lr_row = 0
    ttk.Label(hyper_frame, text="Taux d'apprentissage (α):").grid(row=lr_row, column=0, sticky=tk.W, pady=5)
    self.learning_rate = tk.DoubleVar(value=0.15)
    self.lr_label = ttk.Label(hyper_frame, text="0.15")
    self.lr_label.grid(row=lr_row, column=2, padx=5)
    lr_scale = ttk.Scale(hyper_frame, from_=0.01, to=0.5, variable=self.learning_rate,
                        orient=tk.HORIZONTAL, length=150,
                        command=lambda v: self.lr_label.config(text=f"{float(v):.2f}"))
    lr_scale.grid(row=lr_row, column=1, sticky=(tk.W, tk.E), pady=5)
    ToolTip(lr_scale, "Élevé (0.3-0.5): Apprend vite\nBas (0.05-0.15): Apprend lentement mais mieux")

    # Discount factor
    lr_row += 1
    ttk.Label(hyper_frame, text="Facteur de discount (γ):").grid(row=lr_row, column=0, sticky=tk.W, pady=5)
    self.discount_factor = tk.DoubleVar(value=0.98)
    self.df_label = ttk.Label(hyper_frame, text="0.98")
    self.df_label.grid(row=lr_row, column=2, padx=5)
    df_scale = ttk.Scale(hyper_frame, from_=0.5, to=0.99, variable=self.discount_factor,
                        orient=tk.HORIZONTAL, length=150,
                        command=lambda v: self.df_label.config(text=f"{float(v):.2f}"))
    df_scale.grid(row=lr_row, column=1, sticky=(tk.W, tk.E), pady=5)
    ToolTip(df_scale, "Proche de 1: Planifie à long terme\nProche de 0.5: Préfère les récompenses immédiates")

    # Epsilon decay
    lr_row += 1
    ttk.Label(hyper_frame, text="Décroissance exploration (ε):").grid(row=lr_row, column=0, sticky=tk.W, pady=5)
    self.epsilon_decay = tk.DoubleVar(value=0.996)
    self.ed_label = ttk.Label(hyper_frame, text="0.996")
    self.ed_label.grid(row=lr_row, column=2, padx=5)
    ed_scale = ttk.Scale(hyper_frame, from_=0.95, to=0.999, variable=self.epsilon_decay,
                        orient=tk.HORIZONTAL, length=150,
                        command=lambda v: self.ed_label.config(text=f"{float(v):.3f}"))
    ed_scale.grid(row=lr_row, column=1, sticky=(tk.W, tk.E), pady=5)
    ToolTip(ed_scale, "Contrôle la vitesse à laquelle l'agent arrête d'explorer")

    # Episodes
    lr_row += 1
    ttk.Label(hyper_frame, text="Nombre d'épisodes:").grid(row=lr_row, column=0, sticky=tk.W, pady=5)
    self.episodes = tk.IntVar(value=10000)
    episodes_entry = ttk.Entry(hyper_frame, textvariable=self.episodes, width=10)
    episodes_entry.grid(row=lr_row, column=1, sticky=tk.W, pady=5)

    # Update frequency
    lr_row += 1
    ttk.Label(hyper_frame, text="Fréquence d'affichage:").grid(row=lr_row, column=0, sticky=tk.W, pady=5)
    self.update_frequency = tk.IntVar(value=100)
    update_combo = ttk.Combobox(hyper_frame, textvariable=self.update_frequency,
                               values=[50, 100, 250, 500], state="readonly", width=8)
    update_combo.grid(row=lr_row, column=1, sticky=tk.W, pady=5)

    # Call simple mode's map configuration (reuse it)
    self.build_simple_mode(parent) # For map section
    row += 4  # Account for map section rows

    # === REWARDS (Positive only) ===
    rewards_frame = ttk.LabelFrame(parent, text="🎁 Récompenses (Positif)", padding="10")
    rewards_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
    row += 1

    self.rewards = {}
    reward_configs = [
        ("goal_reached", "🎯 Atteindre le but:", 0.0, 10.0, 1.0,
         "Récompense quand l'agent atteint le but"),
        ("step_towards_goal", "👣 Se rapprocher du but:", 0.0, 1.0, 0.0,
         "Petite récompense quand l'agent se rapproche"),
    ]

    for i, (key, label, min_val, max_val, default, tooltip) in enumerate(reward_configs):
        ttk.Label(rewards_frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=5)
        self.rewards[key] = tk.DoubleVar(value=default)
        spinbox = ttk.Spinbox(rewards_frame, from_=min_val, to=max_val, increment=0.1,
                             textvariable=self.rewards[key], width=8, format="%.1f")
        spinbox.grid(row=i, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        ToolTip(spinbox, tooltip)

    # === PENALTIES (Negative only) ===
    penalties_frame = ttk.LabelFrame(parent, text="⚠️ Pénalités (Négatif)", padding="10")
    penalties_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
    row += 1

    self.penalties = {}
    penalty_configs = [
        ("hole_fall", "💀 Tomber dans un trou:", -5.0, 0.0, 0.0,
         "Pénalité quand l'agent tombe dans un trou"),
        ("step_cost", "👟 Coût par étape:", -1.0, 0.0, 0.0,
         "Pénalité légère à chaque mouvement (encourage vitesse)"),
        ("wall_hit", "🧱 Frapper un mur:", -1.0, 0.0, 0.0,
         "Pénalité quand l'agent essaie de sortir de la carte"),
        ("revisit_cell", "🔁 Revisiter une case:", -1.0, 0.0, 0.0,
         "Pénalité quand l'agent revisite une case déjà visitée"),
        ("step_away_goal", "↩️ S'éloigner du but:", -2.0, 0.0, 0.0,
         "Pénalité quand l'agent s'éloigne du but"),
        ("loop_detected", "🔄 Boucle détectée:", -1.0, 0.0, 0.0,
         "Pénalité quand l'agent fait une boucle (5 étapes)"),
    ]

    for i, (key, label, min_val, max_val, default, tooltip) in enumerate(penalty_configs):
        ttk.Label(penalties_frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=5)
        self.penalties[key] = tk.DoubleVar(value=default)
        spinbox = ttk.Spinbox(penalties_frame, from_=min_val, to=max_val, increment=0.1,
                             textvariable=self.penalties[key], width=8, format="%.1f")
        spinbox.grid(row=i, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        ToolTip(spinbox, tooltip)

    # === ACTION BIASES ===
    action_frame = ttk.LabelFrame(parent, text="🎮 Biais des actions", padding="10")
    action_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
    row += 1

    ttk.Label(action_frame, text="Favoriser certaines directions:",
             font=('Arial', 9, 'italic')).grid(row=0, column=0, columnspan=2, pady=(0, 10))

    self.action_weights = {}
    actions = [("⬅️ Gauche", "left"), ("⬇️ Bas", "down"),
               ("➡️ Droite", "right"), ("⬆️ Haut", "up")]

    for i, (label, direction) in enumerate(actions):
        ttk.Label(action_frame, text=label).grid(row=i+1, column=0, sticky=tk.W, pady=5)
        self.action_weights[direction] = tk.DoubleVar(value=1.0)
        scale = ttk.Scale(action_frame, from_=0.1, to=5.0, variable=self.action_weights[direction],
                         orient=tk.HORIZONTAL, length=150)
        scale.grid(row=i+1, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        scale.config(command=self.update_action_probabilities)

    # Probability display
    self.action_prob_label = ttk.Label(action_frame, text="", foreground="blue",
                                       font=('Courier', 8))
    self.action_prob_label.grid(row=len(actions)+1, column=0, columnspan=2, pady=(10, 0))
    self.update_action_probabilities()


# Method 4: Preset configurations for simple mode
def preset_slow_learner(self):
    """Preset: Slow learner - learns slowly but retains well."""
    self.learning_rate.set(0.05)
    self.discount_factor.set(0.99)
    self.epsilon_decay.set(0.9995)
    self.episodes.set(20000)
    self.is_slippery.set(True)
    self.log("🐢 Stratégie 'Apprend lentement mais retient bien' chargée!")
    self.log("   → Idéal pour un apprentissage stable et de qualité")

def preset_balanced(self):
    """Preset: Balanced - recommended for most cases."""
    self.learning_rate.set(0.15)
    self.discount_factor.set(0.98)
    self.epsilon_decay.set(0.996)
    self.episodes.set(10000)
    self.is_slippery.set(True)
    self.log("⚖️ Stratégie 'Équilibré' chargée! (Recommandé)")
    self.log("   → Bon compromis entre vitesse et qualité")

def preset_fast_learner(self):
    """Preset: Fast learner - learns quickly but may forget."""
    self.learning_rate.set(0.3)
    self.discount_factor.set(0.95)
    self.epsilon_decay.set(0.997)
    self.episodes.set(5000)
    self.is_slippery.set(False)
    self.log("🐰 Stratégie 'Apprend vite mais oublie facilement' chargée!")
    self.log("   → Idéal pour des tests rapides")


# Method 5: Map creator dialog (improved)
def open_map_creator(self):
    """Open an improved map creation dialog."""
    dialog = tk.Toplevel(self.root)
    dialog.title("✏️ Créateur de Carte Personnalisée")
    dialog.geometry("400x300")
    dialog.transient(self.root)
    dialog.grab_set()

    # Instructions
    ttk.Label(dialog, text="Créez votre carte de jeu personnalisée",
             font=('Arial', 12, 'bold')).pack(pady=10)

    ttk.Label(dialog, text="La carte sera générée aléatoirement avec vos paramètres",
             wraplength=350).pack(pady=5)

    # Size
    size_frame = ttk.Frame(dialog)
    size_frame.pack(pady=10, padx=20, fill=tk.X)

    ttk.Label(size_frame, text="Taille de la carte:").pack(side=tk.LEFT)
    size_var = tk.IntVar(value=8)
    size_label = ttk.Label(size_frame, text="8x8")
    size_label.pack(side=tk.RIGHT)

    size_scale = ttk.Scale(size_frame, from_=4, to=12, variable=size_var,
                          orient=tk.HORIZONTAL,
                          command=lambda v: size_label.config(text=f"{int(float(v))}x{int(float(v))}"))
    size_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=10)

    # Hole probability
    hole_frame = ttk.Frame(dialog)
    hole_frame.pack(pady=10, padx=20, fill=tk.X)

    ttk.Label(hole_frame, text="Difficulté (trous):").pack(side=tk.LEFT)
    hole_var = tk.DoubleVar(value=0.2)
    hole_label = ttk.Label(hole_frame, text="20%")
    hole_label.pack(side=tk.RIGHT)

    hole_scale = ttk.Scale(hole_frame, from_=0.1, to=0.4, variable=hole_var,
                          orient=tk.HORIZONTAL,
                          command=lambda v: hole_label.config(text=f"{int(float(v)*100)}%"))
    hole_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=10)

    # Preview area
    preview_frame = ttk.LabelFrame(dialog, text="Aperçu", padding="10")
    preview_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

    preview_text = tk.Text(preview_frame, height=6, width=30, font=('Courier', 10))
    preview_text.pack()

    def update_preview(*args):
        """Generate and show preview."""
        size = size_var.get()
        prob = hole_var.get()

        # Generate map
        custom_map = generate_random_map(size, prob)

        # Display
        preview_text.delete('1.0', tk.END)
        for row in custom_map:
            preview_text.insert(tk.END, ' '.join(row) + '\n')

    # Auto-update preview
    size_var.trace_add('write', update_preview)
    hole_var.trace_add('write', update_preview)
    update_preview()

    # Buttons
    btn_frame = ttk.Frame(dialog)
    btn_frame.pack(pady=10)

    def on_generate():
        size = size_var.get()
        prob = hole_var.get()
        self.custom_map = generate_random_map(size, prob)
        self.custom_map_label.config(text=f"✓ Carte {size}x{size} créée!")
        self.map_size.set("Personnalisée")
        self.log(f"✅ Carte personnalisée {size}x{size} générée!")
        dialog.destroy()

    ttk.Button(btn_frame, text="✓ Créer cette carte", command=on_generate).pack(side=tk.LEFT, padx=5)
    ttk.Button(btn_frame, text="✗ Annuler", command=dialog.destroy).pack(side=tk.LEFT, padx=5)


# Method 6: Main setup with mode toggle
def setup_parameters_panel_new(self, parent):
    """Setup the main parameters panel with mode toggle."""
    # Container frame
    container = ttk.LabelFrame(parent, text="⚙️ Configuration", padding="5")
    container.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
    container.rowconfigure(1, weight=1)  # Make content scrollable
    container.columnconfigure(0, weight=1)

    # Mode toggle button at top
    self.mode_toggle_button = ttk.Button(container, text="📊 Passer en Mode Avancé",
                                         command=self.toggle_mode)
    self.mode_toggle_button.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)

    # Scrollable content area
    canvas = tk.Canvas(container, highlightthickness=0, width=380)
    scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    self.params_container = ttk.Frame(canvas)

    self.params_container.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=self.params_container, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))

    # Mouse wheel scrolling
    def on_mousewheel(event):
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    canvas.bind_all("<MouseWheel>", on_mousewheel)

    # Build initial content (simple mode)
    self.build_parameters_content(self.params_container)

def build_parameters_content(self, parent):
    """Build the actual parameter controls based on current mode."""
    if self.advanced_mode:
        self.build_advanced_mode(parent)
    else:
        self.build_simple_mode(parent)
