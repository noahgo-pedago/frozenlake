#!/usr/bin/env python3
"""
Pygame Demo Script for FrozenLake
Run separately to avoid threading issues on macOS
"""

import sys
import pickle
import gymnasium as gym
import numpy as np
import pygame
import time


def show_result_screen(screen, success, episode, total_episodes, steps):
    """Display victory/defeat overlay on pygame window."""
    # Get screen dimensions
    width, height = screen.get_size()

    # Create semi-transparent overlay
    overlay = pygame.Surface((width, height))
    overlay.set_alpha(220)
    overlay.fill((0, 0, 0))
    screen.blit(overlay, (0, 0))

    # Calculate font sizes proportional to window size
    # Base sizes for a 640x640 window (8x8 grid typical size)
    scale = min(width, height) / 640.0
    title_size = max(24, int(72 * scale))
    subtitle_size = max(16, int(36 * scale))
    info_size = max(12, int(28 * scale))
    emoji_size = max(32, int(100 * scale))

    # Fonts
    try:
        title_font = pygame.font.SysFont('arial', title_size, bold=True)
        subtitle_font = pygame.font.SysFont('arial', subtitle_size, bold=True)
        info_font = pygame.font.SysFont('arial', info_size)
        emoji_font = pygame.font.SysFont('arial', emoji_size)
    except:
        title_font = pygame.font.Font(None, title_size)
        subtitle_font = pygame.font.Font(None, subtitle_size)
        info_font = pygame.font.Font(None, info_size)
        emoji_font = pygame.font.Font(None, emoji_size)

    if success:
        # Victory screen
        bg_color = (34, 139, 34)
        border_color = (50, 205, 50)
        title_color = (255, 255, 100)
        title_text = "VICTOIRE!"
        emoji = "🎉"
        subtitle_text = f"But atteint en {steps} étape{'s' if steps > 1 else ''}!"
    else:
        # Defeat screen
        bg_color = (139, 0, 0)
        border_color = (220, 20, 60)
        title_color = (255, 200, 200)
        title_text = "DÉFAITE"
        emoji = "💀"
        subtitle_text = "Tombé dans un trou!"

    # Calculate box dimensions proportional to window
    margin = max(30, int(60 * scale))
    box_width = width - (margin * 2)
    box_height = max(150, int(320 * scale))
    box_x = margin
    box_y = height // 2 - box_height // 2

    # Adjust border radius proportionally
    border_radius = max(10, int(20 * scale))

    # Draw outer glow effect
    glow_iterations = max(3, int(5 * scale))
    for i in range(glow_iterations):
        glow_alpha = 30 - (i * 5)
        glow_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.rect(glow_surface, (*border_color, glow_alpha),
                       (box_x - i*2, box_y - i*2, box_width + i*4, box_height + i*4),
                       border_radius=border_radius)
        screen.blit(glow_surface, (0, 0))

    # Draw main background box
    pygame.draw.rect(screen, bg_color, (box_x, box_y, box_width, box_height), border_radius=border_radius)
    pygame.draw.rect(screen, border_color, (box_x, box_y, box_width, box_height), max(2, int(5 * scale)), border_radius=border_radius)

    # Calculate vertical spacing proportionally
    emoji_y = box_y + int(80 * scale)
    title_y = box_y + int(160 * scale)
    subtitle_y = box_y + int(220 * scale)
    episode_y = box_y + int(285 * scale)

    # Render emoji
    emoji_render = emoji_font.render(emoji, True, (255, 255, 255))
    emoji_rect = emoji_render.get_rect(center=(width // 2, emoji_y))
    screen.blit(emoji_render, emoji_rect)

    # Render title with shadow
    shadow_offset = max(1, int(2 * scale))
    title_render = title_font.render(title_text, True, (0, 0, 0))
    title_rect = title_render.get_rect(center=(width // 2 + shadow_offset, title_y + shadow_offset))
    screen.blit(title_render, title_rect)

    title_render = title_font.render(title_text, True, title_color)
    title_rect = title_render.get_rect(center=(width // 2, title_y))
    screen.blit(title_render, title_rect)

    # Render subtitle
    subtitle_render = subtitle_font.render(subtitle_text, True, (255, 255, 255))
    subtitle_rect = subtitle_render.get_rect(center=(width // 2, subtitle_y))
    screen.blit(subtitle_render, subtitle_rect)

    # Episode info
    episode_render = info_font.render(f"Épisode {episode}/{total_episodes}", True, (220, 220, 220))
    episode_rect = episode_render.get_rect(center=(width // 2, episode_y))
    screen.blit(episode_render, episode_rect)

    pygame.display.flip()


def run_demo(q_table_path, map_name, is_slippery, delay, custom_map_path=None):
    """Run the pygame demo."""
    # Load Q-table
    with open(q_table_path, 'rb') as f:
        q_table = pickle.load(f)

    # Load custom map if provided
    custom_map = None
    if custom_map_path:
        with open(custom_map_path, 'rb') as f:
            custom_map = pickle.load(f)

    # Create environment
    if custom_map:
        env = gym.make("FrozenLake-v1", desc=custom_map,
                      is_slippery=is_slippery, render_mode="human")
    else:
        env = gym.make("FrozenLake-v1", map_name=map_name,
                      is_slippery=is_slippery, render_mode="human")

    action_names = {0: "←GAUCHE", 1: "↓BAS", 2: "→DROITE", 3: "↑HAUT"}
    successes = 0
    quit_requested = False

    # Initial render
    env.reset()
    env.render()

    # Get pygame window
    pygame_window = None
    if hasattr(env.unwrapped, 'window'):
        pygame_window = env.unwrapped.window
    elif hasattr(env.unwrapped, 'screen'):
        pygame_window = env.unwrapped.screen
    else:
        pygame_window = pygame.display.get_surface()

    for episode in range(5):
        if quit_requested:
            break

        state, info = env.reset()
        print(f"\n🎮 Épisode {episode + 1}/5")
        done = False
        step = 0

        while not done and step < 100:
            # Check for pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_requested = True
                    print("🛑 Fermeture demandée")
                    break

            if quit_requested:
                break

            time.sleep(delay)
            action = np.argmax(q_table[state])
            print(f"   {action_names[action]}")

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if done:
                if reward > 0:
                    print(f"   ✅ SUCCÈS en {step + 1} étapes!")
                    if pygame_window:
                        show_result_screen(pygame_window, True, episode + 1, 5, step + 1)
                    successes += 1
                else:
                    print(f"   ❌ DÉFAITE après {step + 1} étapes")
                    if pygame_window:
                        show_result_screen(pygame_window, False, episode + 1, 5, step + 1)

                # Wait to show result
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

    print(f"\n🏁 Démo terminée! Succès: {successes}/5 ({successes*20}%)")
    env.close()


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: pygame_demo.py <q_table_path> <map_name> <is_slippery> <delay> [custom_map_path]")
        sys.exit(1)

    q_table_path = sys.argv[1]
    map_name = sys.argv[2]
    is_slippery = sys.argv[3].lower() == 'true'
    delay = float(sys.argv[4])
    custom_map_path = sys.argv[5] if len(sys.argv) > 5 else None

    run_demo(q_table_path, map_name, is_slippery, delay, custom_map_path)
