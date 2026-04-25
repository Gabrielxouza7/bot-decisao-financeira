import os
from datetime import datetime
from environment import (FinancialMDP, MarketTendency, MarketPositions, MarketActions)
import numpy as np
import matplotlib.pyplot as plt

ACTION_NAMES = ['Manter', 'Comprar', 'Vender']
MARKET_NAMES = ['Caindo', 'Estável', 'Subindo']
POSITION_NAMES = ['Sem posição', 'Comprado']


# ================= RUN MANAGEMENT =================
def create_run_dir(base_dir="runs"):
    os.makedirs(base_dir, exist_ok=True)
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_path = os.path.join(base_dir, run_name)
    os.makedirs(run_path, exist_ok=True)
    return run_path


# ================= PLOTS =================
def plot_learning_curve(episode_rewards: list, save_dir, window=50, title="Q-learning"):
    smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
    plt.figure(figsize=(10, 4))
    plt.plot(episode_rewards, alpha=0.3, label='Episódio')
    plt.plot(smoothed, label=f'Média {window} ep.')
    plt.xlabel('Episódio')
    plt.ylabel('Recompensa acumulada')
    plt.title(f'Curva de Aprendizado — {title}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_curve.png'), dpi=150)
    plt.show()


def plot_value_map(V: np.ndarray, n_market: int, n_positions: int, save_dir):
    grid = V.reshape(n_market, n_positions)
    _, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(grid, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(n_positions))
    ax.set_xticklabels(POSITION_NAMES)
    ax.set_yticks(range(n_market))
    ax.set_yticklabels(MARKET_NAMES)
    ax.set_title('Mapa de Valores V(s)')
    plt.colorbar(im, ax=ax)
    for i in range(n_market):
        for j in range(n_positions):
            ax.text(j, i, f'{grid[i,j]:.2f}', ha='center', va='center', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'value_map.png'), dpi=150)
    plt.show()


def plot_policy(policy: np.ndarray, n_market: int, n_positions: int, save_dir):
    grid = policy.reshape(n_market, n_positions)
    _, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(grid, cmap='coolwarm', aspect='auto', vmin=0, vmax=2)
    ax.set_xticks(range(n_positions))
    ax.set_xticklabels(POSITION_NAMES)
    ax.set_yticks(range(n_market))
    ax.set_yticklabels(MARKET_NAMES)
    ax.set_title('Política Aprendida')
    for i in range(n_market):
        for j in range(n_positions):
            ax.text(j, i, ACTION_NAMES[grid[i,j]], ha='center', va='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'policy_map.png'), dpi=150)
    plt.show()


def plot_trajectory(env: FinancialMDP, policy: np.ndarray, save_dir, n_steps=50):
    state = env.encode_state(MarketTendency.STABLE, MarketPositions.NO_POSITION)
    states, actions, rewards = [], [], []
    total = 0

    for _ in range(n_steps):
        action = MarketActions(policy[state])
        next_state, reward = env.step(state, action)

        states.append(state)
        actions.append(action.value)
        rewards.append(reward)

        total += reward
        state = next_state

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    ax1.plot(rewards, marker='o', markersize=4)
    ax1.set_ylabel('Recompensa')
    ax1.set_title(f'Trajetória do Agente (total={total:.2f})')

    ax2.plot(actions, marker='s', color='orange', markersize=4)
    ax2.set_yticks([0,1,2])
    ax2.set_yticklabels(ACTION_NAMES)
    ax2.set_ylabel('Ação')
    ax2.set_xlabel('Passo')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'trajectory.png'), dpi=150)
    plt.show()


def compare_gammas(env: FinancialMDP, save_dir, gammas=[0.7, 0.9, 0.99], n_episodes=1500):
    from qlearning import qlearning

    plt.figure(figsize=(10, 4))

    for gamma in gammas:
        _, rewards, _ = qlearning(env, n_episodes=n_episodes, gamma=gamma)
        window = 50
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(smoothed, label=f'γ={gamma}')

    plt.xlabel('Episódio')
    plt.ylabel('Recompensa (média móvel)')
    plt.title('Impacto de gamma no aprendizado')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gamma_comparison.png'), dpi=150)
    plt.show()


def compare_epsilons(env: FinancialMDP, save_dir, n_episodes=2000):
    from qlearning import qlearning

    estrategias = [
        {
            'label': 'ε fixo alto (0.9)',
            'params': dict(epsilon_start=0.9, epsilon_end=0.9, epsilon_decay=1.0)
        },
        {
            'label': 'ε fixo baixo (0.1)',
            'params': dict(epsilon_start=0.1, epsilon_end=0.1, epsilon_decay=1.0)
        },
        {
            'label': 'ε com decaimento (1.0 → 0.05)',
            'params': dict(epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995)
        },
    ]

    window = 50
    plt.figure(figsize=(10, 4))

    for e in estrategias:
        _, rewards, _ = qlearning(env, n_episodes=n_episodes, **e['params'])
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(smoothed, label=e['label'])

    plt.xlabel('Episódio')
    plt.ylabel('Recompensa (média móvel)')
    plt.title('Comparação de Estratégias de Exploração (ε-greedy)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'epsilon_comparison.png'), dpi=150)
    plt.show()