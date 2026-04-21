from environment import (FinancialMDP, MarketTendency, MarketPositions, MarketActions)
import numpy as np
import matplotlib.pyplot as plt

ACTION_NAMES = ['Manter', 'Comprar', 'Vender']
MARKET_NAMES = ['Caindo', 'Estável', 'Subindo']
POSITION_NAMES = ['Sem posição', 'Comprado']

def plot_learning_curve(episode_rewards: list, window=50, title="Q-learning"):
    smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
    plt.figure(figsize=(10, 4))
    plt.plot(episode_rewards, alpha=0.3, label='Episódio')
    plt.plot(smoothed, label=f'Média {window} ep.')
    plt.xlabel('Episódio')
    plt.ylabel('Recompensa acumulada')
    plt.title(f'Curva de Aprendizado — {title}')
    plt.legend()
    plt.tight_layout()
    plt.savefig('learning_curve.png', dpi=150)
    plt.show()

def plot_value_map(V: np.ndarray, n_market: int, n_positions: int):
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
    plt.savefig('value_map.png', dpi=150)
    plt.show()

def plot_policy(policy: np.ndarray, n_market: int, n_positions: int):
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
    plt.savefig('policy_map.png', dpi=150)
    plt.show()

def plot_trajectory(env: FinancialMDP, policy: np.ndarray, n_steps=50):
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
    plt.savefig('trajectory.png', dpi=150)
    plt.show()

def compare_gammas(env: FinancialMDP, gammas=[0.7, 0.9, 0.99], n_episodes=1500):
    plt.figure(figsize=(10, 4))
    for gamma in gammas:
        from qlearning import qlearning
        _, rewards, _ = qlearning(env, n_episodes=n_episodes, gamma=gamma)
        window = 50
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(smoothed, label=f'γ={gamma}')
    plt.xlabel('Episódio')
    plt.ylabel('Recompensa (média móvel)')
    plt.title('Impacto de gamma no aprendizado')
    plt.legend()
    plt.tight_layout()
    plt.savefig('gamma_comparison.png', dpi=150)
    plt.show()