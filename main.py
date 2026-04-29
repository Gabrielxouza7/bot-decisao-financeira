from environment import FinancialMDP
from bellman import value_iteration
from qlearning import qlearning
from data import build_from_ticker
from evaluate import (
    plot_bellman_convergence,
    plot_bellman_vs_qlearning,
    plot_policies_comparison,
    plot_learning_curve, 
    plot_value_map,
    plot_policy, 
    plot_trajectory, 
    compare_gammas,
    compare_epsilons, 
    create_run_dir,
    generate_metrics_report
)
import numpy as np
import time

print("=" * 70)
print("  AGENTE INTELIGENTE DE DECISÃO FINANCEIRA — APRENDIZADO POR REFORÇO")
print("=" * 70)

# Constantes
TICKER = 'AAPL'
START  = '2023-01-01'
END    = '2024-01-01'

# ================= SETUP =================
print("\n[1/8] Criando diretório de execução...")
run_dir = create_run_dir()
print(f"     ✓ {run_dir}")

print("\n[2/8] Baixando e preparando dados...")
transition = build_from_ticker(TICKER, start=START, end=END)

print("\n[3/8] Inicializando MDP...")
env = FinancialMDP(seed=42, market_transition=transition)
T, R = env.build_transition_reward_tables()
print(f"     ✓ {env.n_states} estados | {env.n_actions} ações")

# ================= BELLMAN (PLANEJAMENTO) =================
print("\n[4/8] Executando Value Iteration (Bellman)...")
t1 = time.time()
V_star, pi_star, vi_history, n_iter = value_iteration(env, T, R, gamma=0.9)
t2 = time.time()
print(f"     ✓ Convergência em {(t2-t1)*1000:.0f}ms")

print("\n[5/8] Gerando visualizações Bellman...")
msg_convergence = plot_bellman_convergence(vi_history, n_iter, save_dir=run_dir)
print(f"     {msg_convergence.split(chr(10))[0]}")
plot_value_map(V_star, env.n_tendencies, env.n_positions, save_dir=run_dir, 
               title="Mapa de Valores — Bellman V*", method="bellman")
plot_policy(pi_star, env.n_tendencies, env.n_positions, save_dir=run_dir,
           title="Política Ótima — Bellman", method = "bellman")
print("     ✓ convergence, value_map, policy → PNG")

# ================= Q-LEARNING (APRENDIZADO) =================
print("\n[6/8] Executando Q-learning...")
Q, rewards, epsilons = qlearning(env, n_episodes=5000, gamma=0.9)
pi_ql = np.zeros(env.n_states, dtype=int)
for state in range(env.n_states):
    valid = env.valid_actions(env.decode_state(state)[1])
    pi_ql[state] = max(valid, key=lambda a: Q[state, a])
print(f"     ✓ Treinado em 5000 episódios")

print("\n[7/8] Gerando visualizações Q-learning...")
msg_qlearning = plot_learning_curve(rewards, save_dir=run_dir)
print(f"     {msg_qlearning.split(chr(10))[0]}")
plot_value_map(Q.max(axis=1), env.n_tendencies, env.n_positions, save_dir=run_dir,
              title="Mapa de Valores — Q-learning V*", method = "qlearning")
plot_policy(pi_ql, env.n_tendencies, env.n_positions, save_dir=run_dir,
           title="Política Ótima — Q-learning", method = "qlearning")
plot_trajectory(env, pi_ql, save_dir=run_dir, n_steps=50)
print("     ✓ learning_curve, value_map, policy, trajectory → PNG")

print("\n[7.5/8] Gerando COMPARAÇÕES...")
msg_comparison = plot_bellman_vs_qlearning(
    V_star, Q, env.n_tendencies, env.n_positions, save_dir=run_dir
)
print(f"     {msg_comparison}")

msg_policy_agreement = plot_policies_comparison(
    pi_star, pi_ql, env.n_tendencies, env.n_positions, save_dir=run_dir
)
print(f"     {msg_policy_agreement}")
print("     ✓ bellman_vs_qlearning_values, policies_comparison → PNG")

# ================= EXPERIMENTOS AVANÇADOS =================
print("\n[8/8] Rodando experimentos comparativos...")
print("     • Comparando impact de γ (gamma)...")
compare_gammas(env, save_dir=run_dir, gammas=[0.7, 0.9, 0.99], n_episodes=1500)
print("       ✓ gamma_comparison.png")

print("     • Comparando estratégias de exploração (ε)...")
compare_epsilons(env, save_dir=run_dir, n_episodes=2000)
print("       ✓ epsilon_comparison.png")

# ================= RELATÓRIO FINAL =================
print("\n" + "=" * 70)
print("  RESUMO FINAL")
print("=" * 70)

report = generate_metrics_report(
    V_star, pi_star, n_iter, vi_history,
    Q, rewards,
    env.n_tendencies, env.n_positions,
    save_dir=run_dir, env=env
)

print("\n✅ EXECUÇÃO COMPLETA!")
print(f"\n📁 Todos os arquivos em: {run_dir}")
print("\n📊 Visualizações geradas:")
print("   ✓ bellman_convergence.png        (Convergência Value Iteration)")
print("   ✓ learning_curve.png             (Curva de aprendizado Q-learning)")
print("   ✓ bellman_vs_qlearning_values.png (Comparação de valores)")
print("   ✓ policies_comparison.png        (Comparação de políticas)")
print("   ✓ value_map.png                  (Mapa de valores Q-learning)")
print("   ✓ policy_map.png                 (Política Q-learning)")
print("   ✓ trajectory.png                 (Trajetória do agente)")
print("   ✓ gamma_comparison.png           (Impacto de γ)")
print("   ✓ epsilon_comparison.png         (Impacto de ε)")
print("   ✓ metrics_report.txt             (Métricas em texto)")
print("\n" + "=" * 70)
