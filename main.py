from environment import FinancialMDP, MarketActions, MarketPositions, MarketTendency
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
    compare_alphas,
    evaluate_policy,
    create_run_dir,
    generate_metrics_report
)
import numpy as np
import time

print("=" * 70)
print("  AGENTE INTELIGENTE DE DECISÃO FINANCEIRA — APRENDIZADO POR REFORÇO")
print("=" * 70)

dataOrigin = int(input('Selecione a origem dos dados:\n'+'1. Dados reais\n2. Dados fictícios\n\n'))

# Constantes
TICKER = 'IBM'
START  = '2018-01-01'
END    = '2020-01-01'

# ================= SETUP =================
print("\n[1/8] Criando diretório de execução...")
run_dir = create_run_dir()
print(f"     ✓ {run_dir}")

print("\n[2/8] Baixando e preparando dados...")
transition = build_from_ticker(TICKER, start=START, end=END) if dataOrigin == 1 else None

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
Q, rewards, epsilons = qlearning(env, n_episodes=10000, gamma=0.9)
pi_ql = np.zeros(env.n_states, dtype=int)
for state in range(env.n_states):
    valid = env.valid_actions(FinancialMDP.decode_state(state)[1])
    pi_ql[state] = max(valid, key=lambda a: Q[state, a])
print(f"     ✓ Treinado em 10000 episódios")

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
print("     • Comparando impacto de γ (gamma)...")
compare_gammas(env, save_dir=run_dir, gammas=[0.3, 0.6, 0.99], n_episodes=10000)
print("       ✓ gamma_comparison.png")

print("     • Comparando estratégias de exploração (ε)...")
compare_epsilons(env, save_dir=run_dir, n_episodes=10000)
print("       ✓ epsilon_comparison.png")

print("     • Comparando impacto de α (alpha)...")
compare_alphas(env, save_dir=run_dir, n_episodes=10000)
print("       ✓ alpha_comparison.png")

# ================= EXPERIMENTOS COM OUTRAS ESTRATÉGIAS =================

def ql_policy_fn(state, valid):
    return int(pi_ql[state])

def bellman_policy_fn(state, valid):
    return int(pi_star[state])

def sintetic_policy_fn(state, valid):
    mar, pos = FinancialMDP.decode_state(state)
    
    if mar == MarketTendency.FALLING and pos == MarketPositions.BOUGHT:
        return MarketActions.SELL.value
    elif mar == MarketTendency.STABLE and pos == MarketPositions.NO_POSITION:
        return MarketActions.BUY.value
    elif mar == MarketTendency.RISING and pos == MarketPositions.NO_POSITION:
        return MarketActions.BUY.value
    
    return MarketActions.KEEP.value

def cash_policy_fn(state, valid):
    return int(MarketActions.KEEP.value)

def buy_hold_policy_fn(state, valid):
    _, pos = FinancialMDP.decode_state(state)
    return int(MarketActions.BUY.value if pos == MarketPositions.NO_POSITION else MarketActions.KEEP.value)

def random_policy_fn(state, valid):
    return int(np.random.choice(valid))

ql_mean, ql_std = evaluate_policy(env, ql_policy_fn)
bellman_mean, bellman_std = evaluate_policy(env, bellman_policy_fn)
sintetic_mean, sintetic_std = evaluate_policy(env, sintetic_policy_fn)
cash_mean, cash_std = evaluate_policy(env, cash_policy_fn)
bhld_mean, bhld_std = evaluate_policy(env, buy_hold_policy_fn)
rnd_mean, rnd_std = evaluate_policy(env, random_policy_fn)

print("\n[TEST] Comparação de políticas")
print(f"  Q-learning   : {ql_mean:+.4f} ± {ql_std:.4f}")
print(f"  Bellman      : {bellman_mean:+.4f} ± {bellman_std:.4f}")
print(f"  Sintetic     : {sintetic_mean:+.4f} ± {sintetic_std:.4f}")
print(f"  Cash-only    : {cash_mean:+.4f} ± {cash_std:.4f}")
print(f"  Buy & Hold   : {bhld_mean:+.4f} ± {bhld_std:.4f}")
print(f"  Random       : {rnd_mean:+.4f} ± {rnd_std:.4f}")

# ================= RELATÓRIO FINAL =================
print("\n" + "=" * 70)
print("  RESUMO FINAL")
print("=" * 70)

report = generate_metrics_report(
    V_star, pi_star, n_iter, vi_history,
    Q, rewards,
    ql_mean, ql_std,
    bellman_mean, bellman_std,
    sintetic_mean, sintetic_std,
    cash_mean, cash_std,
    bhld_mean, bhld_std,
    rnd_mean, rnd_std,
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
print("   ✓ alpha_comparison.png           (Impacto de α)")
print("   ✓ metrics_report.txt             (Métricas em texto)")
print("\n" + "=" * 70)
