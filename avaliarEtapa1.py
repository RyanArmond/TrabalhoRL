import sys
sys.modules["tensorflow"] = None

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from src.core.envs.KuaiRec.KuaiEnv import KuaiEnv

# Criar ambiente de avaliação e carregar os modelos salvos
env = Monitor(KuaiEnv())
modelo_dqn = DQN.load("modelos/dqn", env=env)
modelo_ppo = PPO.load("modelos/ppo", env=env)

# Avaliar um Agente Aleatório por 100 episódios
recompensas_aleatorio = []
turnos_aleatorio = []
for ep in range(100):
    obs, _ = env.reset(seed=ep)
    done = False
    while not done:
        acao = env.action_space.sample()
        obs, recompensa, terminated, truncated, info = env.step(acao)
        done = terminated or truncated
    recompensas_aleatorio.append(info["cum_reward"])
    turnos_aleatorio.append(env.env.total_turn)

# Avaliar DQN e PPO por 100 episódios
recompensas_dqn, sessoes_dqn = evaluate_policy(modelo_dqn, env, n_eval_episodes=100, return_episode_rewards=True)
recompensas_ppo, sessoes_ppo = evaluate_policy(modelo_ppo, env, n_eval_episodes=100, deterministic=False, return_episode_rewards=True)

print("\n=== RESULTADOS DA AVALIAÇÃO (100 Episódios) ===")
print(f"Aleatório -> Recompensa: {np.mean(recompensas_aleatorio):.2f} ± {np.std(recompensas_aleatorio):.2f} | Turnos: {np.mean(turnos_aleatorio):.1f}")
print(f"DQN       -> Recompensa: {np.mean(recompensas_dqn):.2f} ± {np.std(recompensas_dqn):.2f} | Turnos: {np.mean(sessoes_dqn):.1f}")
print(f"PPO       -> Recompensa: {np.mean(recompensas_ppo):.2f} ± {np.std(recompensas_ppo):.2f} | Turnos: {np.mean(sessoes_ppo):.1f}")

# Ler os logs do Monitor
df_dqn = pd.read_csv("logs/dqn.monitor.csv", skiprows=1)
df_ppo = pd.read_csv("logs/ppo.monitor.csv", skiprows=1)

# Calcular o timestep acumulado somando a coluna 'l' (duração de cada episódio)
df_dqn["timesteps"] = df_dqn["l"].cumsum()
df_ppo["timesteps"] = df_ppo["l"].cumsum()

# Suavizar as curvas com média móvel de 50 episódios para visualizar a tendência
janela = 50
df_dqn["r_media"] = df_dqn["r"].rolling(janela, min_periods=1).mean()
df_ppo["r_media"] = df_ppo["r"].rolling(janela, min_periods=1).mean()

df_dqn["l_media"] = df_dqn["l"].rolling(janela, min_periods=1).mean()
df_ppo["l_media"] = df_ppo["l"].rolling(janela, min_periods=1).mean()

# 5. Plotar os gráficos de Recompensa e Duração da Sessão
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(df_dqn["timesteps"], df_dqn["r_media"], label="DQN", color="tab:blue")
axes[0].plot(df_ppo["timesteps"], df_ppo["r_media"], label="PPO", color="tab:orange")
axes[0].set_title("Curva de Aprendizado — Recompensa Acumulada")
axes[0].set_xlabel("Timesteps")
axes[0].set_ylabel("Recompensa Média (Média Móvel)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(df_dqn["timesteps"], df_dqn["l_media"], label="DQN", color="tab:blue")
axes[1].plot(df_ppo["timesteps"], df_ppo["l_media"], label="PPO", color="tab:orange")
axes[1].set_title("Curva de Aprendizado — Duração da Sessão (Turnos)")
axes[1].set_xlabel("Timesteps")
axes[1].set_ylabel("Turnos por Episódio (Média Móvel)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("grafico_etapa1.png", dpi=150)
print("\nGráfico salvo em 'grafico_etapa1.png'!")
