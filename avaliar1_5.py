import sys
sys.modules["tensorflow"] = None

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.evaluation import evaluate_policy as evaluate_maskable_policy

from src.core.envs.KuaiRec.KuaiEnv import KuaiEnv

def mask_fn(env):
    return env.unwrapped.action_masks()

# Ambiente
env_padrao = Monitor(KuaiEnv())
env_mask = ActionMasker(Monitor(KuaiEnv()), mask_fn)

# Carregar modelos
modelo_ppo = PPO.load("modelos/ppo", env=env_padrao)
modelo_mppo = MaskablePPO.load("modelos/maskable_ppo", env=env_mask)

# Avaliar Agente Aleatório
recompensas_aleatorio, turnos_aleatorio = [], []
for ep in range(100):
    obs, _ = env_padrao.reset(seed=ep)
    done = False
    while not done:
        acao = env_padrao.action_space.sample()
        obs, reward, terminated, truncated, info = env_padrao.step(acao)
        done = terminated or truncated
    recompensas_aleatorio.append(info["cum_reward"])
    turnos_aleatorio.append(env_padrao.unwrapped.total_turn)

# Avaliar PPO (apenas estocástico) e MaskablePPO (determinístico e estocástico)
recompensas_ppo, turnos_ppo = evaluate_policy(modelo_ppo, env_padrao, n_eval_episodes=100, deterministic=False, return_episode_rewards=True)

recompensas_mppo_estoc, turnos_mppo_estoc = evaluate_maskable_policy(modelo_mppo, env_mask, n_eval_episodes=100, deterministic=False, return_episode_rewards=True)

recompensas_mppo_det, turnos_mppo_det = evaluate_maskable_policy(modelo_mppo, env_mask, n_eval_episodes=100, deterministic=True, return_episode_rewards=True)

# Verificando se o Maskable PPO repete vídeo em um ep
obs, _ = env_mask.reset(seed=7)
done = False

while not done:
    mascara = env_mask.unwrapped.action_masks()
    acao, _ = modelo_mppo.predict(obs, action_masks=mascara, deterministic=True)
    obs, _, terminated, truncated, _ = env_mask.step(acao)
    done = terminated or truncated

videos_vistos = env_mask.unwrapped.sequence_action
repetidos = len(videos_vistos) - len(set(videos_vistos))
print(f"\n[Verificação de Máscara] Turnos no episódio teste: {len(videos_vistos)} | Vídeos repetidos: {repetidos}")

print("\n=== RESULTADOS DA AVALIAÇÃO (100 Episódios) ===")
print(f"Aleatório             -> Recompensa: {np.mean(recompensas_aleatorio):.2f} ± {np.std(recompensas_aleatorio):.2f} | Turnos: {np.mean(turnos_aleatorio):.1f}")
print(f"PPO (estocástico)     -> Recompensa: {np.mean(recompensas_ppo):.2f} ± {np.std(recompensas_ppo):.2f} | Turnos: {np.mean(turnos_ppo):.1f}")
print(f"MaskablePPO (estoc.)  -> Recompensa: {np.mean(recompensas_mppo_estoc):.2f} ± {np.std(recompensas_mppo_estoc):.2f} | Turnos: {np.mean(turnos_mppo_estoc):.1f}")
print(f"MaskablePPO (determ.) -> Recompensa: {np.mean(recompensas_mppo_det):.2f} ± {np.std(recompensas_mppo_det):.2f} | Turnos: {np.mean(turnos_mppo_det):.1f}")

# Gráfico comparativo das curvas de treino (PPO vs MaskablePPO)
df_ppo = pd.read_csv("logs/ppo.monitor.csv", skiprows=1)
df_mppo = pd.read_csv("logs/maskable_ppo.monitor.csv", skiprows=1)
df_ppo["timesteps"] = df_ppo["l"].cumsum()
df_mppo["timesteps"] = df_mppo["l"].cumsum()

janela = 50
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(df_ppo["timesteps"], df_ppo["r"].rolling(janela, min_periods=1).mean(), label="PPO (sem máscara)", color="tab:orange")
axes[0].plot(df_mppo["timesteps"], df_mppo["r"].rolling(janela, min_periods=1).mean(), label="MaskablePPO (com máscara)", color="tab:green")
axes[0].set_title("Curva de Aprendizado — Recompensa Acumulada")
axes[0].set_xlabel("Timesteps")
axes[0].set_ylabel("Recompensa Média (Média Móvel)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(df_ppo["timesteps"], df_ppo["l"].rolling(janela, min_periods=1).mean(), label="PPO (sem máscara)", color="tab:orange")
axes[1].plot(df_mppo["timesteps"], df_mppo["l"].rolling(janela, min_periods=1).mean(), label="MaskablePPO (com máscara)", color="tab:green")
axes[1].set_title("Curva de Aprendizado — Duração da Sessão (Turnos)")
axes[1].set_xlabel("Timesteps")
axes[1].set_ylabel("Turnos por Episódio (Média Móvel)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("grafico_treino_1_5.png", dpi=150)
print("\nGráfico salvo em 'grafico_etapa_1_5.png'!")

