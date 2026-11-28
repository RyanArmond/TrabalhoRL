import sys
sys.modules["tensorflow"] = None

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3.common.monitor import Monitor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.evaluation import evaluate_policy as evaluate_maskable_policy

from src.core.envs.KuaiRec.KuaiEnv import KuaiEnv


def mask_fn(env):
    return env.unwrapped.action_masks()


# Configuração dos 4 experimentos do MaskablePPO
experimentos = [
    ("original",  "Original (IDs)",             "modelos/maskable_ppo", "logs/maskable_ppo.monitor.csv", "tab:gray"),
    ("ultimo",    "Usuário + Último Vídeo",     "modelos/mppo_ultimo",  "logs/mppo_ultimo.monitor.csv",  "tab:blue"),
    ("ultimos_5", "Usuário + Últimos 5 Vídeos", "modelos/mppo_ultimos_5", "logs/mppo_ultimos_5.monitor.csv", "tab:orange"),
    ("media_5",   "Usuário + Média 5 Vídeos",   "modelos/mppo_media_5", "logs/mppo_media_5.monitor.csv", "tab:green"),
]

# Avaliar Agente Aleatório por 100 episódios
env_aleatorio = Monitor(KuaiEnv(obs_mode="original"))
recompensas_aleatorio, turnos_aleatorio = [], []
for ep in range(100):
    obs, _ = env_aleatorio.reset(seed=ep)
    done = False
    while not done:
        acao = env_aleatorio.action_space.sample()
        obs, reward, terminated, truncated, info = env_aleatorio.step(acao)
        done = terminated or truncated
    recompensas_aleatorio.append(info["cum_reward"])
    turnos_aleatorio.append(env_aleatorio.unwrapped.total_turn)

print("\n=== RESULTADOS DA AVALIAÇÃO — ETAPA 2 (100 Episódios) ===")
print(f"{'Agente / Observação':<30} | {'Estocástico (Recomp. / Turnos)':<32} | {'Determinístico (Recomp. / Turnos)'}")
print("-" * 100)
print(
    f"{'Aleatório':<30} | "
    f"{np.mean(recompensas_aleatorio):6.2f} ± {np.std(recompensas_aleatorio):5.2f} / {np.mean(turnos_aleatorio):4.1f} turnos  | "
    f"-"
)

# Guardar métricas para o gráfico de barras da avaliação
resultados_avaliacao = {
    "rotulos": ["Aleatório"],
    "recompensas_estoc": [np.mean(recompensas_aleatorio)],
    "turnos_estoc": [np.mean(turnos_aleatorio)],
    "recompensas_det": [np.mean(recompensas_aleatorio)],
    "turnos_det": [np.mean(turnos_aleatorio)],
}

# Avaliar cada um dos 4 modelos MaskablePPO
for modo, rotulo, caminho_modelo, _, _ in experimentos:
    env_mask = ActionMasker(Monitor(KuaiEnv(obs_mode=modo)), mask_fn)
    modelo = MaskablePPO.load(caminho_modelo, env=env_mask)

    recompensas_estoc, turnos_estoc = evaluate_maskable_policy(
        modelo, env_mask, n_eval_episodes=100, deterministic=False, return_episode_rewards=True
    )
    recompensas_det, turnos_det = evaluate_maskable_policy(
        modelo, env_mask, n_eval_episodes=100, deterministic=True, return_episode_rewards=True
    )

    print(
        f"{rotulo:<30} | "
        f"{np.mean(recompensas_estoc):6.2f} ± {np.std(recompensas_estoc):5.2f} / {np.mean(turnos_estoc):4.1f} turnos  | "
        f"{np.mean(recompensas_det):6.2f} ± {np.std(recompensas_det):5.2f} / {np.mean(turnos_det):4.1f} turnos"
    )

    resultados_avaliacao["rotulos"].append(rotulo)
    resultados_avaliacao["recompensas_estoc"].append(np.mean(recompensas_estoc))
    resultados_avaliacao["turnos_estoc"].append(np.mean(turnos_estoc))
    resultados_avaliacao["recompensas_det"].append(np.mean(recompensas_det))
    resultados_avaliacao["turnos_det"].append(np.mean(turnos_det))

# Gráfico 1: Curvas de Treinamento (4 observações)
janela = 50
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

for _, rotulo, _, caminho_log, cor in experimentos:
    df = pd.read_csv(caminho_log, skiprows=1)
    df["timesteps"] = df["l"].cumsum()
    axes[0].plot(df["timesteps"], df["r"].rolling(janela, min_periods=1).mean(), label=rotulo, color=cor, linewidth=2)
    axes[1].plot(df["timesteps"], df["l"].rolling(janela, min_periods=1).mean(), label=rotulo, color=cor, linewidth=2)

axes[0].axhline(np.mean(recompensas_aleatorio), color="red", linestyle="--", linewidth=1.5, label=f"Aleatório ({np.mean(recompensas_aleatorio):.1f})")
axes[0].set_title("Curva de Treino (Etapa 2) — Recompensa Acumulada")
axes[0].set_xlabel("Timesteps")
axes[0].set_ylabel("Recompensa Média (Média Móvel)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].axhline(np.mean(turnos_aleatorio), color="red", linestyle="--", linewidth=1.5, label=f"Aleatório ({np.mean(turnos_aleatorio):.1f})")
axes[1].set_title("Curva de Treino (Etapa 2) — Duração da Sessão (Turnos)")
axes[1].set_xlabel("Timesteps")
axes[1].set_ylabel("Turnos por Episódio (Média Móvel)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("graficos/etapa2/grafico_treino_etapa2.png", dpi=150)

# Gráfico 2: Comparação dos 100 Episódios de Avaliação (Determinístico vs Estocástico)
x = np.arange(len(resultados_avaliacao["rotulos"]))
largura = 0.35

fig2, axes2 = plt.subplots(1, 2, figsize=(15, 5.5))

b1 = axes2[0].bar(x - largura / 2, resultados_avaliacao["recompensas_estoc"], largura, label="Estocástico", color="tab:blue")
b2 = axes2[0].bar(x + largura / 2, resultados_avaliacao["recompensas_det"], largura, label="Determinístico", color="tab:green")
axes2[0].set_title("Avaliação (100 Episódios) — Recompensa Acumulada Média")
axes2[0].set_xticks(x)
axes2[0].set_xticklabels(resultados_avaliacao["rotulos"], rotation=15, ha="right")
axes2[0].set_ylabel("Recompensa Média")
axes2[0].bar_label(b1, fmt="%.1f", padding=2, fontsize=8.5)
axes2[0].bar_label(b2, fmt="%.1f", padding=2, fontsize=8.5)
axes2[0].legend()
axes2[0].grid(axis="y", alpha=0.3)

b3 = axes2[1].bar(x - largura / 2, resultados_avaliacao["turnos_estoc"], largura, label="Estocástico", color="tab:blue")
b4 = axes2[1].bar(x + largura / 2, resultados_avaliacao["turnos_det"], largura, label="Determinístico", color="tab:green")
axes2[1].set_title("Avaliação (100 Episódios) — Duração Média da Sessão")
axes2[1].set_xticks(x)
axes2[1].set_xticklabels(resultados_avaliacao["rotulos"], rotation=15, ha="right")
axes2[1].set_ylabel("Turnos Médios")
axes2[1].bar_label(b3, fmt="%.1f", padding=2, fontsize=8.5)
axes2[1].bar_label(b4, fmt="%.1f", padding=2, fontsize=8.5)
axes2[1].legend()
axes2[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("graficos/etapa2/grafico_avaliacao_etapa2.png", dpi=150)

print("\nGráficos salvos em 'graficos/etapa2/grafico_treino_etapa2.png' e 'graficos/etapa2/grafico_avaliacao_etapa2.png'!")