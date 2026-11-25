import os
import sys
import time

sys.modules["tensorflow"] = None

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from src.core.envs.KuaiRec.KuaiEnv import KuaiEnv

os.makedirs("logs", exist_ok=True)
os.makedirs("modelos", exist_ok=True)

def mask_fn(env):
    return env.unwrapped.action_masks()

modos = ["ultimo", "ultimos_5", "media_5"]

for modo in modos:
    print(f"\n=== Treinando MaskabelPPO | obs_mode='{modo}' ===")
    inicio = time.time()

    env = KuaiEnv(obs_mode=modo)
    print(f"Dimensão da observação: {env.observation_space.shape}")
    env = Monitor(env, filename=f"logs/mppo_{modo}")
    env = ActionMasker(env, mask_fn)

    modelo = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        n_steps=1024,
        batch_size=128,
        ent_coef=0.01,
        seed=7,
    )

    modelo.learn(total_timesteps=20000)
    modelo.save(f"modelos/mppo_{modo}")
    
    duracao = time.time() - inicio
    print(f"Concluído em {duracao:.1f}s -> salvo em modelos/mppo_{modo}.zip")

print("\nTodos os 3 modelos da Etapa 2 foram treinados e salvos!")