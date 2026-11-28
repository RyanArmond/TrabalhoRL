import os
import sys
sys.modules["tensorflow"] = None

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from src.core.envs.KuaiRec.KuaiEnv import KuaiEnv

os.makedirs("logs", exist_ok=True)
os.makedirs("modelos", exist_ok=True)

# Função que pega as ações válidas, ActionMasker chama
def mask_fn(env):
    return env.unwrapped.action_masks()

# Ambiente com Wrapper Monitor e Wrapper ActionMasker
env = KuaiEnv()
env = Monitor(env, filename="logs/maskable_ppo")
env = ActionMasker(env, mask_fn)

# Instanciar e Treinar
modelo_mppo = MaskablePPO(
    policy="MlpPolicy",
    env=env,
)

print("Treinando MaskablePPO...")
modelo_mppo.learn(total_timesteps=100000)
modelo_mppo.save("modelos/maskable_ppo")
print("Treinamento do MaskablePPO concluído e modelo salvo!")