import os
import sys
sys.modules["tensorflow"] = None

from src.core.envs.KuaiRec.KuaiEnv import KuaiEnv
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor

os.makedirs("logs", exist_ok=True)
os.makedirs("modelos", exist_ok=True)

# Treinamento DQN
env_dqn = Monitor(KuaiEnv(), filename="logs/dqn")

modelo_dqn = DQN(
    policy="MlpPolicy",
    env=env_dqn,
)

print("Treinando DQN...")
modelo_dqn.learn(total_timesteps=100000)
modelo_dqn.save("modelos/dqn")


# Treinamento PPO

env_ppo = Monitor(KuaiEnv(), filename="logs/ppo")

modelo_ppo = PPO(
    policy="MlpPolicy",
    env=env_ppo,
)

print("Treinando PPO...")
modelo_ppo.learn(total_timesteps=100000)
modelo_ppo.save("modelos/ppo")

print("Treinamento concluído e modelos salvos")