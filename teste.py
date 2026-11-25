import sys
sys.modules["tensorflow"] = None

from stable_baselines3.common.env_checker import check_env
from src.core.envs.KuaiRec.KuaiEnv import KuaiEnv

for modo in ["original", "ultimo", "ultimos_5", "media_5"]:
    env = KuaiEnv(obs_mode=modo)
    check_env(env)
    obs, _ = env.reset(seed=42)
    print(f"obs_mode={modo:<8} | shape={obs.shape} | dtype={obs.dtype} | check_env=OK")