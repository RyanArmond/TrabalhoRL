from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym
import random
import torch

class BaseEnv(ABC, gym.Env):
    def __init__(self, num_leave_compute, leave_threshold, max_turn, random_init, obs_mode="original"):
        
        self.max_turn = max_turn
        self.random_init = random_init
        self.obs_mode = obs_mode
        self.history_window = 5
        
        if self.obs_mode == "original":
            self.observation_space = gym.spaces.Box(
                low=np.array([0, 0], dtype=np.int32),
                high=np.array([len(self.mat) - 1, self.mat.shape[1]], dtype=np.int32),
                shape=(2,),
                dtype=np.int32,
            )
        else:
            usuario_dim = self.caracteristicas_usuario.shape[1] # 25
            video_dim = self.caracteristicas_video.shape[1] # 32
            video_min = float(np.min(self.caracteristicas_video)) - 1.0
            video_max = float(np.max(self.caracteristicas_video)) + 1.0

            if self.obs_mode in ("ultimo", "media_5"):
                n_video_quant = 1 # 25 + 32 = 57
            elif self.obs_mode == "ultimos_5":
                n_video_quant = self.history_window # 25 + 5 * 32 = 185
            else:
                raise ValueError(
                    f"obs_mode inválido: '{self.obs_mode}'. Use 'original', 'ultimo', 'ultimos_5' ou 'media_5'."
                )
            
            low = np.concatenate([
                np.zeros(usuario_dim, dtype=np.float32),
                np.full(n_video_quant * video_dim, video_min, dtype=np.float32),
            ])

            high = np.concatenate([
                np.ones(usuario_dim, dtype=np.float32),
                np.full(n_video_quant * video_dim, video_max, dtype=np.float32),
            ])

            self.observation_space = gym.spaces.Box(low=low, high=high, shape=low.shape, dtype=np.float32)

        self.action_space = gym.spaces.Discrete(self.mat.shape[1])

        self.num_leave_compute = num_leave_compute
        self.leave_threshold = leave_threshold

        self.reset()

    @property
    def state(self):
        if self.obs_mode == "original":
            return np.array([self.cur_user, self.action], dtype=np.int32)
         
        vetor_usuario = self.caracteristicas_usuario[self.cur_user] # 25 dimensões
        video_dim = self.caracteristicas_video.shape[1]

        if self.obs_mode == "ultimo":
            if len(self.sequence_action) > 0:
                vetor_video = self.caracteristicas_video[self.sequence_action[-1]]
            else:
                vetor_video = np.zeros(video_dim, dtype=np.float32)
            return np.concatenate([vetor_usuario, vetor_video]).astype(np.float32)
        
        elif self.obs_mode == "ultimos_5":
            recentes = self.sequence_action[-self.history_window:] # pega até os últimos 5 vídeos
            
            # Coloca o mais recente primeiro
            vetores = [self.caracteristicas_video[vid] for vid in reversed(recentes)]

            # Completa com vetores de zeros se ainda tiver menos que 5 vídeos
            while len(vetores) < self.history_window:
                vetores.append(np.zeros(video_dim, dtype=np.float32))
            return np.concatenate([vetor_usuario] + vetores).astype(np.float32)

        elif self.obs_mode == "media_5":
            recentes = self.sequence_action[-self.history_window:]
            if len(recentes) > 0:
                vetor_media = np.mean(self.caracteristicas_video[recentes], axis=0)
            else:
                vetor_media = np.zeros(video_dim, dtype=np.float32)
            return np.concatenate([vetor_usuario, vetor_media]).astype(np.float32)

    def __user_generator(self):
        user = random.randint(0, len(self.mat) - 1)
        return user
    
    def __item_generator(self):
        if self.random_init:
            item = random.randint(0, len(self.mat[0]) - 1)
        else:
            item = len(self.mat[0])
        return item

    def step(self, action):
        action = int(action)
        self.action = action
        t = self.total_turn
        terminated = self._determine_whether_to_leave(t, action)
        if t >= (self.max_turn - 1):
            terminated = True
        self._add_action_to_history(t, action)

        reward = float(self.mat[self.cur_user, action])

        self.cum_reward += reward
        self.total_turn += 1

        truncated = False

        return self.state, reward, terminated, truncated, {'cum_reward': self.cum_reward}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.cum_reward = 0
        self.total_turn = 0
        self.cur_user = self.__user_generator()
        self.action = self.__item_generator()
        self._reset_history()

        # return self.state, {'key': 1, 'env': self}
        return self.state, {'cum_reward': 0.0}
    

    def _reset_history(self):
        self.history_action = {}
        self.sequence_action = []
        self.max_history = 0
        self.valid_actions = np.ones(self.mat.shape[1], dtype=bool)

    def _add_action_to_history(self, t, action):
        self.sequence_action.append(action)
        self.history_action[t] = action
        self.valid_actions[action] = False
        assert self.max_history == t
        self.max_history += 1

    def action_masks(self) -> np.ndarray:
        return self.valid_actions

    def seed(self, sd=0):
        torch.manual_seed(sd)
