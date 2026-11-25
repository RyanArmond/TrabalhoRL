import os
import sys
import numpy as np
import itertools
from collections import Counter
from sklearn.preprocessing import MinMaxScaler

from src.core.envs.BaseEnv import BaseEnv
sys.path.extend(["./src", "./src/DeepCTR-Torch", "./src/tianshou"])

from src.core.envs.KuaiRec.KuaiData import KuaiData

# ROOTPATH = os.path.dirname(__file__)
ROOTPATH = "data/KuaiRec"
DATAPATH = os.path.join(ROOTPATH, "data_raw")


class KuaiEnv(BaseEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, mat=None, lbe_user=None, lbe_item=None, list_feat=None, df_dist_small=None,
                 num_leave_compute=5, leave_threshold=1, max_turn=100, random_init=False, obs_mode="original"):
        if mat is not None:
            self.mat = mat
            self.lbe_user = lbe_user
            self.lbe_item = lbe_item
            self.list_feat = list_feat
            self.df_dist_small = df_dist_small
        else:
            self.mat, self.lbe_user, self.lbe_item, self.list_feat, self.df_dist_small = self.load_env_data()

        self.list_feat_small = list(map(lambda x: self.list_feat[x], self.lbe_item.classes_))
        self._construir_matrizes_caracteristicas()
        # smallmat shape: (1411, 3327)
        super(KuaiEnv, self).__init__(num_leave_compute, leave_threshold, max_turn, random_init, obs_mode=obs_mode)

    @staticmethod
    def load_env_data():
        mat, lbe_user, lbe_item = KuaiData.load_mat()
        list_feat, df_feat = KuaiData.load_category()
        df_dist_small = None # KuaiData.get_saved_distance_mat(list_feat, lbe_item.classes_)
        return mat, lbe_user, lbe_item, list_feat, df_dist_small

    def render(self, mode='human', close=False):
        history_action = self.history_action
        category = {k: self.list_feat_small[v] for k, v in history_action.items()}
        # category_debug = {k:self.list_feat[v] for k,v in history_action.items()}
        # return history_action, category, category_debug
        return self.cur_user, history_action, category

    def _determine_whether_to_leave(self, t, action):
        # self.list_feat[action]
        if t == 0:
            return False
        window_actions = self.sequence_action[t - self.num_leave_compute:t]
        hist_categories_each = list(map(lambda x: self.list_feat_small[x], window_actions))

        # hist_set = set.union(*list(map(lambda x: self.list_feat[x], self.sequence_action[t - self.num_leave_compute:t-1])))

        hist_categories = list(itertools.chain(*hist_categories_each))
        hist_dict = Counter(hist_categories)
        category_a = self.list_feat_small[action]
        for c in category_a:
            if hist_dict[c] > self.leave_threshold:
                return True

        # if action in window_actions:
        #     return True

        return False

    # Método Auxiliar para construir as matrizes de características dos vídeos e dos usuários
    def _construir_matrizes_caracteristicas(self):
        # Características dos usuários: (1411, 25) normalizadas em [0.0, 1.0]
        df_usuario = KuaiData().load_user_feat(only_small=True)

        # Garante a mesma ordem de linhas
        df_usuario = df_usuario.loc[self.lbe_user.classes_]
        
        self.caracteristicas_usuario = MinMaxScaler().fit_transform(df_usuario.to_numpy()).astype(np.float32)

        # Características dos vídeos: (3327, 32) -> 31 categorias multi-hot + 1 duração
        duracao_video = KuaiData.load_video_duration()
        num_categorias = 31
        n_itens = len(self.list_feat_small)
        self.caracteristicas_video = np.zeros((n_itens, num_categorias + 1), dtype=np.float32)

        for i, orig_item_id in enumerate(self.lbe_item.classes_):
            # Liga (1.0) as posições correspondentes às categorias do vídeo i
            for c in self.list_feat_small[i]:
                self.caracteristicas_video[i,c] = 1.0
            
            # Coloca a duração normalizada na última posição
            self.caracteristicas_video[i, num_categorias] = float(duracao_video.loc[orig_item_id])