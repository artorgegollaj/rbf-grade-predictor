"""
modelet_krahasuese.py - Modelet me të cilat krahasojmë rrjetën RBF.

1. Regresioni linear  - nga e para (vetëm një vijë/plan, nuk kap jo-linearitetin)
2. MLP i vogël        - nga sklearn, VETËM si model referencë
"""

import numpy as np
from sklearn.neural_network import MLPRegressor


class RegresioniLinear:
    """y = w_0 + w_1*x_1 + w_2*x_2 + w_3*x_3, peshat me pseudo-inverse."""

    def fit(self, X, y):
        X1 = np.hstack([np.ones((len(X), 1)), X])   # kolona e bias-it
        self.peshat = np.linalg.pinv(X1) @ y
        return self

    def predict(self, X):
        X1 = np.hstack([np.ones((len(X), 1)), X])
        return X1 @ self.peshat


def krijo_mlp(seed=42):
    """
    MLP me 1 shtresë të fshehur (20 neurone, aktivizim tanh).
    Trajnohet me L-BFGS (metodë kuazi-Njuton), e cila për dataset-e të vogla
    konvergjon më mirë se Adam. Ndryshe nga RBF, këtu TË GJITHA peshat
    mësohen njëkohësisht me backpropagation (optimizim jo-linear iterativ).
    """
    return MLPRegressor(hidden_layer_sizes=(20,), activation="tanh", solver="lbfgs",
                        max_iter=5000, random_state=seed)
