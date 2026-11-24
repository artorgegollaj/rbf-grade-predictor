"""
metrikat.py - Metrikat e vlerësimit të modelit (nga e para).
"""

import numpy as np


def mse(y, y_pred):
    """Gabimi mesatar katror: mesatarja e (y - y_pred)^2. Ndëshkon fort gabimet e mëdha."""
    return float(np.mean((y - y_pred) ** 2))


def mae(y, y_pred):
    """Gabimi mesatar absolut: sa pikë note gabon modeli mesatarisht."""
    return float(np.mean(np.abs(y - y_pred)))


def r2(y, y_pred):
    """
    Koeficienti i përcaktimit R^2 = 1 - SS_res / SS_tot.
    1 = parashikim perfekt; 0 = sa të parashikosh gjithmonë mesataren.
    """
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - ss_res / ss_tot)


def te_gjitha(y, y_pred):
    return {"MSE": mse(y, y_pred), "MAE": mae(y, y_pred), "R2": r2(y, y_pred)}
