"""
rbf.py - Rrjeta RBF (Radial Basis Function) e implementuar nga e para.

Arkitektura me 3 shtresa:
  1. Shtresa hyrëse:  merr vektorin x = (studimi, gjumi, prezenca)
  2. Shtresa e fshehur: k neurone Gaussiane
         phi_j(x) = exp( -||x - c_j||^2 / (2 * sigma^2) )
     ku c_j është qendra e neuronit j (nga K-Means).
  3. Shtresa dalëse: kombinim LINEAR i daljeve të fshehura
         y(x) = w_0 + sum_j  w_j * phi_j(x)

Trajnimi bëhet në dy faza:
  - Faza 1 (pa mbikëqyrje): qendrat me K-Means, sigma me formulë
  - Faza 2 (me mbikëqyrje):  peshat me pseudo-inverse  W = Phi^+ * y
"""

import numpy as np
from .kmeans import kmeans


class RrjetaRBF:

    def __init__(self, k=10, sigma=None, seed=42):
        self.k = k            # numri i neuroneve në shtresën e fshehur
        self.sigma = sigma    # None => llogaritet automatikisht me formulë
        self.seed = seed

    def _aktivizimet(self, X):
        """Llogarit matricën Phi (n x (k+1)): daljet e neuroneve Gaussiane + bias."""
        # Distanca në katror nga çdo pikë te çdo qendër
        d2 = np.sum((X[:, None, :] - self.qendrat[None, :, :]) ** 2, axis=2)
        phi = np.exp(-d2 / (2.0 * self.sigma ** 2))
        # Shtojmë kolonën e njësheve për peshën bias w_0
        return np.hstack([np.ones((len(X), 1)), phi])

    def fit(self, X, y):
        # --- Faza 1a: qendrat me K-Means ---
        self.qendrat, _ = kmeans(X, self.k, seed=self.seed)

        # --- Faza 1b: gjerësia sigma ---
        if self.sigma is None:
            # Formula: sigma = d_max / sqrt(2k),
            # d_max = distanca më e madhe midis dy qendrave.
            # Kështu "kambanat" nuk janë as shumë të ngushta, as shumë të gjera.
            d = np.linalg.norm(self.qendrat[:, None] - self.qendrat[None, :], axis=2)
            d_max = d.max()
            self.sigma = d_max / np.sqrt(2 * self.k) if d_max > 0 else 1.0

        # --- Faza 2: peshat me pseudo-inverse (zgjidhja e katrorëve më të vegjël) ---
        Phi = self._aktivizimet(X)
        self.peshat = np.linalg.pinv(Phi) @ y
        return self

    def predict(self, X):
        return self._aktivizimet(X) @ self.peshat

    # ---------- ruajtja / ngarkimi i modelit ----------
    def ruaj(self, rruga, norm_min, norm_max):
        """Ruan modelin (dhe parametrat e normalizimit) në një skedar .npz."""
        np.savez(rruga, qendrat=self.qendrat, sigma=self.sigma, peshat=self.peshat,
                 k=self.k, norm_min=norm_min, norm_max=norm_max)

    @staticmethod
    def ngarko(rruga):
        """Ngarkon modelin e ruajtur. Kthen (modeli, norm_min, norm_max)."""
        t = np.load(rruga)
        m = RrjetaRBF(k=int(t["k"]), sigma=float(t["sigma"]))
        m.qendrat, m.peshat = t["qendrat"], t["peshat"]
        return m, t["norm_min"], t["norm_max"]
