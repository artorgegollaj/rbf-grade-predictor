"""
kmeans.py - Algoritmi K-Means i implementuar nga e para me NumPy.

Në rrjetën RBF, K-Means përdoret për të gjetur QENDRAT e neuroneve të
shtresës së fshehur: çdo qendër përfaqëson një "grup tipik" studentësh.
"""

import numpy as np


def kmeans(X, k, max_iter=100, seed=42):
    """
    Grupon pikat X (n x d) në k grupe.

    Hapat:
      1. Zgjedh k pika të rastësishme si qendra fillestare.
      2. Cakto çdo pikë te qendra më e afërt (distanca Euklidiane).
      3. Rillogarit çdo qendër si mesatare e pikave të grupit të saj.
      4. Përsërit 2-3 derisa qendrat të mos lëvizin më.

    Kthen: qendrat (k x d) dhe etiketat (grupi i secilës pikë).
    """
    rng = np.random.default_rng(seed)
    qendrat = X[rng.choice(len(X), size=k, replace=False)].copy()

    for _ in range(max_iter):
        # Hapi 2: distancat nga çdo pikë te çdo qendër -> matricë (n x k)
        distancat = np.linalg.norm(X[:, None, :] - qendrat[None, :, :], axis=2)
        etiketat = np.argmin(distancat, axis=1)

        # Hapi 3: qendrat e reja
        qendrat_e_reja = qendrat.copy()
        for j in range(k):
            pikat_e_grupit = X[etiketat == j]
            if len(pikat_e_grupit) > 0:
                qendrat_e_reja[j] = pikat_e_grupit.mean(axis=0)
            else:
                # Grup bosh: e zhvendosim qendrën te një pikë e rastësishme
                qendrat_e_reja[j] = X[rng.integers(len(X))]

        # Hapi 4: kriteri i ndalimit (konvergjenca)
        if np.allclose(qendrat, qendrat_e_reja):
            break
        qendrat = qendrat_e_reja

    return qendrat, etiketat
