"""
dataset.py - Gjenerimi, ngarkimi dhe përgatitja e të dhënave.

Dataset-i është SINTETIK: notat llogariten me një formulë jo-lineare
(që ne e njohim) plus pak zhurmë të rastësishme. Kjo na lejon të shohim
nëse rrjeta RBF arrin ta "zbulojë" lidhjen e fshehur.
"""

import os
import numpy as np
import pandas as pd

# Emrat e kolonave (hyrjet dhe dalja)
HYRJET = ["oret_studimit", "oret_gjumit", "prezenca"]
DALJA = "nota"

# Kufijtë e secilës hyrje (përdoren edhe në app.py për slider-at)
KUFIJTE = {
    "oret_studimit": (0.0, 20.0),   # orë në javë
    "oret_gjumit": (4.0, 10.0),     # orë në natë
    "prezenca": (30.0, 100.0),      # përqindje e ligjëratave
}


def nota_e_vertete(studimi, gjumi, prezenca):
    """
    Formula e "vërtetë" (pa zhurmë) që lidh hyrjet me notën.
    Është qëllimisht JO-LINEARE:
      - studimi: efekt që ngopet (pas ~10 orësh, çdo orë shtesë ndihmon pak)
      - gjumi:   formë "U e përmbysur" - optimumi rreth 7.5 orë
      - prezenca: efekt pothuajse linear
    """
    efekti_studimit = 3.0 * (1.0 - np.exp(-studimi / 6.0))      # 0 ... ~2.9
    efekti_gjumit = 1.0 - 0.15 * (gjumi - 7.5) ** 2              # ~-0.8 ... 1
    efekti_prezences = 1.5 * (prezenca - 30.0) / 70.0            # 0 ... 1.5
    nota = 5.0 + efekti_studimit + efekti_gjumit + efekti_prezences
    return np.clip(nota, 5.0, 10.0)   # notat janë nga 5 deri në 10


def gjenero_dataset(n=500, seed=42, zhurma=0.3):
    """Gjeneron n studentë me hyrje të rastësishme dhe notë me zhurmë."""
    rng = np.random.default_rng(seed)
    studimi = rng.uniform(*KUFIJTE["oret_studimit"], n)
    gjumi = rng.uniform(*KUFIJTE["oret_gjumit"], n)
    prezenca = rng.uniform(*KUFIJTE["prezenca"], n)

    # Nota = formula e vërtetë + zhurmë Gaussiane (faktorë që nuk i njohim)
    nota = nota_e_vertete(studimi, gjumi, prezenca) + rng.normal(0, zhurma, n)
    nota = np.clip(nota, 5.0, 10.0)

    return pd.DataFrame({
        "oret_studimit": studimi.round(2),
        "oret_gjumit": gjumi.round(2),
        "prezenca": prezenca.round(1),
        "nota": nota.round(2),
    })


def ngarko_dataset(rruga, n=500, seed=42):
    """Ngarkon CSV-në; nëse nuk ekziston, e gjeneron dhe e ruan."""
    if not os.path.exists(rruga):
        os.makedirs(os.path.dirname(rruga), exist_ok=True)
        gjenero_dataset(n, seed).to_csv(rruga, index=False)
        print(f"  Dataset-i u gjenerua dhe u ruajt në: {rruga}")
    df = pd.read_csv(rruga)
    X = df[HYRJET].to_numpy(dtype=float)
    y = df[DALJA].to_numpy(dtype=float)
    return X, y


def ndaj_train_test(X, y, pjesa_test=0.2, seed=42):
    """Ndan të dhënat rastësisht në trajnim (80%) dhe testim (20%)."""
    rng = np.random.default_rng(seed)
    indekset = rng.permutation(len(X))
    n_test = int(len(X) * pjesa_test)
    i_test, i_train = indekset[:n_test], indekset[n_test:]
    return X[i_train], X[i_test], y[i_train], y[i_test]


def k_fold_ndarjet(n, k=5, seed=42):
    """
    Kthen listën e k çifteve (indekset_train, indekset_validim)
    për k-fold cross-validation. Çdo mostër del në validim saktësisht një herë.
    """
    rng = np.random.default_rng(seed)
    indekset = rng.permutation(n)
    palosjet = np.array_split(indekset, k)       # k grupe me madhësi të përafërt
    ndarjet = []
    for i in range(k):
        i_val = palosjet[i]
        i_train = np.concatenate([palosjet[j] for j in range(k) if j != i])
        ndarjet.append((i_train, i_val))
    return ndarjet


class NormalizuesMinMax:
    """
    Shndërron çdo hyrje në intervalin [0, 1]:  x' = (x - min) / (max - min).
    E rëndësishme për RBF, sepse distanca Euklidiane duhet t'i trajtojë
    të gjitha hyrjet njësoj (përndryshe prezenca 30-100 do të dominonte gjumin 4-10).
    Min dhe max llogariten VETËM nga të dhënat e trajnimit.
    """

    def fit(self, X):
        self.min = X.min(axis=0)
        self.max = X.max(axis=0)
        return self

    def transform(self, X):
        return (X - self.min) / (self.max - self.min)

    def fit_transform(self, X):
        return self.fit(X).transform(X)
