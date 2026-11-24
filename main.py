"""
main.py - Ekzekuton të gjithë eksperimentin nga fillimi në fund.

    python main.py

Hapat:
  1. Ngarkimi / gjenerimi i dataset-it dhe ndarja train/test (80/20)
  2. Normalizimi i hyrjeve në [0, 1]
  3. 5-fold cross-validation për zgjedhjen e numrit të neuroneve k
  4. Eksperimenti k × σ (heatmap)
  5. Trajnimi i modelit përfundimtar RBF + krahasimi me Linear dhe MLP
  6. Ruajtja e tabelave, grafikëve dhe modelit
"""

import os
import numpy as np
import pandas as pd

from src.dataset import (ngarko_dataset, ndaj_train_test, k_fold_ndarjet,
                         NormalizuesMinMax, nota_e_vertete, HYRJET, KUFIJTE)
from src.rbf import RrjetaRBF
from src.modelet_krahasuese import RegresioniLinear, krijo_mlp
from src.metrikat import mse, te_gjitha
from src import vizualizimi as viz

SEED = 42                                    # seed fiks => rezultate të përsëritshme
DOSJA = os.path.dirname(os.path.abspath(__file__))
RRUGA_DATA = os.path.join(DOSJA, "data", "studentet.csv")
DOSJA_REZ = os.path.join(DOSJA, "rezultatet")

K_LISTA = [2, 5, 10, 15, 20, 30, 40, 60, 80, 100, 150, 200]   # numri i neuroneve
SIGMA_LISTA = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, None]           # None = formula automatike


def etiketa_sigma(s):
    return "auto" if s is None else str(s)


def cv_mse(X, y, k, sigma, n_palosje=5):
    """
    5-fold cross-validation: trajnon 5 herë, çdo herë me 4/5 e të dhënave,
    dhe mat gabimin në 1/5 e mbetur. Kthen MSE mesatare të validimit.
    """
    gabimet = []
    for i_train, i_val in k_fold_ndarjet(len(X), n_palosje, seed=SEED):
        m = RrjetaRBF(k=k, sigma=sigma, seed=SEED).fit(X[i_train], y[i_train])
        gabimet.append(mse(y[i_val], m.predict(X[i_val])))
    return float(np.mean(gabimet))


def main():
    np.random.seed(SEED)
    os.makedirs(DOSJA_REZ, exist_ok=True)

    # ------------------------------------------------------------------ 1
    print("1) Ngarkimi i të dhënave")
    X, y = ngarko_dataset(RRUGA_DATA, n=500, seed=SEED)
    X_train, X_test, y_train, y_test = ndaj_train_test(X, y, 0.2, seed=SEED)
    print(f"  {len(X)} studentë -> {len(X_train)} për trajnim, {len(X_test)} për testim")

    # ------------------------------------------------------------------ 2
    norm = NormalizuesMinMax().fit(X_train)          # vetëm nga train!
    Xn_train, Xn_test = norm.transform(X_train), norm.transform(X_test)

    # ------------------------------------------------------------------ 3
    print("\n2) Eksperimenti k × σ: MSE e validimit me 5-fold CV për çdo kombinim")
    matrica = np.zeros((len(K_LISTA), len(SIGMA_LISTA)))
    for i, k in enumerate(K_LISTA):
        for j, s in enumerate(SIGMA_LISTA):
            matrica[i, j] = cv_mse(Xn_train, y_train, k, s)
    etiketat = [etiketa_sigma(s) for s in SIGMA_LISTA]
    tabela_ks = pd.DataFrame(matrica, index=K_LISTA, columns=etiketat)
    tabela_ks.index.name = "k"
    tabela_ks.round(4).to_csv(os.path.join(DOSJA_REZ, "tabela_k_sigma.csv"))
    print(tabela_ks.round(4).to_string())

    # Kolona e fundit e matricës = σ automatike (formula d_max / sqrt(2k))
    j_auto = SIGMA_LISTA.index(None)
    k_auto = K_LISTA[int(np.argmin(matrica[:, j_auto]))]
    print(f"  => me σ automatike, k optimal sipas CV = {k_auto}")
    i_min, j_min = np.unravel_index(np.argmin(matrica), matrica.shape)
    k_grid, sigma_grid = K_LISTA[i_min], SIGMA_LISTA[j_min]
    print(f"  => kombinimi më i mirë: k = {k_grid}, σ = {etiketa_sigma(sigma_grid)}"
          f"  (CV MSE = {matrica[i_min, j_min]:.4f})")

    # ------------------------------------------------------------------ 4
    print("\n3) Gabimi sipas numrit të neuroneve (trajnim / CV / testim)")
    rreshtat_k = []
    for sigma in [None, sigma_grid]:
        j = SIGMA_LISTA.index(sigma)
        for i, k in enumerate(K_LISTA):
            m = RrjetaRBF(k=k, sigma=sigma, seed=SEED).fit(Xn_train, y_train)
            rreshtat_k.append({"sigma_lloji": etiketa_sigma(sigma), "k": k,
                               "sigma": round(m.sigma, 4),
                               "MSE_trajnim": mse(y_train, m.predict(Xn_train)),
                               "MSE_validim_CV": matrica[i, j],
                               "MSE_testim": mse(y_test, m.predict(Xn_test))})
    tabela_k = pd.DataFrame(rreshtat_k)
    tabela_k.round(4).to_csv(os.path.join(DOSJA_REZ, "tabela_neuronet.csv"), index=False)
    print(tabela_k.round(4).to_string(index=False))

    # ------------------------------------------------------------------ 5
    print("\n4) Trajnimi i modeleve përfundimtare dhe krahasimi")
    rbf_auto = RrjetaRBF(k=k_auto, seed=SEED).fit(Xn_train, y_train)
    rbf_grid = RrjetaRBF(k=k_grid, sigma=sigma_grid, seed=SEED).fit(Xn_train, y_train)
    linear = RegresioniLinear().fit(Xn_train, y_train)
    mlp = krijo_mlp(SEED).fit(Xn_train, y_train)

    # Modeli përfundimtar RBF = kombinimi me gabimin më të vogël në CV
    # (zgjedhja bëhet VETËM me CV, bashkësia e testimit nuk preket)
    rbf = rbf_grid

    modelet = {
        f"RBF (k={k_auto}, σ auto={rbf_auto.sigma:.3f})": rbf_auto,
        f"RBF (k={k_grid}, σ={etiketa_sigma(sigma_grid)})": rbf_grid,
        "Regresioni linear": linear,
        "MLP (sklearn, 20 neurone)": mlp,
    }
    rreshtat = []
    for emri, m in modelet.items():
        mt, ms = te_gjitha(y_train, m.predict(Xn_train)), te_gjitha(y_test, m.predict(Xn_test))
        rreshtat.append({"Modeli": emri,
                         "MSE_trajnim": mt["MSE"], "MAE_trajnim": mt["MAE"], "R2_trajnim": mt["R2"],
                         "MSE_testim": ms["MSE"], "MAE_testim": ms["MAE"], "R2_testim": ms["R2"]})
    tabela = pd.DataFrame(rreshtat).round(4)
    tabela.to_csv(os.path.join(DOSJA_REZ, "tabela_rezultateve.csv"), index=False)
    print(tabela.to_string(index=False))

    # Ruajtja e modelit për app.py
    rbf.ruaj(os.path.join(DOSJA_REZ, "modeli_rbf.npz"), norm.min, norm.max)
    print(f"  Modeli RBF përfundimtar (k={rbf.k}, σ={rbf.sigma:.3f}) u ruajt në "
          f"rezultatet/modeli_rbf.npz")

    # ------------------------------------------------------------------ 6
    print("\n5) Vizatimi i grafikëve")
    R = lambda emri: os.path.join(DOSJA_REZ, emri)
    parashikimet = {"RBF": rbf.predict(Xn_test), "Linear": linear.predict(Xn_test),
                    "MLP": mlp.predict(Xn_test)}
    metrikat = {e: te_gjitha(y_test, p) for e, p in parashikimet.items()}

    viz.grafik_reale_vs_parashikuar(y_test, parashikimet, metrikat,
                                    R("1_nota_reale_vs_parashikuar.png"))
    panelet = []
    for sigma, k_opt in [(None, k_auto), (sigma_grid, k_grid)]:
        t = tabela_k[tabela_k["sigma_lloji"] == etiketa_sigma(sigma)]
        titulli = ("σ automatike = d_max/√(2k)" if sigma is None
                   else f"σ = {sigma} (më e mira sipas CV)")
        panelet.append((titulli, t["MSE_trajnim"].to_numpy(),
                        t["MSE_validim_CV"].to_numpy(), t["MSE_testim"].to_numpy(), k_opt))
    viz.grafik_gabimi_neuronet(K_LISTA, panelet, R("2_gabimi_sipas_neuroneve.png"))
    viz.grafik_heatmap(K_LISTA, etiketat, matrica, R("3_heatmap_k_sigma.png"))
    i_k = K_LISTA.index(k_grid)
    sigma_fikse = [s for s in SIGMA_LISTA if s is not None]
    sigma_auto_k = float(tabela_k[(tabela_k["sigma_lloji"] == "auto") &
                                  (tabela_k["k"] == k_grid)]["sigma"].iloc[0])
    viz.grafik_sigma(sigma_fikse, matrica[i_k, :len(sigma_fikse)], sigma_auto_k,
                     matrica[i_k, j_auto], k_grid, R("4_ndikimi_i_sigma.png"))
    funksionet = {"RBF": lambda Xb: rbf.predict(norm.transform(Xb)),
                  "Linear": lambda Xb: linear.predict(norm.transform(Xb)),
                  "MLP": lambda Xb: mlp.predict(norm.transform(Xb))}
    viz.grafik_efekti_hyrjeve(funksionet, X_train, HYRJET, KUFIJTE, nota_e_vertete,
                              R("5_efekti_i_hyrjeve.png"))
    viz.grafik_gaussiane_1d(R("6_funksionet_gaussiane_1d.png"), seed=SEED)
    viz.grafik_residualet(y_test, parashikimet, R("7_histogrami_residualeve.png"))

    print("\nEksperimenti përfundoi me sukses. Rezultatet janë në dosjen rezultatet/.")


if __name__ == "__main__":
    main()
