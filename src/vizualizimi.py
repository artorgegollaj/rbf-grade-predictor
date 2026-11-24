"""
vizualizimi.py - Të gjithë grafikët e projektit (tituj dhe etiketa në shqip).
Çdo funksion ruan një skedar PNG në dosjen rezultatet/.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")            # vizatim pa dritare (vetëm ruajtje në skedar)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from .rbf import RrjetaRBF

# Ngjyrat: çdo model ka gjithmonë të njëjtën ngjyrë në të gjithë grafikët
NGJYRAT = {"RBF": "#2a78d6", "Linear": "#eb6834", "MLP": "#1baf7a"}
NGJYRA_E_VERTETE = "#52514e"     # gri e errët për kurbën e "vërtetë"
NGJYRA_GRID = "#e1e0d9"
# Shkallë sekuenciale blu (e çelët -> e errët) për heatmap-in
BLU_SEKUENCIALE = LinearSegmentedColormap.from_list(
    "blu", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c3c2b7", "axes.grid": True,
    "grid.color": NGJYRA_GRID, "grid.linewidth": 0.6,
    "axes.titleweight": "bold", "legend.frameon": False,
})


def _ruaj(fig, rruga):
    fig.tight_layout()
    fig.savefig(rruga, bbox_inches="tight")
    plt.close(fig)
    print(f"  Grafiku u ruajt: {rruga}")


# ---------------------------------------------------------------------------
# 1. Nota reale vs. e parashikuar
# ---------------------------------------------------------------------------
def grafik_reale_vs_parashikuar(y_test, parashikimet, metrikat, rruga):
    fig, boshtet = plt.subplots(1, len(parashikimet), figsize=(13, 4.4), sharey=True)
    for ax, (emri, y_pred) in zip(boshtet, parashikimet.items()):
        ax.plot([5, 10], [5, 10], "--", color=NGJYRA_E_VERTETE, lw=1,
                label="Parashikim perfekt (y = x)")
        ax.scatter(y_test, y_pred, s=22, color=NGJYRAT[emri], alpha=0.75,
                   edgecolor="white", linewidth=0.6)
        m = metrikat[emri]
        ax.set_title(f"{emri}\nMSE = {m['MSE']:.3f} · MAE = {m['MAE']:.3f} · R² = {m['R2']:.3f}",
                     fontsize=10)
        ax.set_xlabel("Nota reale")
        ax.set_xlim(4.8, 10.2)
        ax.set_ylim(4.8, 10.2)
        ax.set_aspect("equal")
    boshtet[0].set_ylabel("Nota e parashikuar")
    boshtet[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Nota reale vs. nota e parashikuar (bashkësia e testimit)",
                 fontweight="bold")
    _ruaj(fig, rruga)


# ---------------------------------------------------------------------------
# 2. Gabimi sipas numrit të neuroneve
# ---------------------------------------------------------------------------
def grafik_gabimi_neuronet(k_lista, panelet, rruga):
    """
    panelet: listë me (titulli, mse_train, mse_cv, mse_test, k_optimal),
    një panel për çdo vlerë të σ.
    """
    fig, boshtet = plt.subplots(1, len(panelet), figsize=(14, 5), sharey=True)
    for ax, (titulli, mse_train, mse_cv, mse_test, k_opt) in zip(boshtet, panelet):
        ax.plot(k_lista, mse_train, "o-", color="#2a78d6", lw=2, ms=6, label="Trajnim")
        ax.plot(k_lista, mse_cv, "s-", color="#eb6834", lw=2, ms=6,
                label="Validim (5-fold CV)")
        ax.plot(k_lista, mse_test, "^-", color="#1baf7a", lw=2, ms=6, label="Testim")
        ax.set_yscale("log")
        ax.set_xlabel("Numri i neuroneve në shtresën e fshehur (k)")
        ax.set_title(titulli, fontsize=11)

        # Shënimi i k-së optimale (minimumi i gabimit të validimit)
        ax.axvline(k_opt, color=NGJYRA_E_VERTETE, ls=":", lw=1.2)
        ax.annotate(f"k optimal = {k_opt}\n(minimumi i CV)",
                    xy=(k_opt, min(mse_cv)), xytext=(15, -45),
                    textcoords="offset points", fontsize=9,
                    arrowprops=dict(arrowstyle="->", color=NGJYRA_E_VERTETE))

        # Zona e overfitting-ut: pas k optimale gabimi i trajnimit vazhdon
        # të bjerë, por gabimi i validimit/testimit rritet
        ax.axvspan(k_opt, k_lista[-1], color="#e34948", alpha=0.06)
        ax.text(0.98, 0.97, "Zona e overfitting-ut:\ntrajnimi bie,\nvalidimi rritet",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                color="#a8302f")
    boshtet[0].set_ylabel("MSE (shkallë logaritmike)")
    boshtet[0].legend(loc="lower left")
    fig.suptitle("Gabimi sipas numrit të neuroneve: trajnim vs. validim vs. testim",
                 fontweight="bold")
    _ruaj(fig, rruga)


# ---------------------------------------------------------------------------
# 3. Heatmap k x sigma dhe efekti i sigma
# ---------------------------------------------------------------------------
def grafik_heatmap(k_lista, sigma_etiketat, matrica, rruga):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    vmax = min(matrica.max(), 10.0)   # kufizojmë ngjyrën që vlerat ekstreme të mos dominojnë
    im = ax.imshow(np.clip(matrica, None, vmax), cmap=BLU_SEKUENCIALE,
                   norm=LogNorm(vmin=matrica.min(), vmax=vmax), aspect="auto")
    ax.set_xticks(range(len(sigma_etiketat)), sigma_etiketat)
    ax.set_yticks(range(len(k_lista)), k_lista)
    ax.set_xlabel("σ (gjerësia e funksionit Gaussian)")
    ax.set_ylabel("Numri i neuroneve (k)")
    ax.set_title("MSE e validimit (5-fold CV) për çdo kombinim k × σ")
    ax.grid(False)
    # Vlerat brenda qelizave; qeliza më e mirë me kornizë
    i_min, j_min = np.unravel_index(np.argmin(matrica), matrica.shape)
    prag = np.sqrt(matrica.min() * vmax)
    for i in range(matrica.shape[0]):
        for j in range(matrica.shape[1]):
            v = matrica[i, j]
            teksti = f"{v:.3f}" if v < 10 else ">10"
            ax.text(j, i, teksti, ha="center", va="center", fontsize=8,
                    color="white" if v > prag else "#0b0b0b",
                    fontweight="bold" if (i, j) == (i_min, j_min) else "normal")
    ax.add_patch(plt.Rectangle((j_min - 0.5, i_min - 0.5), 1, 1, fill=False,
                               ec="#eb6834", lw=2.5))
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("MSE (shkallë log; më e çelët = më mirë)")
    _ruaj(fig, rruga)


def grafik_sigma(sigma_vlerat, mse_sipas_sigma, sigma_auto, mse_auto, k, rruga):
    """Vlerat fikse të σ si vijë; σ automatike si pikë e veçantë në vlerën e saj."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sigma_vlerat, mse_sipas_sigma, "o-", color="#2a78d6", lw=2, ms=8,
            label="σ e zgjedhur me dorë")
    for s, v in zip(sigma_vlerat, mse_sipas_sigma):
        ax.annotate(f"{v:.3f}", (s, v), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9)
    ax.plot(sigma_auto, mse_auto, "D", color="#eb6834", ms=10,
            label=f"σ automatike = d_max/√(2k) = {sigma_auto:.3f}")
    ax.annotate(f"{mse_auto:.3f}", (sigma_auto, mse_auto), textcoords="offset points",
                xytext=(-10, -4), ha="right", va="top", fontsize=9, color="#a8441c")
    ax.set_xscale("log")
    ax.set_xticks(sigma_vlerat, [str(s) for s in sigma_vlerat])
    ax.minorticks_off()
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    ax.set_xlabel("σ (shkallë logaritmike)")
    ax.set_ylabel("MSE e validimit (shkallë log)")
    ax.set_title(f"Ndikimi i σ në gabim (k = {k} neurone)")
    _ruaj(fig, rruga)


# ---------------------------------------------------------------------------
# 4. Efekti i secilës hyrje veç e veç
# ---------------------------------------------------------------------------
def grafik_efekti_hyrjeve(funksionet, X_ref, emrat, kufijte, f_vertete, rruga):
    """
    Ndryshon një hyrje në gjithë intervalin e saj, ndërsa dy të tjerat
    mbahen konstante në vlerën mesatare. funksionet: {emri: f(X_bruto) -> nota}.
    """
    titujt = {"oret_studimit": "Orët e studimit (në javë)",
              "oret_gjumit": "Orët e gjumit (në natë)",
              "prezenca": "Prezenca në ligjërata (%)"}
    mesataret = X_ref.mean(axis=0)
    fig, boshtet = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    for i, (ax, emri) in enumerate(zip(boshtet, emrat)):
        vlerat = np.linspace(*kufijte[emri], 200)
        X = np.tile(mesataret, (200, 1))
        X[:, i] = vlerat
        ax.plot(vlerat, f_vertete(X[:, 0], X[:, 1], X[:, 2]), "--",
                color=NGJYRA_E_VERTETE, lw=2, label="Lidhja e vërtetë (pa zhurmë)")
        for model, f in funksionet.items():
            ax.plot(vlerat, f(X), color=NGJYRAT[model], lw=2, label=model)
        ax.set_xlabel(titujt[emri])
        ax.set_title(titujt[emri].split(" (")[0])
    boshtet[0].set_ylabel("Nota e parashikuar")
    boshtet[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Efekti i secilës hyrje (të tjerat mbahen në vlerën mesatare)",
                 fontweight="bold")
    _ruaj(fig, rruga)


# ---------------------------------------------------------------------------
# 5. Funksionet Gaussiane në 1D: si mblidhen "kambanat"
# ---------------------------------------------------------------------------
def grafik_gaussiane_1d(rruga, seed=42):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 1, 60))
    y = 0.5 + 0.35 * np.sin(2 * np.pi * x) + 0.2 * x + rng.normal(0, 0.04, 60)

    # σ të vogël (afërsisht sa largësia midis qendrave) që çdo kambanë të dallohet
    k = 8
    rrjeta = RrjetaRBF(k=k, sigma=0.1, seed=seed).fit(x[:, None], y)
    xx = np.linspace(0, 1, 400)
    d2 = (xx[:, None] - rrjeta.qendrat[:, 0][None, :]) ** 2
    phi = np.exp(-d2 / (2 * rrjeta.sigma ** 2))
    w0, w = rrjeta.peshat[0], rrjeta.peshat[1:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8),
                                   gridspec_kw={"width_ratios": [1.6, 1]})
    # Majtas: kambanat e peshuara dhe shuma e tyre
    for j in range(k):
        ax1.fill_between(xx, 0, w[j] * phi[:, j], color="#86b6ef", alpha=0.25, lw=0)
        ax1.plot(xx, w[j] * phi[:, j], color="#3987e5", lw=1.3,
                 label="wⱼ · φⱼ(x)  (neuronet e peshuara)" if j == 0 else None)
        ax1.plot(rrjeta.qendrat[j, 0], 0, "v", color="#1c5cab", ms=8,
                 label="Qendrat cⱼ (nga K-Means)" if j == 0 else None)
    ax1.axhline(w0, color=NGJYRA_E_VERTETE, ls=":", lw=1, label=f"Bias w₀ = {w0:.2f}")
    ax1.plot(xx, rrjeta.predict(xx[:, None]), color="#eb6834", lw=2.5,
             label="Dalja: y(x) = w₀ + Σ wⱼ φⱼ(x)")
    ax1.scatter(x, y, s=20, color="#0b0b0b", zorder=3, label="Të dhënat")
    ax1.axhline(0, color="#c3c2b7", lw=0.8)
    ax1.set_title(f"Si mblidhen {k} \"kambanat\" Gaussiane për të formuar kurbën "
                  f"(σ = {rrjeta.sigma:.3f})", fontsize=10)
    ax1.set_xlabel("x (hyrja)")
    ax1.set_ylabel("y")
    ax1.legend(fontsize=8, loc="upper right")

    # Djathtas: një funksion Gaussian me σ të ndryshme
    ngjyrat_sigma = ["#0d366b", "#1c5cab", "#3987e5", "#86b6ef"]
    for s, ngj in zip([0.05, 0.1, 0.2, 0.5], ngjyrat_sigma):
        ax2.plot(xx, np.exp(-(xx - 0.5) ** 2 / (2 * s ** 2)), color=ngj, lw=2,
                 label=f"σ = {s}")
    ax2.set_title("φ(x) = exp(−(x − c)² / 2σ²),  c = 0.5", fontsize=10)
    ax2.set_xlabel("x")
    ax2.set_ylabel("φ(x)")
    ax2.legend(fontsize=9)
    fig.suptitle("Teoria e RBF: funksionet Gaussiane në një dimension", fontweight="bold")
    _ruaj(fig, rruga)


# ---------------------------------------------------------------------------
# 6. Shpërndarja e gabimeve (residualet)
# ---------------------------------------------------------------------------
def grafik_residualet(y_test, parashikimet, rruga):
    fig, boshtet = plt.subplots(1, len(parashikimet), figsize=(13, 4), sharey=True)
    kufiri = max(np.abs(y_test - p).max() for p in parashikimet.values())
    kosha = np.linspace(-kufiri, kufiri, 25)
    for ax, (emri, y_pred) in zip(boshtet, parashikimet.items()):
        mbetjet = y_test - y_pred
        ax.hist(mbetjet, bins=kosha, color=NGJYRAT[emri], edgecolor="white", lw=1)
        ax.axvline(0, color=NGJYRA_E_VERTETE, ls="--", lw=1)
        ax.set_title(f"{emri}: mesatarja = {mbetjet.mean():+.3f}, "
                     f"dev. std. = {mbetjet.std():.3f}", fontsize=10)
        ax.set_xlabel("Gabimi (nota reale − e parashikuar)")
    boshtet[0].set_ylabel("Numri i studentëve")
    fig.suptitle("Shpërndarja e gabimeve (residualeve) në bashkësinë e testimit",
                 fontweight="bold")
    _ruaj(fig, rruga)
