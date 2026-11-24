# Parashikimi i notës së studentit me rrjetë RBF

Projekt për lëndën **Bazat e Inteligjencës Artificiale**.

Rrjeta neurale **RBF (Radial Basis Function)** parashikon notën e studentit (5–10)
nga tre hyrje:

| Hyrja | Intervali |
|---|---|
| Orët e studimit në javë | 0 – 20 |
| Orët e gjumit në natë | 4 – 10 |
| Prezenca në ligjërata | 30 – 100 % |

Rrjeta RBF, K-Means, regresioni linear, metrikat dhe cross-validation janë
**implementuar nga e para me NumPy**. Biblioteka `scikit-learn` përdoret vetëm për
modelin referencë MLP.

## Rezultatet kryesore (bashkësia e testimit, 100 studentë)

| Modeli | MSE | MAE | R² |
|---|---|---|---|
| **RBF (k = 20, σ = 2.0)** | **0.1128** | **0.2684** | **0.9042** |
| MLP (sklearn, 20 neurone) | 0.1365 | 0.2925 | 0.8840 |
| RBF (k = 60, σ automatike = 0.125) | 0.3143 | 0.4266 | 0.7330 |
| Regresioni linear | 0.3569 | 0.5162 | 0.6968 |
| *Kufiri teorik (formula e vërtetë pa zhurmë)* | *0.1107* | *0.2590* | *0.9059* |

## Instalimi

Nevojitet Python 3.10 ose më i ri.

```bash
cd rbf_projekti
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

## Ekzekutimi

```bash
python main.py     # gjithë eksperimenti (~25 sekonda): tabela, grafikë, modeli
python app.py      # ndërfaqja grafike për parashikim
```

`main.py` gjeneron dataset-in nëse mungon (`data/studentet.csv`), bën të gjitha
eksperimentet dhe ruan rezultatet te `rezultatet/`. Seed-i është fiks (`SEED = 42`),
prandaj çdo ekzekutim jep saktësisht të njëjtat numra.

`app.py` ngarkon modelin e ruajtur (`rezultatet/modeli_rbf.npz`). Nëse modeli mungon,
e trajnon vetë. Me tre slider-a vendosen orët e studimit, të gjumit dhe prezenca, dhe
nota e parashikuar shfaqet menjëherë.

## Struktura e dosjeve

```
rbf_projekti/
├── data/
│   └── studentet.csv              # dataset-i sintetik (500 studentë)
├── src/
│   ├── dataset.py                 # gjenerimi, ngarkimi, ndarja, normalizimi, k-fold
│   ├── kmeans.py                  # K-Means nga e para
│   ├── rbf.py                     # klasa RrjetaRBF (trajnim, parashikim, ruajtje)
│   ├── modelet_krahasuese.py      # regresioni linear (nga e para) + MLP (sklearn)
│   ├── metrikat.py                # MSE, MAE, R²
│   └── vizualizimi.py             # të gjithë grafikët
├── main.py                        # eksperimenti i plotë
├── app.py                         # ndërfaqja Tkinter
├── rezultatet/
│   ├── 1_nota_reale_vs_parashikuar.png
│   ├── 2_gabimi_sipas_neuroneve.png
│   ├── 3_heatmap_k_sigma.png
│   ├── 4_ndikimi_i_sigma.png
│   ├── 5_efekti_i_hyrjeve.png
│   ├── 6_funksionet_gaussiane_1d.png
│   ├── 7_histogrami_residualeve.png
│   ├── tabela_rezultateve.csv     # krahasimi përfundimtar i modeleve
│   ├── tabela_k_sigma.csv         # MSE e CV për çdo kombinim k × σ
│   ├── tabela_neuronet.csv        # gabimi train / CV / test sipas k
│   └── modeli_rbf.npz             # modeli i ruajtur për app.py
├── raporti.md                     # raporti akademik
├── prezantimi.md                  # skica e prezantimit
├── requirements.txt
└── README.md
```

## Si funksionon rrjeta RBF (shkurt)

1. **K-Means** gjen `k` qendra në hapësirën e hyrjeve (studentë "tipikë").
2. Çdo neuron i fshehur jep $\varphi_j(x) = \exp\left(-\frac{\lVert x - c_j\rVert^2}{2\sigma^2}\right)$:
   vlerë afër 1 kur studenti është afër qendrës, afër 0 kur është larg.
3. Dalja është kombinim linear $y = w_0 + \sum_j w_j\varphi_j(x)$, me peshat e llogaritura
   në një hap të vetëm me **pseudo-inverse**: $W = \Phi^{+} y$.

Shpjegimi i plotë është te `raporti.md`.
