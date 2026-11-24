# Skica e prezantimit (12 slide, rreth 12–15 minuta)

**Tema:** Parashikimi i notës së studentit me rrjetë RBF

---

## Slide 1: Titulli
- Parashikimi i notës së studentit me rrjetë RBF (Radial Basis Function)
- Lënda: Bazat e Inteligjencës Artificiale
- Emri i studentit / grupit, data

---

## Slide 2: Problemi dhe qëllimi
- **Pyetja:** a mund ta parashikojmë notën (5–10) nga orët e studimit, gjumi dhe prezenca?
- Lidhjet **nuk janë lineare**: studimi ngopet, gjumi ka optimum, prezenca ndihmon gradualisht
- Qëllimi: të ndërtojmë **nga e para** një rrjetë RBF dhe ta krahasojmë me regresionin linear dhe MLP
- *Pyetje për auditorin: sa orë gjumë mendoni se është optimale?*

---

## Slide 3: Arkitektura e rrjetës RBF
- Diagrami me 3 shtresa: hyrja (3) → shtresa e fshehur (k Gaussiane) → dalja (1)
- Lidhjet hyrje → e fshehur **pa pesha**: çdo neuron mat distancën nga qendra e tij
- Dalja: $\hat{y} = w_0 + \sum w_j \varphi_j(x)$
- *Mesazhi: "çdo neuron = një student tipik; nota = sa i ngjashëm je me secilin"*

---

## Slide 4: Funksioni Gaussian dhe roli i σ
- Formula: $\varphi_j(x) = \exp\left(-\lVert x - c_j\rVert^2 / 2\sigma^2\right)$
- **Figura 6** (`6_funksionet_gaussiane_1d.png`): 8 kambana të peshuara mblidhen dhe formojnë kurbën
- σ e vogël → kambana të ngushta; σ e madhe → kambana të sheshta
- *Kjo është slide-i kyç i teorisë: kalojini kambanat një nga një*

---

## Slide 5: Trajnimi në dy faza
1. **K-Means** → qendrat $c_j$ (pa mbikëqyrje)
2. **σ** → formula $d_{max}/\sqrt{2k}$ ose cross-validation
3. **Pseudo-inverse** → peshat në një hap: $W = \Phi^{+} y$
- Krahasim me MLP: pa backpropagation, pa normë mësimi, pa minimume lokale

---

## Slide 6: Dataset-i
- 500 studentë sintetikë; 400 për trajnim, 100 për testim
- Formula e fshehur + zhurmë $\mathcal{N}(0, 0.3^2)$ → e njohim "të vërtetën"
- Nota mesatare 8.31 (dev. std. 1.01); korrelacioni me notën: studimi 0.67, prezenca 0.38, gjumi 0.22
- Normalizimi min-max në [0, 1], **i domosdoshëm** sepse RBF përdor distancën

---

## Slide 7: Zgjedhja e k dhe σ me 5-fold cross-validation
- **Figura 3** (`3_heatmap_k_sigma.png`): 84 kombinime, 420 trajnime
- Më i miri: **k = 20, σ = 2.0** → MSE e validimit 0.097
- σ = 0.05 dështon gjithmonë (MSE 0.64–3.46)
- Testi **nuk** përdoret për zgjedhjen, vetëm për vlerësimin përfundimtar

---

## Slide 8: Ndikimi i σ: formula automatike nuk mjafton
- **Figura 4** (`4_ndikimi_i_sigma.png`)
- Për k = 20: formula jep σ = 0.188 → MSE 0.301; σ = 2.0 → MSE 0.097 (**3 herë më mirë**)
- Gabimi rrafshohet pas σ ≈ 0.5
- *Mesazhi: formulat empirike janë pikënisje, jo zëvendësim i validimit*

---

## Slide 9: Overfitting
- **Figura 2** (`2_gabimi_sipas_neuroneve.png`)
- Trajnimi: 0.505 (k=2) → 0.061 (k=200), bie vazhdimisht
- Validimi: minimumi 0.097 te k = 20, pastaj rritet në 0.260 te k = 200
- k i vogël = underfitting, k i madh = mëson përmendësh zhurmën

---

## Slide 10: Krahasimi i modeleve
| Modeli | MSE test | MAE test | R² test |
|---|---|---|---|
| **RBF (k=20, σ=2.0)** | **0.113** | **0.268** | **0.904** |
| MLP (20 neurone) | 0.137 | 0.293 | 0.884 |
| Linear | 0.357 | 0.516 | 0.697 |
| *Kufiri teorik* | *0.111* | *0.259* | *0.906* |
- **Figura 1** (`1_nota_reale_vs_parashikuar.png`) pranë tabelës
- RBF gabon mesatarisht **0.27 pikë note** dhe është praktikisht te kufiri teorik

---

## Slide 11: Pse fiton RBF? Efekti i hyrjeve
- **Figura 5** (`5_efekti_i_hyrjeve.png`)
- RBF rindërton ngopjen e studimit dhe optimumin e gjumit (~7.5 orë)
- Linear-i thotë gabimisht "sa më shumë gjumë, aq më mirë"
- **Figura 7** (`7_histogrami_residualeve.png`): gabimet e RBF janë simetrike dhe të ngushta (dev. std. 0.34 kundrejt 0.60 te linear-i)

---

## Slide 12: Përfundime, kufizime dhe demo
- ✔ RBF kap lidhjet jo-lineare: R² 0.904 kundrejt 0.697 te linear-i
- ✔ σ është parametri më i ndjeshëm; k zgjidhet me CV për të shmangur overfitting-un
- ⚠ Kufizime: të dhëna sintetike, hyrje të pavarura, pesha të mëdha me σ = 2.0 (zgjidhja: rregullimi ridge)
- **Demo live:** `python app.py`, lëvizni slider-at (p.sh. 10 orë studim, 7 orë gjumë, 80% → 9.38)
- Faleminderit! Pyetje?

---

### Këshilla për prezantimin
- Hapeni `app.py` para se të filloni, që demo të jetë gati.
- Pyetje të mundshme nga profesori:
  - *"Pse pseudo-inverse dhe jo inversja e zakonshme?"* → $\Phi$ nuk është katrore (400 × 21); pseudo-inversja jep zgjidhjen e katrorëve më të vegjël dhe funksionon edhe kur $\Phi^\top\Phi$ është singulare.
  - *"Pse normalizim?"* → distanca Euklidiane përndryshe dominohet nga prezenca (diapazon 70) kundrejt gjumit (diapazon 6).
  - *"Pse k = 20 dhe jo k = 10, që ka testin më të mirë?"* → zgjodhëm me CV; zgjedhja sipas testit do të ishte "mashtrim".
  - *"Pse peshat janë kaq të mëdha?"* → me σ të madhe kolonat e $\Phi$ janë pothuajse identike (keq-kushtëzim); rregullimi ridge e zgjidh këtë.
