"""
app.py - Ndërfaqe e thjeshtë (Tkinter) për parashikimin e notës me rrjetën RBF.

    python app.py

Ngarkon modelin e ruajtur nga main.py (rezultatet/modeli_rbf.npz).
Nëse modeli nuk ekziston, e trajnon një herë dhe e ruan.
"""

import os
import tkinter as tk
from tkinter import ttk
import numpy as np

from src.dataset import ngarko_dataset, ndaj_train_test, NormalizuesMinMax, KUFIJTE
from src.rbf import RrjetaRBF

DOSJA = os.path.dirname(os.path.abspath(__file__))
RRUGA_MODELI = os.path.join(DOSJA, "rezultatet", "modeli_rbf.npz")
RRUGA_DATA = os.path.join(DOSJA, "data", "studentet.csv")


def merr_modelin():
    """Ngarkon modelin; nëse mungon, e trajnon me parametrat e gjetur nga eksperimenti."""
    if not os.path.exists(RRUGA_MODELI):
        print("Modeli nuk u gjet - po trajnohet (k=20, σ=2.0)...")
        X, y = ngarko_dataset(RRUGA_DATA)
        X_train, _, y_train, _ = ndaj_train_test(X, y, 0.2, seed=42)
        norm = NormalizuesMinMax().fit(X_train)
        m = RrjetaRBF(k=20, sigma=2.0, seed=42).fit(norm.transform(X_train), y_train)
        os.makedirs(os.path.dirname(RRUGA_MODELI), exist_ok=True)
        m.ruaj(RRUGA_MODELI, norm.min, norm.max)
    return RrjetaRBF.ngarko(RRUGA_MODELI)


class Aplikacioni:

    def __init__(self, rrenja):
        self.modeli, self.norm_min, self.norm_max = merr_modelin()

        rrenja.title("Parashikimi i notës me rrjetë RBF")
        rrenja.geometry("460x420")
        rrenja.resizable(False, False)
        korniza = ttk.Frame(rrenja, padding=20)
        korniza.pack(fill="both", expand=True)

        ttk.Label(korniza, text="Parashikimi i notës së studentit",
                  font=("Segoe UI", 14, "bold")).pack(pady=(0, 15))

        # Tre slider-a: (çelësi, etiketa, hapi)
        self.vlerat = {}
        for celesi, etiketa, hapi in [("oret_studimit", "Orët e studimit në javë", 0.5),
                                      ("oret_gjumit", "Orët e gjumit në natë", 0.25),
                                      ("prezenca", "Prezenca në ligjërata (%)", 1)]:
            vmin, vmax = KUFIJTE[celesi]
            var = tk.DoubleVar(value=(vmin + vmax) / 2)
            self.vlerat[celesi] = var

            rreshti = ttk.Frame(korniza)
            rreshti.pack(fill="x", pady=4)
            ttk.Label(rreshti, text=etiketa).pack(anchor="w")
            lbl = ttk.Label(rreshti, width=6, anchor="e")
            lbl.pack(side="right")
            slider = tk.Scale(rreshti, from_=vmin, to=vmax, resolution=hapi,
                              orient="horizontal", variable=var, showvalue=False,
                              length=330, command=lambda _v, l=lbl, v=var: self.perditeso(l, v))
            slider.pack(side="left", fill="x", expand=True)
            lbl.config(text=f"{var.get():g}")

        # Rezultati
        self.lbl_nota = ttk.Label(korniza, font=("Segoe UI", 28, "bold"),
                                  foreground="#1c5cab")
        self.lbl_nota.pack(pady=(20, 0))
        self.lbl_info = ttk.Label(korniza, foreground="#52514e")
        self.lbl_info.pack()
        ttk.Label(korniza, foreground="#898781",
                  text=f"Modeli: RBF me k = {self.modeli.k} neurone, "
                       f"σ = {self.modeli.sigma:.3f}").pack(side="bottom")
        self.parashiko()

    def perditeso(self, etiketa, var):
        etiketa.config(text=f"{var.get():g}")
        self.parashiko()

    def parashiko(self):
        x = np.array([[self.vlerat[c].get() for c in ("oret_studimit", "oret_gjumit",
                                                       "prezenca")]])
        # I njëjti normalizim si gjatë trajnimit
        x_norm = (x - self.norm_min) / (self.norm_max - self.norm_min)
        nota = float(np.clip(self.modeli.predict(x_norm)[0], 5, 10))
        self.lbl_nota.config(text=f"Nota: {nota:.2f}")
        self.lbl_info.config(text=f"(e rrumbullakosur: {int(round(nota))})")


if __name__ == "__main__":
    rrenja = tk.Tk()
    Aplikacioni(rrenja)
    rrenja.mainloop()
