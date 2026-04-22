#!/usr/bin/env python3

"""
Q38 Kvantna optika — koherentna i squeezed stanja, multi-mode interferencija
(dva moda × 3 qubit-a, displaced-squeezed + coherent + beam splitter) — čisto
kvantno.

Paradigma:
  Kvantna optika gradi kvantna stanja iz tri osnovna elementa:
    1) KOHERENTNO STANJE  |α⟩ = D(α) |0⟩,   D(α) = exp(α a† − α* a)
    2) SQUEEZED STANJE    S(r) |0⟩,         S(r) = exp((r/2) (a² − a†²))
    3) MULTI-MODE INTERFERENCIJA preko beam splitter-a
                          BS(θ) = exp(θ (a† b − a b†))
  Ova tri operatora zajedno obuhvataju Gaussian kvantnooptičke transformacije.

Mapiranje na loto:
  NQ = 6 qubit-a po poziciji se deli na DVA MODA:
        Mode A: qubits [0, 1, 2]   — Fock dim 8,  n_A ∈ [0, 7]   (low 3 bits)
        Mode B: qubits [3, 4, 5]   — Fock dim 8,  n_B ∈ [0, 7]   (high 3 bits)
        j = n_B · 8 + n_A      ∈ [0, 63]
        Num_i = i + j

  Strukturalni target (non-freq, kombinatorna order-statistika):
        target_i(prev) = prev + (N_MAX − prev) / (N_NUMBERS − i + 2)
        j_target = round(target_i) − i   ∈ [0, 32]
        n_A_target = j_target mod 8
        n_B_target = j_target div 8   ∈ [0, 4]

  Parametri kvantnooptičkog stanja (deterministički:
        α_A = √(n_A_target)        — koherentna displacement u modu A
        r   = SQUEEZE_R  (fiksno)  — squeeze strength u modu A
        α_B = √(n_B_target)        — koherentna displacement u modu B
        θ   = π / 4                — 50/50 beam splitter

  Priprema stanja (redosled Gaussian operacija):
        |ψ⟩ = BS(θ) · [ D_A(α_A) · S_A(r) ] ⊗ [ D_B(α_B) ] · |0, 0⟩
          — mode A: displaced squeezed (non-Gaussian photon number)
          — mode B: coherent
          — beam splitter: multi-mode interferencija (A ↔ B entanglement)

Merenje i predikcija:
  Joint photon-number distribucija:
        P(n_A, n_B) = |⟨n_A, n_B | ψ⟩|²
  j = n_B · 8 + n_A → mask valid (num > prev_pick, num ∈ [i, i+32], j < 33)
  → renormalize → Born sempling sa numpy.default_rng(SEED)
  → Num_i = i + j.

(fit): Gaussian-optičke transformacije su jedinstvena kvantna
građevinska kocka koja ne duplira nijedan prethodni model — displaced
squeezed + coherent + beam splitter daje ne-faktorizovan joint photon-number
pattern sa fizičkom interferencijom između modova.
tačno 6 qubit-a po poziciji (3 + 3 moda, reciklirani registar).
kvantna optika (multi-mode + squeezing) nije obrađivana.

Okruženje: Python 3.11.13, qiskit 1.4.4, macOS M1, seed = 39.
CSV = /Users/4c/Desktop/GHQ/data/loto7hh_4602_k32.csv
CSV u celini (S̄ kao info).
DeprecationWarning / FutureWarning se gase.
"""


from __future__ import annotations

import csv
import math
import random
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from scipy.sparse import SparseEfficiencyWarning

    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
except ImportError:
    pass

from scipy.linalg import expm


# =========================
# Seed 
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
try:
    from qiskit_machine_learning.utils import algorithm_globals

    algorithm_globals.random_seed = SEED
except ImportError:
    pass


# =========================
# Konfiguracija
# =========================
CSV_PATH = Path("/Users/4c/Desktop/GHQ/data/loto7hh_4602_k32.csv")
N_NUMBERS = 7
N_MAX = 39

NQ = 6                              
DIM = 1 << NQ                       # 64 (joint)
POS_RANGE = 33                      # Num_i ∈ [i, i + 32]

MODE_DIM = 8                        # 3 qubit-a po modu, truncated Fock dim
SQUEEZE_R = 0.4                     # fiksna squeeze strength (mode A)
BS_THETA = math.pi / 4              # 50/50 beam splitter


# =========================
# CSV 
# =========================
def load_rows(path: Path) -> np.ndarray:
    rows: List[List[int]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, None)
        if not header or "Num1" not in header[0]:
            f.seek(0)
            r = csv.reader(f)
            next(r, None)
        for row in r:
            if not row or row[0].strip() == "Num1":
                continue
            rows.append([int(row[i]) for i in range(N_NUMBERS)])
    return np.array(rows, dtype=int)


def sort_rows_asc(H: np.ndarray) -> np.ndarray:
    return np.sort(H, axis=1)


# =========================
# Structural target (bez frekvencije)
# =========================
def target_num_structural(position_1based: int, prev_pick: int) -> float:
    denom = float(N_NUMBERS - position_1based + 2)
    return float(prev_pick) + float(N_MAX - prev_pick) / denom


def compute_j_target(position_1based: int, prev_pick: int) -> Tuple[int, float]:
    target = target_num_structural(position_1based, prev_pick)
    j = int(round(target)) - position_1based
    j = max(0, min(POS_RANGE - 1, j))
    return j, target


# =========================
# Truncated ladder operatori po modu (MODE_DIM × MODE_DIM)
# =========================
def _single_mode_ladder() -> Tuple[np.ndarray, np.ndarray]:
    a = np.zeros((MODE_DIM, MODE_DIM), dtype=np.complex128)
    for n in range(1, MODE_DIM):
        a[n - 1, n] = math.sqrt(float(n))
    return a, a.conj().T


A_1M, A_DAG_1M = _single_mode_ladder()

# Embed u joint (64 × 64) prostor: red = kron(mode_A, mode_B)
I_1M = np.eye(MODE_DIM, dtype=np.complex128)

A_A = np.kron(A_1M, I_1M)          # a_A  (64×64)
A_A_DAG = np.kron(A_DAG_1M, I_1M)
A_B = np.kron(I_1M, A_1M)          # a_B
A_B_DAG = np.kron(I_1M, A_DAG_1M)


# =========================
# Gaussian operatori (na joint 64-dim prostoru)
# =========================
def displacement_mode_A(alpha: float) -> np.ndarray:
    gen = alpha * A_A_DAG - alpha * A_A
    return expm(gen)


def displacement_mode_B(alpha: float) -> np.ndarray:
    gen = alpha * A_B_DAG - alpha * A_B
    return expm(gen)


def squeeze_mode_A(r: float) -> np.ndarray:
    # S(r) = exp( (r/2) (a² − a†²) )
    gen = (r / 2.0) * (A_A @ A_A - A_A_DAG @ A_A_DAG)
    return expm(gen)


def beam_splitter(theta: float) -> np.ndarray:
    # BS(θ) = exp( θ (a_A† a_B − a_A a_B†) )
    gen = theta * (A_A_DAG @ A_B - A_A @ A_B_DAG)
    return expm(gen)


# =========================
# Priprema |ψ⟩ = BS · (D_A · S_A ⊗ D_B) · |00⟩
# =========================
def prepare_optical_state(alpha_a: float, alpha_b: float, r: float) -> np.ndarray:
    vac = np.zeros(DIM, dtype=np.complex128)
    vac[0] = 1.0                           # |n_A=0, n_B=0⟩

    S_A = squeeze_mode_A(r)
    D_A = displacement_mode_A(alpha_a)
    D_B = displacement_mode_B(alpha_b)
    BS = beam_splitter(BS_THETA)

    psi = S_A @ vac
    psi = D_A @ psi
    psi = D_B @ psi
    psi = BS @ psi

    n = np.linalg.norm(psi)
    if n > 0:
        psi /= n
    return psi


# =========================
# Indeksiranje: j = n_B · 8 + n_A  (red-major kron(mode_A, mode_B))
# =========================
def joint_index_to_j(idx: int) -> int:
    n_A = idx // MODE_DIM
    n_B = idx % MODE_DIM
    return n_B * MODE_DIM + n_A


# =========================
# Predikcija jedne pozicije
# =========================
def optics_pick_one_position(
    position_1based: int,
    prev_pick: int,
    rng: np.random.Generator,
) -> Tuple[int, int, float, float, float, float]:
    j_target, target = compute_j_target(position_1based, prev_pick)
    n_A_tgt = j_target % MODE_DIM
    n_B_tgt = j_target // MODE_DIM

    alpha_a = math.sqrt(float(n_A_tgt))
    alpha_b = math.sqrt(float(n_B_tgt))

    psi = prepare_optical_state(alpha_a, alpha_b, SQUEEZE_R)

    probs_joint = (np.abs(psi) ** 2).real
    probs_joint = np.clip(probs_joint, 0.0, None)

    # Remap joint indeks → j ∈ [0, 63]
    probs_j = np.zeros(DIM, dtype=np.float64)
    for idx in range(DIM):
        j = joint_index_to_j(idx)
        probs_j[j] += probs_joint[idx]

    mask = np.zeros(DIM, dtype=np.float64)
    for j in range(DIM):
        num = position_1based + j
        if 1 <= num <= N_MAX and num > prev_pick and j < POS_RANGE:
            mask[j] = 1.0

    probs_valid = probs_j * mask
    s = float(probs_valid.sum())
    if s < 1e-15:
        for j in range(POS_RANGE):
            num = position_1based + j
            if 1 <= num <= N_MAX and num > prev_pick:
                return num, j_target, target, alpha_a, alpha_b, 0.0
        return (
            max(prev_pick + 1, position_1based),
            j_target,
            target,
            alpha_a,
            alpha_b,
            0.0,
        )

    probs_valid /= s
    j_sampled = int(rng.choice(DIM, p=probs_valid))
    num = position_1based + j_sampled
    return num, j_target, target, alpha_a, alpha_b, float(probs_valid[j_sampled])


# =========================
# Autoregresivni run (reciklirani 6-qubit / 2-mod prostor)
# =========================
def run_optics_autoregressive() -> List[int]:
    rng = np.random.default_rng(SEED)
    picks: List[int] = []
    prev_pick = 0

    for i in range(1, N_NUMBERS + 1):
        num, j_t, target, alpha_a, alpha_b, p_samp = optics_pick_one_position(
            i, prev_pick, rng
        )
        picks.append(int(num))
        print(
            f"  [pos {i}]  target={target:.3f}  j_target={j_t:2d}  "
            f"α_A={alpha_a:.3f}  α_B={alpha_b:.3f}  r={SQUEEZE_R:.2f}  "
            f"P(sample)={p_samp:.4f}  num={num:2d}"
        )
        prev_pick = int(num)

    return picks


# =========================
# Main
# =========================
def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Nema CSV: {CSV_PATH}")

    H = load_rows(CSV_PATH)
    H_sorted = sort_rows_asc(H)
    S_bar = float(H_sorted.sum(axis=1).mean())

    print("=" * 84)
    print("Q38 Kvantna optika — dual-mode coherent + squeezed + beam splitter")
    print("=" * 84)
    print(f"CSV:            {CSV_PATH}")
    print(f"Broj redova:    {H.shape[0]}")
    print(f"Qubit budget:   {NQ} po poziciji  (2 moda × 3 qubit-a, joint dim={DIM})")
    print(f"Mode A:         Fock dim={MODE_DIM}, displaced squeezed (α_A, r)")
    print(f"Mode B:         Fock dim={MODE_DIM}, coherent (α_B)")
    print(f"Squeeze r:      {SQUEEZE_R}  (fiksno)")
    print(f"Beam splitter:  θ = π/4 (50/50)")
    print(f"j-mapiranje:    j = n_B · 8 + n_A     (n_A = low 3 bits, n_B = high 3)")
    print(f"Srednja suma S̄: {S_bar:.3f}  (CSV info, nije driver)")
    print(f"Seed:           {SEED}")
    print()
    print("Pokretanje kvantnooptičkog pripreme + Born sempling po pozicijama:")

    picks = run_optics_autoregressive()

    n_odd = sum(1 for v in picks if v % 2 == 1)
    gaps = [picks[i + 1] - picks[i] for i in range(N_NUMBERS - 1)]

    print()
    print("=" * 84)
    print("REZULTAT Q38 (NEXT kombinacija)")
    print("=" * 84)
    print(f"Suma:  {sum(picks)}   (S̄={S_bar:.2f})")
    print(f"#odd:  {n_odd}")
    print(f"Gaps:  {gaps}")
    print(f"Predikcija NEXT: {picks}")


if __name__ == "__main__":
    main()



"""
====================================================================================
Q38 Kvantna optika — dual-mode coherent + squeezed + beam splitter
====================================================================================
CSV:            /data/loto7hh_4602_k32.csv
Broj redova:    4602
Qubit budget:   6 po poziciji  (2 moda × 3 qubit-a, joint dim=64)
Mode A:         Fock dim=8, displaced squeezed (α_A, r)
Mode B:         Fock dim=8, coherent (α_B)
Squeeze r:      0.4  (fiksno)
Beam splitter:  θ = π/4 (50/50)
j-mapiranje:    j = n_B · 8 + n_A     (n_A = low 3 bits, n_B = high 3)
Srednja suma S̄: 140.509  (CSV info, nije driver)
Seed:           39

Pokretanje kvantnooptičkog pripreme + Born sempling po pozicijama:
  [pos 1]  target=4.875  j_target= 4  α_A=2.000  α_B=0.000  r=0.40  P(sample)=0.1097  num=19
  [pos 2]  target=21.857  j_target=20  α_A=2.000  α_B=1.414  r=0.40  P(sample)=0.4531  num=24
  [pos 3]  target=26.500  j_target=23  α_A=2.646  α_B=1.414  r=0.40  P(sample)=0.2206  num=25
  [pos 4]  target=27.800  j_target=24  α_A=0.000  α_B=1.732  r=0.40  P(sample)=0.2281  num=29
  [pos 5]  target=31.500  j_target=27  α_A=1.732  α_B=1.732  r=0.40  P(sample)=0.6419  num=35
  [pos 6]  target=36.333  j_target=30  α_A=2.449  α_B=1.732  r=0.40  P(sample)=0.3103  num=37
  [pos 7]  target=38.000  j_target=31  α_A=2.646  α_B=1.732  r=0.40  P(sample)=0.9985  num=38

====================================================================================
REZULTAT Q38 (NEXT kombinacija)
====================================================================================
Suma:  207   (S̄=140.51)
#odd:  5
Gaps:  [5, 1, 4, 6, 2, 1]
Predikcija NEXT: [19, 24, 25, 29, 35, 37, 38]
"""



"""
REZULTAT — Q38 Kvantna optika (dual-mode coherent + squeezed + beam splitter)
-----------------------------------------------------------------------
(Popunjava se iz printa main()-a nakon pokretanja.)

Koncept:
  • Čisto kvantno: Gaussian kvantnooptičke transformacije
    (displacement, squeezing, beam splitter) nad truncated 2-mode Fock prostorom;
    Born sempling iz joint photon-number distribucije.
  • QO paradigm: KORELISANO stanje dva optička moda kroz multi-mode interferenciju;
    squeezing uvodi non-Gaussian photon-number statistiku, beam splitter uvodi
    entanglement između modova.
  • NQ = 6 qubit-a po poziciji = 2 moda × 3 qubit-a, reciklirani.
  • deterministički Gaussian parametri + seeded RNG za Born sempling.

Tehnike:
  • Truncated ladder operatori po modu, embedded u joint 64-dim prostor preko kron.
  • Displacement D_A(α_A), D_B(α_B), squeeze S_A(r), beam splitter BS(π/4)
    preko scipy.linalg.expm.
  • Joint photon-number merenje → remap joint indeks u j = n_B · 8 + n_A.
  • Born sempling iz valid-masked distribucije.
"""
