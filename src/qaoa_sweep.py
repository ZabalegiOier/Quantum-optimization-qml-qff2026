import json
import time
import itertools
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # sin ventana: guarda la gráfica en un archivo
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz, StatePreparation
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import SparsePauliOp
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import LinearEqualityToPenalty

warnings.filterwarnings("ignore", category=Warning, module="scipy")

# ==========================================
# CONFIGURACION
# ==========================================
K = 4
q = 0.5
RESTARTS = 5
MAXITER = 300
SHOTS = 4096
lambdas_to_test = [1.0, 5.0, 10.0, 20.0, 50.0]
p_depths = [1, 2, 3, 4]
CHECK_FAST = True    # compara energia rapida con Qiskit una vez
RUN_XY = False       # mixer XY sin penalizacion (mas lento, usa Qiskit)

# ==========================================
# 1. DATOS
# ==========================================
ruta = Path(__file__).resolve().parent.parent / "data" / "beginner" / "parametros_8_activos.json"
with open(ruta, "r") as f:
    datos = json.load(f)

n = datos["n_activos"]
tickers = datos["tickers"]
mu = np.array(datos["mu"])
sigma = np.array(datos["sigma"])

# ==========================================
# 2. PROBLEMA (QP y QUBO)
# ==========================================
def build_portfolio_qp(penalty_lam):
    qp = QuadraticProgram()
    for t in tickers:
        qp.binary_var(name=t)
    qp.maximize(linear=mu, quadratic=-q * sigma)
    qp.linear_constraint(linear={t: 1 for t in tickers}, sense="==", rhs=K, name="budget")
    qubo = LinearEqualityToPenalty(penalty=penalty_lam).convert(qp)
    return qp, qubo

# ==========================================
# 3. BENCHMARK EXACTO (fuerza bruta sobre carteras validas)
# ==========================================
qp_ref, _ = build_portfolio_qp(penalty_lam=1.0)

optimal_value_exact, best_combo = -np.inf, None
for combo in itertools.combinations(range(n), K):
    x = [1 if i in combo else 0 for i in range(n)]
    val = qp_ref.objective.evaluate(x)
    if val > optimal_value_exact:
        optimal_value_exact, best_combo = val, x

# ==========================================
# 4. ENERGIA RAPIDA (numpy) Y QAOA
# ==========================================
def make_fast_energy(H, n_qubits, p):
    diag = np.real(H.to_matrix(sparse=True).diagonal())
    dim = 2 ** n_qubits

    def energy(params):
        betas, gammas = params[:p], params[p:]
        psi = np.full(dim, 1 / np.sqrt(dim), dtype=complex)
        for b, g in zip(betas, gammas):
            psi = psi * np.exp(-1j * g * diag)
            c, s = np.cos(b), -1j * np.sin(b)
            rx = np.array([[c, s], [s, c]])
            psi = psi.reshape([2] * n_qubits)
            for ax in range(n_qubits):
                psi = np.moveaxis(np.tensordot(rx, psi, axes=([1], [ax])), 0, ax)
            psi = psi.reshape(dim)
        return float(np.sum(np.abs(psi) ** 2 * diag))

    return energy


def run_qaoa(H, qp, K, p, restarts=RESTARTS, shots=SHOTS, maxiter=MAXITER, seed=0,
             init_state=None, mixer=None, x0=None):
    H = H * (1.0 / np.max(np.abs(H.coeffs)))

    kw = {}
    if init_state is not None:
        kw["initial_state"] = init_state
    if mixer is not None:
        kw["mixer_operator"] = mixer

    ansatz_no_meas = QAOAAnsatz(cost_operator=H, reps=p, **kw)
    ansatz = ansatz_no_meas.copy()
    ansatz.measure_all()

    estimator, sampler = StatevectorEstimator(), StatevectorSampler()

    if mixer is None and init_state is None:
        cost_function = make_fast_energy(H, H.num_qubits, p)
    else:
        def cost_function(params):
            pub = (ansatz_no_meas, H, params)
            return float(estimator.run([pub]).result()[0].data.evs)

    rng = np.random.default_rng(seed)
    best = None
    for r in range(restarts):
        if x0 is not None and r == 0:
            start = x0
        else:
            start = rng.uniform(0, np.pi, ansatz_no_meas.num_parameters)
        res = minimize(cost_function, start, method="COBYLA", options={"maxiter": maxiter})
        if best is None or res.fun < best.fun:
            best = res

    counts = sampler.run([(ansatz, best.x)], shots=shots).result()[0].data.meas.get_counts()

    feasible_shots, best_x, best_val = 0, None, -np.inf
    for bitstring, c in counts.items():
        x = [int(b) for b in bitstring.replace(" ", "")[::-1]]
        if sum(x) == K:
            feasible_shots += c
            val = qp.objective.evaluate(x)
            if val > best_val:
                best_val, best_x = val, x

    return dict(x=best_x, value=best_val, p_feasible=feasible_shots / shots,
                params=best.x, energy=best.fun)


def check_fast_energy(H, p=2):
    Hn = H * (1.0 / np.max(np.abs(H.coeffs)))
    params = np.random.default_rng(0).uniform(0, np.pi, 2 * p)
    ans = QAOAAnsatz(cost_operator=Hn, reps=p)
    e_qiskit = float(StatevectorEstimator().run([(ans, Hn, params)]).result()[0].data.evs)
    e_fast = make_fast_energy(Hn, Hn.num_qubits, p)(params)
    ok = abs(e_qiskit - e_fast) < 1e-6
    print(f"Comprobacion energia: qiskit={e_qiskit:.8f} rapida={e_fast:.8f} -> {'OK' if ok else 'NO COINCIDEN'}")
    return ok


def report(out):
    if out["x"] is None:
        print("  Sin muestras factibles")
        return 0.0
    ratio = out["value"] / optimal_value_exact
    print(f"  Activos: {[tickers[i] for i in range(n) if out['x'][i]]}")
    print(f"  Valor: {out['value']:.4f}  Ratio: {ratio:.4f}")
    return ratio


# ==========================================
# 5. PRINCIPAL
# ==========================================
def main():
    print(f"Optimizando portafolio para {n} activos: {tickers}")
    print(f"Optimo exacto: {optimal_value_exact:.4f}  "
          f"Cartera: {[tickers[i] for i in range(n) if best_combo[i]]}")

    # --- Comprobacion de la energia rapida ---
    if CHECK_FAST:
        _, qubo_c = build_portfolio_qp(10.0)
        H_c, _ = qubo_c.to_ising()
        if not check_fast_energy(H_c):
            print("Parando: intercambia betas y gammas en make_fast_energy.")
            return

    # --- Ejecucion individual ---
    lam, p_depth = 10.0, 2
    qp, qubo = build_portfolio_qp(lam)
    H, offset = qubo.to_ising()
    t = time.time()
    out = run_qaoa(H, qp, K, p_depth)
    print(f"\nQAOA p={p_depth}, lambda={lam}  ({time.time() - t:.1f}s)")
    report(out)
    print(f"  Probabilidad factible: {out['p_feasible']:.2%}")

    # --- Barrido de lambda y p ---
    results_sweep = []
    for lam_val in lambdas_to_test:
        qp, qubo = build_portfolio_qp(lam_val)
        H, _ = qubo.to_ising()
        prev = None
        for p in p_depths:
            x0 = None
            if prev is not None:
                betas, gammas = prev[:len(prev) // 2], prev[len(prev) // 2:]
                x0 = np.concatenate([betas, [betas[-1]], gammas, [gammas[-1]]])
            t = time.time()
            out = run_qaoa(H, qp, K, p, x0=x0)
            prev = out["params"]
            ratio = out["value"] / optimal_value_exact if out["x"] is not None else 0.0
            results_sweep.append({"lambda": lam_val, "p": p,
                                  "approx_ratio": ratio, "p_feasible": out["p_feasible"]})
            print(f"lambda={lam_val:5} p={p}  ratio={ratio:.4f}  "
                  f"feasible={out['p_feasible']:.1%}  ({time.time() - t:.1f}s)")

    # --- Grafica ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for l_val in lambdas_to_test:
        sub = [r for r in results_sweep if r["lambda"] == l_val]
        ps = [r["p"] for r in sub]
        axes[0].plot(ps, [r["approx_ratio"] for r in sub], marker="o", label=f"lambda={l_val}")
        axes[1].plot(ps, [r["p_feasible"] for r in sub], marker="o", label=f"lambda={l_val}")
    axes[0].set_title("Ratio de aproximacion")
    axes[1].set_title("Probabilidad de muestras factibles")
    for ax in axes:
        ax.set_xlabel("Profundidad del circuito (p)")
        ax.set_xticks(p_depths)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
    axes[0].set_ylabel("Ratio")
    axes[1].set_ylabel("Fraccion factible")
    plt.tight_layout()
    plt.savefig("barrido_qaoa.png", dpi=150)
    print("\nGrafica guardada en barrido_qaoa.png")

    # --- Mixer XY opcional ---
    if RUN_XY:
        qp_free = QuadraticProgram()
        for tk in tickers:
            qp_free.binary_var(tk)
        qp_free.maximize(linear=mu, quadratic=-q * sigma)
        H_free, _ = qp_free.to_ising()

        amps = np.zeros(2 ** n)
        for combo in itertools.combinations(range(n), K):
            amps[sum(1 << i for i in combo)] = 1
        amps /= np.linalg.norm(amps)
        init_state = QuantumCircuit(n)
        init_state.append(StatePreparation(amps), range(n))

        mixer = SparsePauliOp.from_sparse_list(
            [(P, [i, (i + 1) % n], 0.5) for i in range(n) for P in ("XX", "YY")],
            num_qubits=n)

        out_xy = run_qaoa(H_free, qp_ref, K, 3, restarts=2, maxiter=100,
                          init_state=init_state, mixer=mixer)
        print("\nMixer XY (p=3):")
        report(out_xy)
        print(f"  Probabilidad factible: {out_xy['p_feasible']:.2%}")


if __name__ == "__main__":
    main()
