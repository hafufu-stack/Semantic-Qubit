# -*- coding: utf-8 -*-
"""
Phase CQ1: Complex S-Qubit Algebra (CPU, no model required)
Pure numerical verification of the S-Qubit algebra.

An S-Qubit is a unit vector in C^d (complex d-dimensional space).
Unlike a quantum bit which is C^2, an S-Qubit lives in C^{1536}
(the hidden dimension of our LLM).

Verify:
1. Inner product structure: <phi|psi> = phi^dagger @ psi
2. Unitarity of gates: U @ U^dagger = I
3. Measurement operators: P_k = |e_k><e_k| (projection)
4. Born rule: P(k) = |<e_k|psi>|^2
5. Tensor product for 2-qubit analog: C^d x C^d = C^{d^2}
6. Schmidt decomposition of a bipartite state
"""
import numpy as np, json, os, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

D = 64  # use 64-dim for visualization clarity (not 1536)
np.random.seed(42)


def random_unit_vec(d, dtype=complex):
    """Random unit vector in C^d."""
    v = np.random.randn(d) + 1j * np.random.randn(d)
    return v / np.linalg.norm(v)


def random_unitary(d):
    """Random unitary matrix via QR decomposition."""
    Z = np.random.randn(d, d) + 1j * np.random.randn(d, d)
    Q, R = np.linalg.qr(Z)
    # Phase correction to ensure det(Q) = 1
    D_mat = np.diag(R.diagonal() / np.abs(R.diagonal()))
    return Q @ D_mat


def born_probability(psi, e_k):
    """Born rule: P(k) = |<e_k|psi>|^2"""
    return np.abs(np.dot(e_k.conj(), psi)) ** 2


def schmidt_decomposition(psi, d_a, d_b):
    """
    Schmidt decomposition of |psi> in C^{d_a} x C^{d_b}.
    Reshape psi into d_a x d_b matrix and SVD.
    """
    M = psi.reshape(d_a, d_b)
    U, S, Vh = np.linalg.svd(M, full_matrices=False)
    return U, S, Vh


def main():
    print("[CQ1] Complex S-Qubit Algebra (pure numerical)")
    start = time.time()
    results = {}

    # === Test 1: Inner product and normalization ===
    print("  Test 1: Inner product structure...")
    psi = random_unit_vec(D)
    phi = random_unit_vec(D)
    norm_psi = np.dot(psi.conj(), psi).real
    inner = np.dot(phi.conj(), psi)
    cauchy_schwarz = abs(inner) <= 1.0 + 1e-10  # |<phi|psi>| <= 1 for unit vectors
    results['inner_product'] = {
        'norm_psi': round(float(norm_psi), 8),
        'inner_phi_psi_abs': round(float(abs(inner)), 6),
        'cauchy_schwarz_satisfied': bool(cauchy_schwarz),
    }
    print("    norm(psi)=%.8f, |<phi|psi>|=%.6f, CS satisfied: %s" % (
        norm_psi, abs(inner), cauchy_schwarz))

    # === Test 2: Unitarity of random gate ===
    print("  Test 2: Unitary gate U @ U^dagger = I...")
    U = random_unitary(D)
    UUd = U @ U.T.conj()
    UdU = U.T.conj() @ U
    unitarity_err = float(np.linalg.norm(UUd - np.eye(D)))
    results['unitarity'] = {
        'U_Udagger_minus_I_norm': round(unitarity_err, 8),
        'is_unitary': bool(unitarity_err < 1e-10),
    }
    print("    ||U@U^dag - I||=%.2e, is_unitary: %s" % (unitarity_err, unitarity_err < 1e-10))

    # === Test 3: Born rule ===
    print("  Test 3: Born rule P(k) = |<e_k|psi>|^2...")
    # Standard basis e_k
    e_vecs = np.eye(D, dtype=complex)
    probs = np.array([born_probability(psi, e_vecs[k]) for k in range(D)])
    prob_sum = float(probs.sum())
    results['born_rule'] = {
        'sum_of_probs': round(prob_sum, 8),
        'sums_to_one': bool(abs(prob_sum - 1.0) < 1e-10),
        'max_prob': round(float(probs.max()), 6),
        'min_prob': round(float(probs.min()), 8),
    }
    print("    sum_P(k)=%.8f (should=1.0), is_valid: %s" % (prob_sum, abs(prob_sum - 1.0) < 1e-10))

    # === Test 4: Superposition linearity ===
    print("  Test 4: Superposition linearity (|+> = (|0>+|1>)/sqrt2)...")
    # Use 2D subspace of the D-dim space
    e0 = e_vecs[0]
    e1 = e_vecs[1]
    plus_state = (e0 + e1) / np.sqrt(2)
    minus_state = (e0 - e1) / np.sqrt(2)
    p0_plus = born_probability(plus_state, e0)
    p1_plus = born_probability(plus_state, e1)
    p0_minus = born_probability(minus_state, e0)
    p1_minus = born_probability(minus_state, e1)
    # |+> and |-> should be orthogonal
    inner_plus_minus = abs(np.dot(plus_state.conj(), minus_state))
    results['superposition'] = {
        'P(e0 | +state)': round(p0_plus, 6),
        'P(e1 | +state)': round(p1_plus, 6),
        'P(e0 | -state)': round(p0_minus, 6),
        'P(e1 | -state)': round(p1_minus, 6),
        'inner_plus_minus': round(float(inner_plus_minus), 8),
        'plus_minus_orthogonal': bool(inner_plus_minus < 1e-10),
    }
    print("    |+>: P(e0)=%.4f P(e1)=%.4f" % (p0_plus, p1_plus))
    print("    |->: P(e0)=%.4f P(e1)=%.4f" % (p0_minus, p1_minus))
    print("    |<+|->|=%.2e (should=0)" % inner_plus_minus)

    # === Test 5: Hadamard gate H^2 = I ===
    print("  Test 5: Hadamard H^2 = I in 2D subspace...")
    H2 = np.array([[1, 1], [1, -1]]) / np.sqrt(2)  # 2D Hadamard
    H2H2 = H2 @ H2
    h2_err = float(np.linalg.norm(H2H2 - np.eye(2)))
    psi2 = np.array([1, 0], dtype=complex)  # |0>
    H_psi2 = H2 @ psi2
    HH_psi2 = H2 @ H_psi2
    recovery = float(np.linalg.norm(HH_psi2 - psi2))
    results['hadamard'] = {
        'H2_squared_minus_I_norm': round(h2_err, 8),
        'H2_is_identity': bool(h2_err < 1e-10),
        'H2_psi_norm_error': round(recovery, 8),
        'H_applied_twice_recovers': bool(recovery < 1e-10),
    }
    print("    ||H^2 - I||=%.2e, vector recovery err=%.2e" % (h2_err, recovery))

    # === Test 6: Schmidt decomposition ===
    print("  Test 6: Schmidt decomposition (bipartite entanglement)...")
    d_a, d_b = 8, 8  # 8x8 bipartite system
    # Create maximally entangled state (Bell state analog)
    bell = np.zeros(d_a * d_b, dtype=complex)
    for i in range(d_a):
        bell[i * d_b + i] = 1.0 / np.sqrt(d_a)
    # Create product state (no entanglement)
    psi_a = random_unit_vec(d_a)
    psi_b = random_unit_vec(d_b)
    product = np.outer(psi_a, psi_b).flatten()

    U_bell, S_bell, Vh_bell = schmidt_decomposition(bell, d_a, d_b)
    U_prod, S_prod, Vh_prod = schmidt_decomposition(product, d_a, d_b)

    schmidt_rank_bell = int((S_bell > 1e-8).sum())
    schmidt_rank_prod = int((S_prod > 1e-8).sum())
    results['schmidt'] = {
        'bell_state_schmidt_rank': schmidt_rank_bell,
        'product_state_schmidt_rank': schmidt_rank_prod,
        'bell_sv_top5': [round(float(s), 6) for s in S_bell[:5]],
        'product_sv_top5': [round(float(s), 6) for s in S_prod[:5]],
    }
    print("    Bell state Schmidt rank: %d (should=%d for max entanglement)" % (schmidt_rank_bell, d_a))
    print("    Product state Schmidt rank: %d (should=1, no entanglement)" % schmidt_rank_prod)

    # === PLOT ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Born rule probability distribution
    ax = axes[0]
    sorted_probs = np.sort(probs)[::-1]
    ax.bar(range(min(30, D)), sorted_probs[:30], color='#3F51B5', edgecolor='none', alpha=0.8)
    ax.set_xlabel('Basis state index (sorted by prob)')
    ax.set_ylabel('Probability |<e_k|psi>|^2')
    ax.set_title('Born Rule: Probability Distribution\nRandom %d-dim S-Qubit' % D, fontweight='bold')
    ax.text(0.6, 0.85, 'sum=%.8f' % prob_sum, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.grid(alpha=0.3, axis='y')

    # Panel 2: Schmidt spectra comparison
    ax = axes[1]
    k_show = min(16, d_a)
    ax.plot(range(1, k_show+1), S_bell[:k_show], 'o-', color='#E91E63',
            lw=2, label='Bell (max entangled)\nSchmidt rank=%d' % schmidt_rank_bell)
    ax.plot(range(1, k_show+1), S_prod[:k_show], 's-', color='#4CAF50',
            lw=2, label='Product (separable)\nSchmidt rank=%d' % schmidt_rank_prod)
    ax.set_xlabel('Schmidt index')
    ax.set_ylabel('Schmidt coefficient')
    ax.set_title('Schmidt Decomposition\nEntangled vs Separable', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 3: Summary table
    ax = axes[2]
    ax.axis('off')
    all_pass = all([
        results['inner_product']['cauchy_schwarz_satisfied'],
        results['unitarity']['is_unitary'],
        results['born_rule']['sums_to_one'],
        results['superposition']['plus_minus_orthogonal'],
        results['hadamard']['H2_is_identity'],
        schmidt_rank_prod == 1,
        schmidt_rank_bell == d_a,
    ])
    tests = [
        ('Cauchy-Schwarz', results['inner_product']['cauchy_schwarz_satisfied']),
        ('Unitarity U@U^d=I', results['unitarity']['is_unitary']),
        ('Born rule sum=1', results['born_rule']['sums_to_one']),
        ('|+>perp|->', results['superposition']['plus_minus_orthogonal']),
        ('H^2 = I', results['hadamard']['H2_is_identity']),
        ('Product rank=1', schmidt_rank_prod == 1),
        ('Bell rank=%d' % d_a, schmidt_rank_bell == d_a),
    ]
    summary = "S-Qubit Algebra Verification\n\n"
    for name, passed in tests:
        summary += "  %s  %s\n" % ('[OK]' if passed else '[FAIL]', name)
    summary += "\n  ALL PASSED: %s" % ('YES' if all_pass else 'NO')
    color = '#E8F5E9' if all_pass else '#FFEBEE'
    ax.text(0.05, 0.5, summary, fontsize=12, family='monospace',
            verticalalignment='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.9))

    plt.suptitle(
        'Phase CQ1: Complex S-Qubit Algebra\nNumerical Verification of Quantum-Inspired Properties',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_cq1_complex_algebra.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'CQ1', 'name': 'complex_algebra',
        'dimension': D,
        'all_tests_passed': all_pass,
        'results': results,
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_cq1_complex_algebra.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("\n  CQ1 completed in %.0fs" % (time.time() - start))
    print("  ALL TESTS PASSED: %s" % all_pass)


if __name__ == '__main__':
    main()
