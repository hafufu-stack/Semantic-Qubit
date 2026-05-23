# -*- coding: utf-8 -*-
"""
Phase CQ2: Bloch Sphere Visualization of S-Qubit (CPU, no model)
The Bloch sphere is the standard geometric representation of a qubit |psi>.
For a 2-level system: |psi> = cos(theta/2)|0> + e^{i*phi}*sin(theta/2)|1>
  theta in [0, pi], phi in [0, 2*pi]

S-Qubit analog:
- Our qubit basis is {|MIN>, |MAX>} instead of {|0>, |1>}
- We project the full 1536-dim Soul Vector onto this 2D subspace
- The projected state can be visualized on a Bloch sphere

Experiment:
1. Generate sample S-Qubit states (mixtures of |MIN> and |MAX>)
2. Project to Bloch sphere coordinates (theta, phi)
3. Show: gates (H, Z, X) as rotations on Bloch sphere
4. Show: decoherence as contraction towards center
5. Compare task types' "quantum state" regions on Bloch sphere
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
np.random.seed(42)


def to_bloch(alpha, beta):
    """
    Convert complex amplitudes (alpha, beta) to Bloch sphere (x, y, z).
    |psi> = alpha|0> + beta|1>, |alpha|^2 + |beta|^2 = 1
    x = 2*Re(alpha*beta*)
    y = 2*Im(alpha*beta*)
    z = |alpha|^2 - |beta|^2
    """
    x = 2 * np.real(alpha * np.conj(beta))
    y = 2 * np.imag(alpha * np.conj(beta))
    z = np.abs(alpha)**2 - np.abs(beta)**2
    return np.array([x, y, z])


def state_from_theta_phi(theta, phi):
    """Create |psi> = cos(theta/2)|0> + e^{i*phi}*sin(theta/2)|1>."""
    alpha = np.cos(theta / 2)
    beta = np.exp(1j * phi) * np.sin(theta / 2)
    return alpha, beta


def apply_gate_2d(gate, alpha, beta):
    """Apply 2x2 gate to (alpha, beta) state vector."""
    psi = np.array([alpha, beta])
    psi_out = gate @ psi
    return psi_out[0], psi_out[1]


def gate_X():
    """Pauli X (NOT gate): flips |0><->|1>"""
    return np.array([[0, 1], [1, 0]], dtype=complex)


def gate_Y():
    """Pauli Y: rotation around Y axis"""
    return np.array([[0, -1j], [1j, 0]], dtype=complex)


def gate_Z():
    """Pauli Z: phase flip on |1>"""
    return np.array([[1, 0], [0, -1]], dtype=complex)


def gate_H():
    """Hadamard: |0> -> |+>, |1> -> |->"""
    return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def gate_S():
    """Phase gate S: |1> -> i|1>"""
    return np.array([[1, 0], [0, 1j]], dtype=complex)


def gate_T():
    """T gate: |1> -> e^{i*pi/4}|1>"""
    return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


def draw_bloch_sphere(ax, title="Bloch Sphere"):
    """Draw Bloch sphere wireframe."""
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, alpha=0.04, color='lightblue')
    ax.plot_wireframe(x, y, z, rstride=5, cstride=5, alpha=0.08, color='blue', linewidth=0.5)
    # Axes
    for vec, label in [([0,0,1.2], '|MIN>'), ([0,0,-1.2], '|MAX>'),
                        ([1.2,0,0], '|+>'), ([-1.2,0,0], '|->'),
                        ([0,1.2,0], '|+i>'), ([0,-1.2,0], '|-i>')]:
        ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], color='gray', alpha=0.4, arrow_length_ratio=0.1)
        ax.text(vec[0]*1.05, vec[1]*1.05, vec[2]*1.05, label, fontsize=7, ha='center')
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_zlim(-1.5, 1.5)
    ax.set_box_aspect([1,1,1])
    ax.set_title(title, fontweight='bold', fontsize=9)
    ax.axis('off')


def main():
    print("[CQ2] Bloch Sphere Visualization of S-Qubit")
    start = time.time()
    results = {}

    # Standard states
    states = {
        '|MIN>=|0>':   (1.0+0j,  0.0+0j),
        '|MAX>=|1>':   (0.0+0j,  1.0+0j),
        '|+>=(|0>+|1>)/sqrt2': (1/np.sqrt(2)+0j, 1/np.sqrt(2)+0j),
        '|->=(|0>-|1>)/sqrt2': (1/np.sqrt(2)+0j, -1/np.sqrt(2)+0j),
        '|+i>=(|0>+i|1>)/sqrt2': (1/np.sqrt(2)+0j, 0+1j/np.sqrt(2)),
        '|-i>=(|0>-i|1>)/sqrt2': (1/np.sqrt(2)+0j, 0-1j/np.sqrt(2)),
    }

    # Compute Bloch vectors
    bloch_vecs = {}
    for name, (a, b) in states.items():
        bv = to_bloch(a, b)
        bloch_vecs[name] = bv.tolist()
        print("  %-40s -> Bloch(%+.3f, %+.3f, %+.3f)" % (name[:40], bv[0], bv[1], bv[2]))

    # Gate trajectories on Bloch sphere
    gates = {'H': gate_H(), 'X': gate_X(), 'Y': gate_Y(), 'Z': gate_Z(),
             'S': gate_S(), 'T': gate_T()}
    start_state = (1.0+0j, 0.0+0j)  # |MIN> = |0>

    gate_results = {}
    for gname, G in gates.items():
        a_out, b_out = apply_gate_2d(G, start_state[0], start_state[1])
        bv = to_bloch(a_out, b_out)
        gate_results[gname] = bv.tolist()
        print("  %s|MIN> -> Bloch(%+.3f, %+.3f, %+.3f)" % (gname, bv[0], bv[1], bv[2]))

    # Decoherence simulation: mixed state p|psi><psi| + (1-p)/2 * I
    # Bloch vector shrinks: r_mixed = p * r_pure
    decoherence_ps = np.linspace(0, 1, 11)
    pure_bv = to_bloch(*start_state)
    decoherence_path = [(p, (p * pure_bv).tolist()) for p in decoherence_ps]

    # Simulate "semantic" states from different tasks (synthetic)
    np.random.seed(42)
    task_bloch = {
        'min_task':    [np.random.randn(3)*0.1 + np.array([0, 0, 0.9]) for _ in range(10)],
        'max_task':    [np.random.randn(3)*0.1 + np.array([0, 0, -0.9]) for _ in range(10)],
        'arithmetic':  [np.random.randn(3)*0.2 + np.array([0.6, 0, 0]) for _ in range(10)],
        'natural_lang':[np.random.randn(3)*0.3 + np.array([0, 0, 0]) for _ in range(10)],  # mixed
    }
    # Normalize to surface
    for task, pts in task_bloch.items():
        task_bloch[task] = [(pt / (np.linalg.norm(pt)+1e-8) * 0.85).tolist() for pt in pts]

    # === PLOT ===
    fig = plt.figure(figsize=(18, 6))

    # Panel 1: Standard states + gates on Bloch sphere
    ax1 = fig.add_subplot(131, projection='3d')
    draw_bloch_sphere(ax1, "Standard States + Gates")
    colors = {'|MIN>=|0>': '#E91E63', '|MAX>=|1>': '#2196F3',
              '|+>=(|0>+|1>)/sqrt2': '#4CAF50', '|->=(|0>-|1>)/sqrt2': '#FF9800',
              '|+i>=(|0>+i|1>)/sqrt2': '#9C27B0', '|-i>=(|0>-i|1>)/sqrt2': '#00BCD4'}
    for name, bv in bloch_vecs.items():
        ax1.scatter(*bv, s=100, c=colors.get(name, 'gray'), zorder=5)
    gate_cols = {'H': '#FF5722', 'X': '#795548', 'Y': '#607D8B',
                 'Z': '#009688', 'S': '#F44336', 'T': '#673AB7'}
    for gname, bv in gate_results.items():
        ax1.scatter(*bv, marker='^', s=80, c=gate_cols.get(gname, 'red'), zorder=5)
        ax1.text(bv[0], bv[1], bv[2]+0.1, gname+' gate', fontsize=6)

    # Panel 2: Decoherence path
    ax2 = fig.add_subplot(132, projection='3d')
    draw_bloch_sphere(ax2, "Decoherence: |psi> -> Mixed State")
    dec_pts = np.array([bv for p, bv in decoherence_path])
    ax2.plot(dec_pts[:, 0], dec_pts[:, 1], dec_pts[:, 2],
             'r-o', lw=2, ms=5, label='r=p*r_pure')
    ax2.scatter([dec_pts[-1, 0]], [dec_pts[-1, 1]], [dec_pts[-1, 2]],
                s=100, c='red', zorder=10)
    ax2.scatter([0], [0], [0], s=80, c='black', zorder=10, label='Maximally mixed')
    ax2.legend(fontsize=7)

    # Panel 3: Semantic task regions
    ax3 = fig.add_subplot(133, projection='3d')
    draw_bloch_sphere(ax3, "Semantic Task Regions")
    task_colors = {'min_task': '#E91E63', 'max_task': '#2196F3',
                   'arithmetic': '#FF9800', 'natural_lang': '#4CAF50'}
    for task, pts in task_bloch.items():
        pts_arr = np.array(pts)
        ax3.scatter(pts_arr[:, 0], pts_arr[:, 1], pts_arr[:, 2],
                    s=60, c=task_colors[task], alpha=0.7, label=task)
    ax3.legend(fontsize=7)

    plt.suptitle(
        'Phase CQ2: S-Qubit Bloch Sphere Visualization\n'
        'Semantic states as points on Bloch sphere',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_cq2_bloch_sphere_viz.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    output = {
        'phase': 'CQ2', 'name': 'bloch_sphere_viz',
        'standard_bloch_vectors': bloch_vecs,
        'gate_outputs': gate_results,
        'decoherence_path': [(p, bv) for p, bv in decoherence_path],
        'elapsed': round(time.time() - start, 1),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_cq2_bloch_sphere_viz.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("\n  CQ2 completed in %.0fs" % (time.time() - start))


if __name__ == '__main__':
    main()
