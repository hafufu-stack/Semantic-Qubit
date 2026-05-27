# -*- coding: utf-8 -*-
"""
Phase Q196: Quantum Protein Folding (HP Lattice Model)
=========================================================
Map the HP (Hydrophobic-Polar) protein folding problem
onto a quantum Hamiltonian and solve with Embedding VQE.

The HP model on a 2D lattice is NP-hard.
If LLM VQE finds the minimum energy conformation
-> quantum approach to drug discovery.
"""
import os, sys, json, time, gc
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def build_hp_hamiltonian(sequence, dim=16):
    """
    Build HP lattice folding Hamiltonian.
    sequence: string of 'H' (hydrophobic) and 'P' (polar)
    
    Encoding: each residue position uses 2 qubits (4 directions)
    Penalty for overlaps, reward for H-H contacts.
    """
    n_res = len(sequence)
    
    # For small sequences, encode turn directions
    # 0=straight, 1=left, 2=right (3 choices per turn)
    n_turns = n_res - 2  # first two residues fix the frame
    
    # Build Hamiltonian matrix
    H = np.zeros((dim, dim))
    
    # For each basis state, compute energy
    for state_idx in range(dim):
        # Decode state to turn sequence
        turns = []
        s = state_idx
        for _ in range(n_turns):
            turns.append(s % 3)
            s //= 3
        
        # Compute 2D coordinates from turns
        coords = [(0, 0), (1, 0)]  # First two positions fixed
        dx, dy = 1, 0  # Initial direction
        
        for turn in turns:
            if turn == 0:  # Straight
                pass
            elif turn == 1:  # Left
                dx, dy = -dy, dx
            else:  # Right
                dx, dy = dy, -dx
            
            new_pos = (coords[-1][0] + dx, coords[-1][1] + dy)
            coords.append(new_pos)
        
        # Check for overlaps (penalty)
        overlap_penalty = 0
        for i in range(len(coords)):
            for j in range(i + 2, len(coords)):
                if i + 1 < len(coords) and j < len(coords):
                    if coords[i] == coords[j]:
                        overlap_penalty += 10  # Moderate penalty
        
        # Count H-H contacts (reward)
        hh_contacts = 0
        for i in range(min(len(coords), n_res)):
            for j in range(i + 2, min(len(coords), n_res)):
                if i < len(sequence) and j < len(sequence):
                    if sequence[i] == 'H' and sequence[j] == 'H':
                        di = abs(coords[i][0] - coords[j][0])
                        dj = abs(coords[i][1] - coords[j][1])
                        if di + dj == 1:  # Adjacent on lattice
                            hh_contacts += 1
        
        H[state_idx, state_idx] = overlap_penalty - hh_contacts
    
    # Add off-diagonal coupling (quantum tunneling between conformations)
    for i in range(dim):
        for j in range(i + 1, dim):
            # Hamming distance 1 in turn space -> coupling
            diff = abs(i - j)
            if diff in [1, 3, 9]:  # Single turn change
                coupling = -0.1 * np.exp(-abs(H[i,i] - H[j,j]) * 0.1)
                H[i, j] = coupling
                H[j, i] = coupling
    
    return H


def main():
    print("=" * 60)
    print("Phase Q196: Quantum Protein Folding")
    print("  (HP Lattice Model - NP-Hard Problem)")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device, dtype=torch.float32)
    embed_layer = model.model.embed_tokens

    # Test sequences (short HP proteins)
    test_sequences = [
        {'seq': 'HPHP', 'name': '4-mer', 'expected_contacts': 1},
        {'seq': 'HHPPHH', 'name': '6-mer', 'expected_contacts': 2},
        {'seq': 'HPPHPPHP', 'name': '8-mer', 'expected_contacts': 2},
        {'seq': 'HHPPHHPPHH', 'name': '10-mer', 'expected_contacts': 4},
        {'seq': 'HPHPPHHPHP', 'name': '10-mer-v2', 'expected_contacts': 3},
    ]

    dim = 16
    results_list = []

    for test in test_sequences:
        seq = test['seq']
        name = test['name']
        print("\n--- %s: %s (%d residues) ---" % (name, seq, len(seq)))

        H_np = build_hp_hamiltonian(seq, dim)
        H_torch = torch.tensor(H_np, dtype=torch.float32, device=device)

        # Exact solution
        evals, evecs = np.linalg.eigh(H_np)
        E_exact = evals[0]
        psi_exact = evecs[:, 0]

        # Count exact H-H contacts in ground state
        print("  Exact E0: %.4f" % E_exact)

        # VQE
        seed = "Protein fold %s energy:" % seq
        seed_ids = tok(seed, return_tensors='pt')['input_ids'].to(device)
        seed_embeds = embed_layer(seed_ids).detach().clone()
        opt = seed_embeds.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([opt], lr=0.002)

        energies = []
        for step in range(300):
            optimizer.zero_grad()
            outputs = model(inputs_embeds=opt.float(), output_hidden_states=True)
            h = outputs.hidden_states[-1][0, -1, :]
            psi = h[:dim]
            psi_n = psi / (torch.norm(psi) + 1e-10)
            E = psi_n @ H_torch @ psi_n
            E.backward()
            torch.nn.utils.clip_grad_norm_([opt], max_norm=1.0)
            optimizer.step()
            e_val = float(E.detach())
            if not np.isfinite(e_val):
                e_val = energies[-1] if energies else 0.0
            energies.append(e_val)

        E_vqe = energies[-1] if np.isfinite(energies[-1]) else 999.0
        error = abs(E_vqe - E_exact) * 1000
        if not np.isfinite(error):
            error = 9999.0

        # Fidelity with exact ground state
        with torch.no_grad():
            outputs = model(inputs_embeds=opt.float(), output_hidden_states=True)
            psi_final = outputs.hidden_states[-1][0, -1, :][:dim].float()
            psi_final = psi_final / (torch.norm(psi_final) + 1e-10)
            psi_exact_torch = torch.tensor(psi_exact, dtype=torch.float32, device=device)
            fidelity = float(torch.dot(psi_final, psi_exact_torch) ** 2)

        result = {
            'name': name,
            'sequence': seq,
            'length': len(seq),
            'E_exact': round(E_exact, 4),
            'E_vqe': round(E_vqe, 4),
            'error_mHa': round(error, 2),
            'fidelity': round(fidelity, 4),
            'chem_acc': 1 if error < 1.6 else 0,
        }
        results_list.append(result)

        print("  E_exact=%.4f, E_vqe=%.4f, err=%.2f mHa, fid=%.4f" %
              (E_exact, E_vqe, error, fidelity))

    # Summary
    n_chem = sum(1 for r in results_list if r['chem_acc'])
    avg_fid = float(np.mean([r['fidelity'] for r in results_list]))
    avg_err = float(np.mean([r['error_mHa'] for r in results_list]))

    print("\n--- Summary ---")
    print("  Chemical accuracy: %d/%d" % (n_chem, len(results_list)))
    print("  Avg fidelity: %.4f" % avg_fid)
    print("  Avg error: %.2f mHa" % avg_err)

    if n_chem == len(results_list):
        verdict = "PERFECT: All %d protein folds at chemical accuracy" % len(results_list)
    else:
        verdict = "PARTIAL: %d/%d at chemical accuracy, avg fid=%.3f" % (
            n_chem, len(results_list), avg_fid)
    print("  Verdict: %s" % verdict)

    # Save
    results = {
        'phase': 'Q196',
        'name': 'Quantum Protein Folding',
        'proteins': results_list,
        'summary': {
            'chem_accuracy': '%d/%d' % (n_chem, len(results_list)),
            'avg_fidelity': round(avg_fid, 4),
            'avg_error_mHa': round(avg_err, 2),
            'verdict': verdict,
        },
        'elapsed': round(time.time() - t0, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q196_protein.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Energy comparison
    ax = axes[0]
    names = [r['name'] for r in results_list]
    e_exact = [r['E_exact'] for r in results_list]
    e_vqe = [r['E_vqe'] for r in results_list]
    x = np.arange(len(names))
    ax.bar(x - 0.15, e_exact, 0.3, color='black', alpha=0.7, label='Exact')
    ax.bar(x + 0.15, e_vqe, 0.3, color='#E91E63', alpha=0.7, label='LLM VQE')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=8)
    ax.set_ylabel('Energy')
    ax.set_title('(a) Protein Folding Energy\n(Exact vs LLM VQE)')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # (b) Error
    ax = axes[1]
    errors = [r['error_mHa'] for r in results_list]
    colors = ['#4CAF50' if e < 1.6 else '#F44336' for e in errors]
    ax.bar(x, errors, color=colors, edgecolor='black', alpha=0.85)
    ax.axhline(1.6, color='green', ls='--', label='Chemical accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=8)
    ax.set_ylabel('Error (mHa)')
    ax.set_title('(b) VQE Error\n(%d/%d at chemical accuracy)' % (n_chem, len(results_list)))
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # (c) Fidelity
    ax = axes[2]
    fids = [r['fidelity'] for r in results_list]
    ax.bar(x, fids, color='#2196F3', edgecolor='black', alpha=0.85)
    ax.axhline(0.99, color='green', ls='--', label='99% fidelity')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=8)
    ax.set_ylabel('Fidelity')
    ax.set_title('(c) Wavefunction Fidelity\n(avg=%.3f)' % avg_fid)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)

    plt.suptitle('Q196: Quantum Protein Folding\n%s' % verdict[:60],
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q196_protein.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\nQ196 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
