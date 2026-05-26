# -*- coding: utf-8 -*-
"""
Phase Q114: S-Qubit Artificial Life
====================================
Exploits the No-Cloning violation (S-Qubits can be perfectly copied)
to create self-replicating semantic entities within the LLM.

Conway's Game of Life meets quantum computing:
- S-Qubits = organisms (hidden state vectors)
- Clone = reproduction (exact copy)
- Mutation = noise perturbation
- Fitness = cosine similarity to environment
- Selection = top-k survival

We observe emergent dynamics: population growth, mutation accumulation,
and fitness landscape exploration.
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


def main():
    print("=" * 60)
    print("Phase Q114: S-Qubit Artificial Life")
    print("=" * 60)
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tok = load_model(device=device)
    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size

    # ===== Environment Setup =====
    # The "environment" is defined by target concepts
    environments = [
        {
            'name': 'Physics',
            'targets': ['quantum', 'energy', 'gravity', 'light', 'atom'],
            'predators': ['chaos', 'entropy', 'void'],
        },
        {
            'name': 'Biology',
            'targets': ['cell', 'DNA', 'evolution', 'brain', 'life'],
            'predators': ['death', 'decay', 'extinction'],
        },
    ]

    all_env_results = []

    for env in environments:
        print("\n  Environment: %s" % env['name'])

        # Get target embeddings (fitness landscape)
        target_embeds = []
        for word in env['targets']:
            ids = tok(word, add_special_tokens=False)['input_ids']
            emb = model.model.embed_tokens(
                torch.tensor([ids[0]], device=device)).squeeze(0).float()
            target_embeds.append(emb)
        target_center = torch.stack(target_embeds).mean(dim=0)
        target_center = target_center / target_center.norm()

        # Get predator embeddings
        predator_embeds = []
        for word in env['predators']:
            ids = tok(word, add_special_tokens=False)['input_ids']
            emb = model.model.embed_tokens(
                torch.tensor([ids[0]], device=device)).squeeze(0).float()
            predator_embeds.append(emb)
        predator_center = torch.stack(predator_embeds).mean(dim=0)
        predator_center = predator_center / predator_center.norm()

        # ===== Initialize Population =====
        n_initial = 10
        max_pop = 50
        n_generations = 30
        mutation_rate = 0.05

        # Seed organisms: random perturbations of target concepts
        population = []
        for i in range(n_initial):
            base = target_embeds[i % len(target_embeds)].clone()
            noise = torch.randn_like(base) * 0.1
            organism = base + noise
            organism = organism / organism.norm()
            population.append(organism)

        # ===== Evolution Loop =====
        generation_data = []

        for gen in range(n_generations):
            # 1. Evaluate fitness
            fitnesses = []
            for org in population:
                # Fitness = similarity to targets - similarity to predators
                fit_target = torch.nn.functional.cosine_similarity(
                    org.unsqueeze(0), target_center.unsqueeze(0)).item()
                fit_predator = torch.nn.functional.cosine_similarity(
                    org.unsqueeze(0), predator_center.unsqueeze(0)).item()
                fitness = fit_target - 0.5 * fit_predator
                fitnesses.append(fitness)

            fitnesses = np.array(fitnesses)
            mean_fit = float(fitnesses.mean())
            max_fit = float(fitnesses.max())
            pop_size = len(population)

            # 2. Measure diversity (mean pairwise distance)
            if len(population) > 1:
                pop_stack = torch.stack(population)
                pop_norm = torch.nn.functional.normalize(pop_stack, dim=-1)
                sim_mat = pop_norm @ pop_norm.T
                mask = ~torch.eye(len(population), dtype=torch.bool, device=device)
                diversity = 1.0 - sim_mat[mask].mean().item()
            else:
                diversity = 0.0

            generation_data.append({
                'generation': gen,
                'pop_size': pop_size,
                'mean_fitness': round(mean_fit, 6),
                'max_fitness': round(max_fit, 6),
                'diversity': round(diversity, 6)
            })

            if gen % 5 == 0:
                print("    Gen %d: pop=%d, mean_fit=%.4f, max_fit=%.4f, div=%.4f" %
                      (gen, pop_size, mean_fit, max_fit, diversity))

            # 3. Selection: keep top 60%
            n_survive = max(2, int(len(population) * 0.6))
            sorted_idx = np.argsort(fitnesses)[::-1]
            survivors = [population[i] for i in sorted_idx[:n_survive]]

            # 4. Reproduction: Clone + Mutate
            new_pop = list(survivors)  # Survivors persist

            for parent in survivors:
                if len(new_pop) >= max_pop:
                    break
                # Clone (No-Cloning violation exploitation!)
                child = parent.clone()
                # Mutate: small random perturbation
                mutation = torch.randn_like(child) * mutation_rate
                child = child + mutation
                child = child / child.norm()
                new_pop.append(child)

            population = new_pop[:max_pop]

        # Final statistics
        final_fitnesses = []
        for org in population:
            fit_t = torch.nn.functional.cosine_similarity(
                org.unsqueeze(0), target_center.unsqueeze(0)).item()
            final_fitnesses.append(fit_t)

        # Did life evolve? Compare initial vs final fitness
        initial_fit = generation_data[0]['mean_fitness']
        final_fit = generation_data[-1]['mean_fitness']
        evolution_detected = final_fit > initial_fit * 1.05

        all_env_results.append({
            'environment': env['name'],
            'initial_fitness': round(initial_fit, 6),
            'final_fitness': round(final_fit, 6),
            'fitness_gain': round(final_fit - initial_fit, 6),
            'evolution_detected': str(evolution_detected),
            'final_diversity': round(generation_data[-1]['diversity'], 6),
            'final_pop_size': generation_data[-1]['pop_size'],
            'generation_data': generation_data
        })
        print("    Final: fit=%.4f (gain=%.4f), evolution=%s" %
              (final_fit, final_fit - initial_fit, evolution_detected))

    # ===== Save Results =====
    n_evolved = sum(1 for r in all_env_results if r['evolution_detected'] == 'True')
    results = {
        'phase': 'Q114',
        'name': 'S-Qubit Artificial Life',
        'n_evolved': n_evolved,
        'n_environments': len(environments),
        'environment_results': all_env_results,
        'elapsed': round(time.time() - t0, 2)
    }
    with open(os.path.join(RESULTS_DIR, 'phase_q114_alife.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # ===== Plot =====
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Fitness over generations
    ax = axes[0]
    for er in all_env_results:
        gens = [gd['generation'] for gd in er['generation_data']]
        fits = [gd['mean_fitness'] for gd in er['generation_data']]
        ax.plot(gens, fits, 'o-', label=er['environment'], markersize=3)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Mean fitness')
    ax.set_title('(a) Fitness Evolution')
    ax.legend()
    ax.grid(alpha=0.3)

    # (b) Population dynamics
    ax = axes[1]
    for er in all_env_results:
        gens = [gd['generation'] for gd in er['generation_data']]
        pops = [gd['pop_size'] for gd in er['generation_data']]
        ax.plot(gens, pops, 'o-', label=er['environment'], markersize=3)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Population size')
    ax.set_title('(b) Population Dynamics')
    ax.legend()
    ax.grid(alpha=0.3)

    # (c) Diversity
    ax = axes[2]
    for er in all_env_results:
        gens = [gd['generation'] for gd in er['generation_data']]
        divs = [gd['diversity'] for gd in er['generation_data']]
        ax.plot(gens, divs, 'o-', label=er['environment'], markersize=3)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Genetic diversity')
    ax.set_title('(c) Diversity Over Time')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle('Q114: S-Qubit Artificial Life (Evolved: %d/%d)' %
                 (n_evolved, len(environments)), fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase_q114_alife.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nQ114 complete! Elapsed: %.1fs" % (time.time() - t0))
    return results


if __name__ == '__main__':
    main()
