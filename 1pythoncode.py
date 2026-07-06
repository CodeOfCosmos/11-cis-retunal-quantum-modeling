import numpy as np
import matplotlib.pyplot as plt

# 1. System parameters
L = 1.0
num_steps = 180000
burn_in = 40000

# Quantum numbers config
qn_ground = np.array([1, 2, 3, 4, 5, 6])
qn_excited = np.array([1, 2, 3, 4, 5, 7]) # last electron jumps 6 -> 7

def single_particle_psi(n, x):
    return np.sqrt(2.0 / L) * np.sin(n * np.pi * x / L)

def eval_6x6_determinant(positions, quantum_numbers):
    """Computes the 6x6 Slater determinant for a given spin channel configurations."""
    matrix = np.zeros((6, 6))
    for row_idx, n in enumerate(quantum_numbers):
        for col_idx, x in enumerate(positions):
            matrix[row_idx, col_idx] = single_particle_psi(n, x)
    # Norm factor for 6 particles is 1/sqrt(6!)
    return np.linalg.det(matrix) / np.sqrt(720)

def get_density(p, is_excited=False):
    """Evaluates probability density |Psi_up|^2 * |Psi_down|^2 for 12 coordinates."""
    if np.any(p <= 0) or np.any(p >= L):
        return 0.0
    
    # 12 particles split into 6 spin-up and 6 spin-down
    psi_up = eval_6x6_determinant(p[0:6], qn_ground)
    
    # After light hit, the down-spin channel contains the excited electron
    qn_down = qn_excited if is_excited else qn_ground
    psi_down = eval_6x6_determinant(p[6:12], qn_down)
    
    return (psi_up * psi_down) ** 2

def run_mcmc(is_excited=False):
    np.random.seed(42 if not is_excited else 7)
    current_positions = np.linspace(0.05, 0.95, 12)
    current_prob = get_density(current_positions, is_excited)
    samples = []
    
    for step in range(num_steps):
        proposal = current_positions + np.random.normal(0, 0.03, 12)
        proposal_prob = get_density(proposal, is_excited)
        
        if proposal_prob > 0:
            if current_prob == 0 or np.random.rand() < (proposal_prob / current_prob):
                current_positions = proposal
                current_prob = proposal_prob
                
        if step >= burn_in and step % 15 == 0:
            samples.append(current_positions.copy())
            
    return np.array(samples).flatten()

# 2. Execute Simulations
print("Simulating ground state (Before light hit)...")
coords_before = run_mcmc(is_excited=False)

print("Simulating excited state (After light hit)...")
coords_after = run_mcmc(is_excited=True)

# 3. Plotting Comparative Density Profiles
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
x_grid = np.linspace(0, L, 600)

# Panel 1: Before Light Hit
ax1.hist(coords_before, bins=90, range=(0, L), density=True, color='teal', alpha=0.6, edgecolor='darkslategrey')
theory_before = (sum(single_particle_psi(n, x_grid)**2 for n in qn_ground) * 2) / 12.0
ax1.plot(x_grid, theory_before, color='darkorange', linewidth=2.5, label='Analytical Ground State')
ax1.set_title('Before Light interaction (Ground State: All $n \leq 6$)', fontsize=12)
ax1.set_xlabel('Box Position (x)')
ax1.set_ylabel('Relative Probability Density')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Panel 2: After Light Hit
ax2.hist(coords_after, bins=90, range=(0, L), density=True, color='crimson', alpha=0.5, edgecolor='maroon')
theory_after = (sum(single_particle_psi(n, x_grid)**2 for n in qn_ground) + 
                sum(single_particle_psi(n, x_grid)**2 for n in qn_excited)) / 12.0
ax2.plot(x_grid, theory_after, color='blue', linewidth=2.5, label='Analytical Excited State')
ax2.set_title('After interaction with light (Excited State: One Electron to $n=7$)', fontsize=12)
ax2.set_xlabel('Box Position (x)')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()
