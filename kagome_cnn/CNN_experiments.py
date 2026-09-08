"""
================================================================================
 kagome_cnn.py  --  size-agnostic, multi-task CNN for the kagome Hubbard model
================================================================================

WHY A CNN AND NOT A GNN
-----------------------
Kagome is NOT an arbitrary graph. It is a triangular Bravais lattice with a
3-site basis (sublattices A, B, C), so it maps exactly onto a tensor

        (L, L, 3)      L x L unit cells,  3 sublattice channels

Every kagome bond fits inside a 3x3 kernel window (verified: cell offsets are
only -1, 0, +1 in each direction).

A message-passing GNN aggregates over a site's neighbours in a
permutation-invariant way, which DISCARDS which neighbour is which. On kagome
the up-triangle and down-triangle bonds are geometrically distinct but identical
as graph edges, so the GNN sees {same,same,same,same} and the frustration
structure is destroyed at the first layer. (Formally: kagome and the square
lattice are both 4-regular, hence indistinguishable by 1-WL with uniform node
features; and message passing provably cannot count triangles.)

A convolution kernel does not throw that away. Kernel position (-1,0) is a
DIFFERENT learnable weight from (+1,0), so both triangles are visible for free.

SIZE-AGNOSTICISM
----------------
Comes free IF the network is fully convolutional. No dense layers anywhere.
Use mean pooling (not sum) for global quantities. Then the same weights run on
L=2, 4, 8 ... unit cells.

WHAT THIS FILE CONTAINS
-----------------------
  1. Kagome lattice geometry + bond table            (pure numpy, testable)
  2. Synthetic data generator (physics-flavoured)    -- so you can run TODAY,
                                                        before ED data exists
  3. The model                                       (PyTorch)
  4. Multi-task loss with hard-wired physics
  5. Baselines you MUST beat before believing anything
  6. Sanity checks against known kagome values

Requires: numpy, torch.   Run:  python kagome_cnn.py
================================================================================
"""

import numpy as np
import time

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAVE_TORCH = True
except ImportError as I:
    HAVE_TORCH = False
    print("[warn] torch not found -- geometry and baseline sections will still run")
    print(I)


# =============================================================================
# 1. KAGOME GEOMETRY
# =============================================================================

A1 = np.array([1.0, 0.0])                    # Bravais vectors (triangular)
A2 = np.array([0.5, np.sqrt(3) / 2])
BASIS = {0: np.array([0.0, 0.0]),            # A
         1: A1 / 2,                          # B
         2: A2 / 2}                          # C

# The 12 directed bonds, as (d_row, d_col, sublattice_from, sublattice_to).
# Verified by distance search: these are ALL nearest-neighbour bonds, degree 4.
BOND_TABLE = [
    (-1,  0, 0, 2), (-1, +1, 1, 2),
    ( 0, -1, 0, 1), ( 0,  0, 0, 1), ( 0,  0, 0, 2),
    ( 0,  0, 1, 0), ( 0,  0, 1, 2), ( 0,  0, 2, 0), ( 0,  0, 2, 1),
    ( 0, +1, 1, 0), (+1, -1, 2, 1), (+1,  0, 2, 0),
]


def site_positions(L):
    """Cartesian position of every site. Returns array of shape (L, L, 3, 2)."""
    pos = np.zeros((L, L, 3, 2))
    for r in range(L):
        for c in range(L):
            for s in range(3):
                pos[r, c, s] = r * A2 + c * A1 + BASIS[s]
    return pos


def flat_index(r, c, s, L):
    """Map (row, col, sublattice) -> flat site index in [0, 3L^2)."""
    return (r * L + c) * 3 + s


def build_adjacency(L):
    """Nearest-neighbour adjacency list, periodic. Use this to build ED
    Hamiltonians so the NN and the ED data agree on the lattice."""
    adj = {i: [] for i in range(3 * L * L)}
    for r in range(L):
        for c in range(L):
            for (dr, dc, s_from, s_to) in BOND_TABLE:
                i = flat_index(r, c, s_from, L)
                j = flat_index((r + dr) % L, (c + dc) % L, s_to, L)
                if j not in adj[i]:
                    adj[i].append(j)
    return adj


def pair_displacements(L):
    """Minimum-image relative displacement r_ij for every ordered pair.
    Returns (N, N, 2) with N = 3L^2. Used as input to the pair decoder."""
    pos = site_positions(L).reshape(-1, 2)          # (N, 2)
    N = pos.shape[0]
    # lattice vectors of the supercell
    T1, T2 = L * A1, L * A2
    best = np.full((N, N, 2), np.inf)
    best_norm = np.full((N, N), np.inf)
    for m in (-1, 0, 1):
        for n in (-1, 0, 1):
            shift = m * T1 + n * T2
            d = pos[None, :, :] - pos[:, None, :] + shift
            nrm = np.linalg.norm(d, axis=-1)
            take = nrm < best_norm
            best_norm[take] = nrm[take]
            best[take] = d[take]
    return best, best_norm


# =============================================================================
# 2. SYNTHETIC DATA  (placeholder -- replace with ED as soon as it exists)
# =============================================================================
# Physics put in deliberately, so you can verify the whole pipeline works
# BEFORE spending compute on exact diagonalization:
#   - correlations decay exponentially, xi ~ 1 lattice spacing
#   - nearest neighbour NEGATIVE  (target ~ -0.22 in the Heisenberg limit)
#   - second neighbour small POSITIVE
#   - double occupancy falls as U/t grows
# If the model cannot learn THIS, it will not learn ED data either.

def synthetic_sample(L, U, filling, rng):
    rij, dist = pair_displacements(L)
    N = dist.shape[0]

    # --- local observables ---
    n_i = np.full(N, filling)
    d_i = np.full(N, 0.25 * filling ** 2 / (1.0 + 0.35 * U))   # double occupancy

    # --- spin correlations ---
    xi = 1.0 + 2.0 / (1.0 + 0.3 * U)          # correlation length shrinks with U
    with np.errstate(divide="ignore"):
        env = np.exp(-dist / xi)

    # radial sign structure: NN negative, 2nd neighbour positive, alternating
    radial = np.cos(2 * np.pi * (dist - 0.5))

    # ANGULAR structure with the p6m symmetry of kagome.  This is what splits
    # pairs at the SAME distance into inequivalent sets (as really happens for
    # third neighbours on kagome: four antiferromagnetic + two ferromagnetic).
    # It is also what makes the distance-only baseline beatable -- without it
    # the benchmark is vacuous.
    theta = np.arctan2(rij[..., 1], rij[..., 0])
    angular = 0.35 * np.cos(6 * theta)

    C = -env * (radial + angular)
    C = 0.5 * (C + C.T)                        # correlations are symmetric

    # calibrate so the nearest-neighbour value approaches the known kagome
    # Heisenberg result  <S.S>_nn ~ -0.22  as U grows
    nn_mask = (dist > 0.4) & (dist < 0.6)
    target_nn = -0.22 * U / (U + 2.0)
    C *= target_nn / C[nn_mask].mean()

    C += 0.002 * rng.standard_normal(C.shape)  # measurement-like noise
    C = 0.5 * (C + C.T)

    # exact on-site identity  <S_i.S_i> = (3/4)(n_i - 2 d_i)   -- set LAST
    np.fill_diagonal(C, 0.75 * (n_i - 2 * d_i))
    return dict(U=U, filling=filling, L=L, n=n_i, d=d_i, C=C)


def make_dataset(L, n_samples, rng):
    out = []
    for i in range(n_samples):
        print("Making dataset: ", i+1, "/", n_samples)
        U = rng.uniform(0.5, 12.0)
        nu = rng.uniform(0.7, 1.0)
        t1 = rng.uniform(-0.5, 0.5)
        #out.append(synthetic_sample(L, U, nu, rng))
        current_time = time.time()
        out.append(solve_exact_diagonalization(L, U, nu, t1))
        total_time = time.time() - current_time
        print("\tTime: ", total_time)
    return out


# =============================================================================
# 3. THE MODEL
# =============================================================================

if HAVE_TORCH:

    class KagomeCNN(nn.Module):
        """Fully convolutional. NO dense layers -> works at any L.

        Input : per-site scalars packed as (B, F_in*3, L, L)
        Output: node embeddings h of shape (B, C, L, L, 3)
        """

        def __init__(self, f_in=2, width=64, depth=4):
            super().__init__()
            layers = []
            c_in = f_in * 3                       # 3 sublattices folded into channels
            for _ in range(depth):
                layers += [
                    nn.Conv2d(c_in, width * 3, kernel_size=3,
                              padding=1, padding_mode="circular"),  # periodic BC
                    nn.SiLU(),
                ]
                c_in = width * 3
            self.body = nn.Sequential(*layers)
            self.width = width

            # ---- node head: 1x1 conv, so still fully convolutional ----
            # predicts  [n_i, d_i]  only.  <S_i.S_i> is DERIVED, never predicted.
            self.node_head = nn.Conv2d(width * 3, 2 * 3, kernel_size=1)

            # ---- pair decoder: f(h_i, h_j, r_ij) -> C_ij ----
            # This replaces an N x N output layer, which would pin the size.
            self.pair = nn.Sequential(
                nn.Linear(2 * width + 3, 128), nn.SiLU(),
                nn.Linear(128, 128), nn.SiLU(),
                nn.Linear(128, 1),
            )

        def forward(self, x, rij, dist):
            """
            x    : (B, F_in*3, L, L)
            rij  : (B, N, N, 2) minimum-image displacements
            dist : (B, N, N)    distances
            """
            B, _, L, _ = x.shape
            h = self.body(x)                              # (B, 3W, L, L)

            # ---- node observables ----
            node = self.node_head(h)                      # (B, 6, L, L)
            node = node.view(B, 2, 3, L, L)
            n_i = torch.sigmoid(node[:, 0]) * 2.0         # density in [0, 2]
            d_i = torch.sigmoid(node[:, 1])               # double occ in [0, 1]
            # physical constraint: d_i <= n_i/2
            d_i = torch.minimum(d_i, n_i / 2)

            # ---- reshape embeddings to a flat site list ----
            hs = h.view(B, self.width, 3, L, L)           # (B, W, 3, L, L)
            hs = hs.permute(0, 3, 4, 2, 1).reshape(B, 3 * L * L, self.width)

            # ---- pair decoder over all pairs ----
            N = hs.shape[1]
            hi = hs[:, :, None, :].expand(B, N, N, self.width)
            hj = hs[:, None, :, :].expand(B, N, N, self.width)
            feat = torch.cat([hi, hj, rij, dist[..., None]], dim=-1)
            Cij = self.pair(feat).squeeze(-1)             # (B, N, N)
            Cij = 0.5 * (Cij + Cij.transpose(1, 2))       # symmetry: C_ij = C_ji

            # ---- HARD-WIRED PHYSICS ----
            n_flat = n_i.permute(0, 2, 3, 1).reshape(B, N)
            d_flat = d_i.permute(0, 2, 3, 1).reshape(B, N)
            diag = 0.75 * (n_flat - 2 * d_flat)           # exact on-site identity
            Cij = Cij - torch.diag_embed(torch.diagonal(Cij, dim1=1, dim2=2))
            Cij = Cij + torch.diag_embed(diag)

            return dict(n=n_flat, d=d_flat, C=Cij)


    # =========================================================================
    # 4. LOSS  --  multi-task with learned uncertainty weighting
    # =========================================================================

    class MultiTaskLoss(nn.Module):
        """Kendall, Gal & Cipolla (CVPR 2018): learn a per-task noise term
        instead of hand-tuning weights. Prevents the largest-scale target from
        swamping the small correlators, which is the usual failure mode."""

        def __init__(self, n_tasks=3):
            super().__init__()
            self.log_sigma = nn.Parameter(torch.zeros(n_tasks))

        def forward(self, losses):
            total = 0.0
            for k, Lk in enumerate(losses):
                total = total + torch.exp(-self.log_sigma[k]) * Lk + self.log_sigma[k]
            return total


    def correlation_loss(pred, target, eps=0.02):
        """Correlations span orders of magnitude and change sign, so plain MSE
        only ever learns the nearest-neighbour term. Weight by inverse magnitude
        so a 10% error on a small distant correlator counts like a 10% error on
        the dominant one."""
        w = 1.0 / (target.abs() + eps)
        return (w * (pred - target) ** 2).mean()


    def energy_from_observables(d, U, kinetic):
        """E = -t sum <c^dag c>  +  U sum <n_up n_dn>.
        Compute the energy, never predict it. Removes a head, removes a
        competing gradient, and gives an exact constraint for free."""
        return kinetic + U * d.sum(dim=1)


# =============================================================================
# 5. BASELINES  --  run these FIRST. If the CNN does not beat them, stop.
# =============================================================================

def _mse(pred_fn, data):
    return float(np.mean([np.mean((s["C"] - pred_fn(s)) ** 2) for s in data]))


def baseline_global_mean(train, test):
    """Level 0: one number for every entry. The floor."""
    mu = np.mean([s["C"].mean() for s in train])
    return _mse(lambda s: mu, test)


def baseline_mean_matrix(train, test):
    """Level 1: the sample-averaged matrix. Captures spatial structure but is
    blind to U."""
    Cbar = np.mean([s["C"] for s in train], axis=0)
    return _mse(lambda s: Cbar, test)


def baseline_distance_U(train, test, L, nd=20, nu=12):
    """Level 2: THE BASELINE THAT MATTERS.  Predicts C_ij from |r_i - r_j| AND
    U, i.e. everything except the angular / sublattice structure.  A model that
    cannot beat this has learned nothing that the radial profile does not
    already give away -- which is exactly the failure mode of a plain GNN."""
    _, dist = pair_displacements(L)
    dbins = np.linspace(0, dist.max() + 1e-9, nd + 1)
    Us = np.array([s["U"] for s in train])
    ubins = np.linspace(Us.min(), Us.max() + 1e-9, nu + 1)
    didx = np.clip(np.digitize(dist.ravel(), dbins) - 1, 0, nd - 1)

    table = np.zeros((nu, nd)); count = np.zeros((nu, nd))
    for s in train:
        u = np.clip(np.digitize(s["U"], ubins) - 1, 0, nu - 1)
        vals = s["C"].ravel()
        np.add.at(table[u], didx, vals)
        np.add.at(count[u], didx, 1.0)
    table /= np.maximum(count, 1.0)

    def pred(s):
        u = np.clip(np.digitize(s["U"], ubins) - 1, 0, nu - 1)
        return table[u][didx].reshape(dist.shape)
    return _mse(pred, test)


# =============================================================================
# 6. SANITY CHECKS  --  physics, not loss curves
# =============================================================================

def sanity_checks(sample, L, verbose=True):
    """Run these on EVERY model output. A good loss curve means nothing if
    these fail."""
    C = sample["C"]
    _, dist = pair_displacements(L)
    nn_mask = (dist > 0.4) & (dist < 0.6)          # nearest neighbours
    results = {
        "nn_correlation": C[nn_mask].mean(),        # target ~ -0.22 (Heisenberg)
        "sum_rule": C.sum(),                        # = 0 for a singlet
        "symmetry_violation": np.abs(C - C.T).max(),# must be ~0
        "diag_identity_error": np.abs(
            np.diag(C) - 0.75 * (sample["n"] - 2 * sample["d"])).max(),
    }
    if verbose:
        print("  nearest-neighbour <S.S>   : "
              f"{results['nn_correlation']:+.4f}   (Heisenberg limit ~ -0.22)")
        print(f"  sum rule  sum_ij <S_i.S_j>: {results['sum_rule']:+.4f}   "
              "(0 for a singlet; NOT satisfied by the")
        print("                              synthetic data above -- this check is"
              " for real ED data)")
        print(f"  |C - C^T|_max             : {results['symmetry_violation']:.2e}")
        print(f"  on-site identity error    : {results['diag_identity_error']:.2e}")
    return results
# =============================================================================
# ED ORACLE

#Helper: Generate 2D real-space coordinates for Kagome clusters
from quspin.basis import spinful_fermion_basis_1d
from quspin.operators import hamiltonian
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import quspin as qp

def get_kagome_positions(nx: int, ny: int) -> np.ndarray:
    """
    Generates (x, y) coordinates for an nx by ny Kagome lattice.
    Each unit cell contains 3 sites (base=0,1; tip=2).
    Total sites generated = nx * ny * 3.
    """
    positions = []
    
    for y in range(ny):
        for x in range(nx):
            # Kagome Bravais lattice shift vectors.
        
            cell_origin = y * A2 + x * A1   # matches site_positions: r*A2 + c*A1
            
            positions.append(cell_origin + BASIS[0])  # site 0 (A)
            positions.append(cell_origin + BASIS[1])  # site 1 (B)
            positions.append(cell_origin + BASIS[2])  # site 2 (C)
            
    return np.array(positions)

def get_kagome_bonds(L) -> list:
    """Helper function to find all bonds of a specific distance using Minimum Image Convention."""
    Lx = L
    Ly = L
    target_dist = [1.0, np.sqrt(3.0)]
    
    positions = get_kagome_positions(Lx, Ly)
    #num_sites = len(positions)
    num_sites = 3 * Lx * Ly
    
    #a1 = np.array([2.0, 0.0])
    #a2 = np.array([1.0, np.sqrt(3)])
    
    L1 = Lx * A1
    L2 = Ly * A2
    
    nx_range = [-1, 0, 1] if Lx > 1 else [0]
    ny_range = [-1, 0, 1] if Ly > 1 else [0]
    
    t1_bonds = []
    t2_bonds = []
    
    for i in range(num_sites):
        for j in range(i + 1, num_sites): 
            delta = positions[j] - positions[i]

            min_dist = float('inf')
            best_nx = 0
            best_ny = 0
            
            for nx in nx_range:
                for ny in ny_range:
                    shift = nx * L1 + ny * L2
                    dist = np.linalg.norm(delta + shift)
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_nx = nx
                        best_ny = ny
                        
            t1_bonds.append((i, j))
                
    for i in range(num_sites):
        for j in range(i + 1, num_sites): 
            delta = positions[j] - positions[i]
            
            min_dist = float('inf')
            best_nx = 0
            best_ny = 0
            
            for nx in nx_range:
                for ny in ny_range:
                    shift = nx * L1 + ny * L2
                    dist = np.linalg.norm(delta + shift)
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_nx = nx
                        best_ny = ny
                        
            t2_bonds.append((i, j))
                
    return t1_bonds, t2_bonds
        
def build_fermi_hubbard_hamiltonian(L, U, filling, t):
    """
    Builds a sparse SciPy matrix for the Fermi-Hubbard model on a Kagome sub-cluster.
    Restricts the Hilbert space to exactly N_up and N_down electrons.
    """
    
    # 1. Restrict the Hilbert Space
    # This single line drops the matrix size from 4^N down to (N C N_up) * (N C N_down)
    Lx = L
    Ly = L
    N_elec = 3*(L**2) * filling
    t1 = t
    t2 = 0.0
    #U = U
    theta_x = 0.0
    theta_y = 0.0
    bc_x = 'periodic'
    bc_y = 'periodic'
    
    #rij, dist = pair_displacements(L)
    #N = dist.shape[0] #num sites
    
    N_up = int(N_elec // 2 + (N_elec % 2))
    N_down = int(N_elec // 2)
    
    sites = 3 * Lx * Ly
    basis = qp.basis.spinful_fermion_basis_1d(L=sites, Nf=(N_up, N_down))

    # 2. Get Lattice Bonds (You will need a helper function for your specific geometry)
    # These should be lists of tuples representing directional bonds: [(i, j), ...]
    # For Kagome, you only want to list each bond ONCE (e.g., i < j) because we use herm_con=True later.
    t1_bonds, t2_bonds = get_kagome_bonds(L)

    # 3. Format Interaction Lists for QuSpin
    # QuSpin uses operator strings where the left side of '|' is spin-up, right side is spin-down.
    
    # Hubbard U: n_{i,up} * n_{i,down}
    interaction_U = [[U, i, i] for i in range(sites)]

    # Hopping t: c^\dagger_i c_j 
    # '-' sign is included here because the physical hopping lowers the energy
    hop_up_t1 = [[-t1, i, j] for i, j in t1_bonds] + [[-t1, j, i] for i, j in t1_bonds]
    hop_dn_t1 = [[-t1, i, j] for i, j in t1_bonds] + [[-t1, j, i] for i, j in t1_bonds]
    
    hop_up_t2 = [[-t2, i, j] for i, j in t2_bonds] + [[-t2, j, i] for i, j in t2_bonds]
    hop_dn_t2 = [[-t2, i, j] for i, j in t2_bonds] + [[-t2, j, i] for i, j in t2_bonds]
    
    # Combine into QuSpin's static operator list
    static_operators = [
        ["n|n", interaction_U],     # U on both spins
        ["+-|", hop_up_t1],         # t1 hopping for spin-up
        ["|+-", hop_dn_t1],         # t1 hopping for spin-down
        ["+-|", hop_up_t2],         # t2 hopping for spin-up
        ["|+-", hop_dn_t2]          # t2 hopping for spin-down
    ]

    # 4. Build the Hamiltonian
    # FALSE herm_con=True automatically generates the Hermitian conjugates (the c^\dagger_j c_i terms), FALSE
    # meaning you don't have to manually write the reverse bonds.
    H = hamiltonian(
        static_operators, 
        dynamic_list=[], 
        basis=basis, 
        dtype=np.complex128, 
        check_herm=False,
        check_pcon=False, # Skips a slow internal symmetry check
        check_symm=False  
    )

    # 5. Extract and return as standard SciPy CSR sparse matrix
    return H.tocsr()

def solve_exact_diagonalization(L, U, filling, t):
    """
    Solves exact quantum properties for small Kagome sub-clusters using SciPy sparse ED.
    Expects L (lx = ly), U, electron filling, t
    """
    # Step A: Parse input JSON string from Agent
    #changed inputs to directly get sites, t2, and U
    

    Lx = L
    Ly = L
    N_elec = 3*(L**2) * filling
    t1 = t
    t2 = 0.0
    #U = U
    theta_x = 0.0
    theta_y = 0.0
    bc_x = 'periodic'
    bc_y = 'periodic'
    
    rij, dist = pair_displacements(L)
    N = dist.shape[0] #num sites
    
    N_up = int(N_elec // 2 + (N_elec % 2))
    N_down = int(N_elec // 2)
    
    sites = 3 * Lx * Ly

    #TABC
    avg_energy = 0.0
    energy_zero = 0.0
    psi_zero = None
    params_zero = None
    average_observables = None
    
    #twists = [(0.0,0.0), (0.0,np.pi), (np.pi, 0.0), (np.pi, np.pi)]
    twists = [(0.0, 0.0)]

    for theta_x, theta_y in twists:
        #params['theta_x'] = theta_x
        #params['theta_y'] = theta_y
        # Step B: Build your sparse Kagome Hamiltonian H
        # (Note: Replace this stub with your actual Hamiltonian builder)
        num_qubits = 2 * sites
        #H_sparse = sp.csr_matrix((2**num_qubits, 2**num_qubits), dtype=np.complex128)
        H_sparse = build_fermi_hubbard_hamiltonian(L, U, filling, t)
        basis = qp.basis.spinful_fermion_basis_1d(L=sites, Nf=(N_up, N_down))
    
        # Step C: Diagonalize to find Ground State (k=1 smallest algebraic eigenvalue)
        #k_request = min(4, H_sparse.shape[0] - 2)
        ground_eigenvalue, ground_eigenvector = spla.eigsh(
            H_sparse, k=1, which="SA"
            #H_sparse, k=3, which="SA"
            #H_sparse, k = k_request, which = 'SA'
        )
        
        # Flatten state vector for 1D array operations
        ground_state = ground_eigenvector[:, 0]
        e_ground = float(np.real(ground_eigenvalue[0]))
        #tolerance = 1e-8
        #degenerate_indices = np.where(np.abs(ground_eigenvalue - e_ground) < tolerance)[0]
        #degeneracy_count = len(degenerate_indices)
    
        # Step D: Get site geometry coordinates
        site_positions = get_kagome_positions(Lx, Ly)
    
        # Step E: Compute Observables via the helper function, NOTE: for all degeneracy states
        observables_n = compute_kagome_observables(
            ground_state = ground_state,
            num_sites = sites,
            site_positions = site_positions,
            basis = basis
        )

        if (theta_x, theta_y) == (0.0, 0.0):
            energy_zero = e_ground
            params_zero = {"L": L, "U": U, "t1": t1, "t2": t2}
            psi_zero = ground_state.copy()
            observables_zero = observables_n.copy()

        if average_observables is None:
            average_observables = {k: np.array(v)/float(len(twists)) for k, v in observables_n.items()}
        else:
            for k in average_observables:
                average_observables[k] += np.array(observables_n[k]) /float(len(twists))

                
        avg_energy += e_ground/float(len(twists))
        
    # Step F: Assemble and Return Full Results Dict as JSON String
    output_payload = {
        "psi": psi_zero,
        #"model_params": params_zero,
        "energy": energy_zero,
        "average_energy": avg_energy,
        "observables": observables_zero#,
        #"average_observables": average_observables
    }
    n = np.array(observables_zero["charge_density"])
    d = np.array(observables_zero["double_occupancy"])
    C = np.array(observables_zero["spin_corr"])
    output = dict(U=U, filling=filling, L=L, n=n, d=d, C=C)

    #return json.dumps(output_payload)
    return output

def compute_kagome_observables(ground_state: np.ndarray, num_sites: int, site_positions: np.ndarray, basis):
    """
    Computes Ground State Energy, Double Occupancy, Charge Correlation,
    Spin-Spin Correlation, and 2D Structure Factors for Kagome clusters
    within the constrained particle-conserving basis.
    
    Parameters:
        ground_state: Ground state eigenvector (dimension matches the constrained basis)
        num_sites: N (e.g., 3, 6, 9, 12)
        site_positions: Nx2 array of (x, y) coordinates for Kagome sites
        basis: The exact QuSpin basis object used to build the Hamiltonian
    """
    
    # 1 & 2. Build Sparse Number Operators n_i_up and n_i_down using QuSpin
    # This replaces the raw Pauli matrix approach and forces the operators 
    # to match the constrained dimension of the ground_state.
    n_up = []
    n_dn = []
    
    for i in range(num_sites):
        # Create constrained operators directly via QuSpin
        # "n|" means particle number operator for spin-up
        # "|n" means particle number operator for spin-down
        op_up = hamiltonian([["n|", [[1.0, i]]]], [], basis=basis, dtype=np.complex128, 
                            check_herm=False, check_symm=False, check_pcon=False)
        op_dn = hamiltonian([["|n", [[1.0, i]]]], [], basis=basis, dtype=np.complex128, 
                            check_herm=False, check_symm=False, check_pcon=False)
        
        # Convert to scipy sparse format so matrix math (@, +) works identically
        n_up.append(op_up.tocsr())
        n_dn.append(op_dn.tocsr())
        
    # 3. Double Occupancy Vector: <n_{i,up} * n_{i,dn}>
    double_occ = []
    for i in range(num_sites):
        D_op = n_up[i] @ n_dn[i]
        d_val = np.real(np.vdot(ground_state, D_op.dot(ground_state)))
        double_occ.append(float(d_val))

    #3.5 charge density:
    # Calculate expectation value: <psi | n | psi>
    #charge_density = []
    val_up = op_up.expt_value(ground_state).real
    val_dn = op_dn.expt_value(ground_state).real
        
    # Total charge at site i is the sum of up and down densities
    #charge_density.append(val_up + val_dn)
    charge_density = [
        np.real(np.vdot(ground_state, (n_up[i] + n_dn[i]).dot(ground_state)))
        for i in range(num_sites)
    ]

    # 4. Total Site Charge Operators N_i = n_{i,up} + n_{i,dn}
    N_op = [n_up[i] + n_dn[i] for i in range(num_sites)]
    
    # Charge Correlation Matrix (NxN)
    charge_corr = np.zeros((num_sites, num_sites))
    for i in range(num_sites):
        for j in range(num_sites):
            C_ij = N_op[i] @ N_op[j]
            charge_corr[i, j] = np.real(np.vdot(ground_state, C_ij.dot(ground_state)))

    # 5. Spin-Spin Correlation Matrix S_i . S_j = 3 * S^z_i S^z_j
    spin_corr = np.zeros((num_sites, num_sites))
    S_plus = []
    for i in range(num_sites):
        op = hamiltonian([["+|-", [[1.0, i, i]]]], [], basis=basis, dtype=np.complex128, check_herm=False, check_symm=False, check_pcon=False)
        S_plus.append(op.tocsr())
    S_minus = [Sp.conj().T for Sp in S_plus]

    for i in range(num_sites):
        S_z_i = 0.5 * (n_up[i] - n_dn[i])
        for j in range(num_sites):
            S_z_j = 0.5 * (n_up[j] - n_dn[j])
            S_ij = (S_z_i @ S_z_j) + 0.5 * (S_plus[i] @ S_minus[j] + S_minus[i] @ S_plus[j])
            spin_corr[i,j] = np.real(np.vdot(ground_state, S_ij.dot(ground_state)))
    '''
    for i in range(num_sites):
        S_z_i = 0.5 * (n_up[i] - n_dn[i])
        for j in range(num_sites):
            S_z_j = 0.5 * (n_up[j] - n_dn[j])
            S_ij = S_z_i @ S_z_j
            S_ij = S_z_i @ S_z_j + 0.5*(Splus_i @ Sminus_j + Sminus_i @ Splus_j)
            #S_ij = 3.0 * (S_z_i @ S_z_j)  # Isotropic multiplier for spin singlet
            spin_corr[i, j] = np.real(np.vdot(ground_state, S_ij.dot(ground_state)))
    '''

    # 6. 2D Spin Structure Factor S(q)
    # Example q-points in 2D Brillouin Zone
    momentum_density = 20
    qx = np.linspace(-np.pi, np.pi, momentum_density)
    qy = np.linspace(-np.pi, np.pi, momentum_density)
    QX, QY = np.meshgrid(qx, qy)
    
    # Flatten into a list of [qx, qy] coordinate pairs
    q_vectors = np.column_stack([QX.ravel(), QY.ravel()])

    # 6. 2D Spin Structure Factor S(q) over the dense grid
    spin_structure_factor = []
    for q in q_vectors:  # <--- Now iterating over the 400 grid points
        sq_val = 0.0
        for j in range(num_sites):
            for k in range(num_sites):
                r_jk = site_positions[j] - site_positions[k]
                phase = np.exp(-1j * np.dot(q, r_jk))
                sq_val += phase * spin_corr[j, k]
        spin_structure_factor.append(np.real(sq_val) / num_sites)
        
    # 7. 2D Charge Structure Factor S(q) over the dense grid
    charge_structure_factor = []
    for q in q_vectors:
        c_sq_val = 0.0
        for j in range(num_sites):
            for k in range(num_sites):
                r_jk = site_positions[j] - site_positions[k]
                phase = np.exp(-1j * np.dot(q, r_jk))
                c_sq_val += phase * charge_corr[j, k]
        charge_structure_factor.append(np.real(c_sq_val) / num_sites)
    return {
        "double_occupancy": double_occ,
        "charge_corr": charge_corr.tolist(),
        "charge_density": charge_density,
        "spin_corr": spin_corr.tolist(),
        "S_q_spin": spin_structure_factor,
        "S_q_charge": charge_structure_factor,
        "q_vectors": q_vectors
    }



# =============================================================================
# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    L = 2                                   # 2x2 unit cells = 12 sites (ED-friendly)
    #L = 1
    N = 3 * L * L

    print("=" * 72)
    print(f"KAGOME GEOMETRY CHECK   L={L}  ->  {N} sites")
    print("=" * 72)
    adj = build_adjacency(L)
    degs = sorted({len(v) for v in adj.values()})
    print(f"  degrees present            : {degs}      (kagome must be 4)")
    tri = sum(1 for i in adj for j in adj[i] for k in adj[j]
              if k in adj[i] and i < j < k)
    print(f"  triangles                  : {tri}      (expect 2N/3 = {2*N//3})")

    print()
    print("=" * 72)
    print("BASELINES  --  the CNN must beat the distance baseline")
    print("=" * 72)
    train = make_dataset(L, 50, rng)
    test = make_dataset(L, 50, rng)
    b0 = baseline_global_mean(train, test)
    b1 = baseline_mean_matrix(train, test)
    b2 = baseline_distance_U(train, test, L)
    print(f"  L0 global mean             : {b0:.6f}")
    print(f"  L1 mean matrix (no U)      : {b1:.6f}")
    print(f"  L2 distance + U            : {b2:.6f}   <-- BEAT THIS")
    print(f"     L2 explains {100*(1-b2/b0):.1f}% of the variance;")
    print(f"     the remainder is the angular / sublattice structure,")
    print(f"     which is precisely what a plain GNN cannot represent.")

    print()
    print("=" * 72)
    print("SANITY CHECKS on a synthetic sample")
    print("=" * 72)
    sanity_checks(train[0], L)

    if HAVE_TORCH:
        print()
        print("=" * 72)
        print("MODEL: size-agnosticism test (same weights, different L)")
        print("=" * 72)
        model = KagomeCNN()
        for Ltest in (2, 3, 4):
            Nt = 3 * Ltest * Ltest
            x = torch.randn(1, 2 * 3, Ltest, Ltest)
            rij, dist = pair_displacements(Ltest)
            out = model(x,
                        torch.tensor(rij, dtype=torch.float32)[None],
                        torch.tensor(dist, dtype=torch.float32)[None])
            print(f"  L={Ltest}  sites={Nt:3d}  ->  C shape {tuple(out['C'].shape)}  OK")
        n_par = sum(p.numel() for p in model.parameters())
        print(f"  parameter count is independent of L: {n_par:,}")
