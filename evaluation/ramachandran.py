import numpy as np
import matplotlib.pyplot as plt
from utils.sampling_utils import generate_specified_num_samples_parallel
from data.utils import unflatten_structure
import torch 
from utils.train_utils import prepare_batch

def unflatten_structure(flattened_structure):
    """
    Reshapes the structure back from (batchsize, num_residues*num_atoms, 3)
    to (batchsize, num_residues, num_atoms=4, 3).
    """
    batchsize = flattened_structure.shape[0]
    total_atoms_per_protein = flattened_structure.shape[1]
    assert total_atoms_per_protein % 4 == 0, "The total number of atoms should be divisible by 4."

    num_residues = total_atoms_per_protein // 4
    # Reshape to (batchsize, num_residues, num_atoms=4, 3)
    unflattened_structure = flattened_structure.reshape(batchsize, num_residues, 4, 3)
    return unflattened_structure

# Helper function to calculate dihedral angles
def calculate_dihedral(p):
    """Calculates dihedral angle for four consecutive atoms."""
    p0, p1, p2, p3 = p  # Unpack four points
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # Normalize b1 to avoid division by zero errors
    b1 /= np.linalg.norm(b1)

    # Compute vectors perpendicular to b1
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    # Return the dihedral angle
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))

def extract_phi_psi(samples):
    """Extracts phi and psi angles from the protein backbone atoms."""
    phi_angles = []
    psi_angles = []

    for protein in samples:
        # Extract backbone atoms: N, CA, C (Assuming standard atom ordering)
        N = protein[:, 0, :]   # Shape (num_residues, 3)
        CA = protein[:, 1, :]  # Shape (num_residues, 3)
        C = protein[:, 2, :]   # Shape (num_residues, 3)

        # Compute phi angles (N-CA-C-N)
        for i in range(1, len(N) - 1):
            phi = calculate_dihedral([N[i-1], CA[i-1], C[i-1], N[i]])
            phi_angles.append(phi)

        # Compute psi angles (CA-C-N-CA)
        for i in range(1, len(C) - 1):
            psi = calculate_dihedral([CA[i-1], C[i-1], N[i], CA[i]])
            psi_angles.append(psi)

    return np.array(phi_angles), np.array(psi_angles)

def plot_ramachandran(phi_real, psi_real, phi_fake, psi_fake):
    """Plots Ramachandran plots for real and generated protein samples."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Real data plot
    axs[0].scatter(phi_real, psi_real, s=1, color='blue')
    axs[0].set_title('Ramachandran Plot (Real Data)')
    axs[0].set_xlabel('Phi (ϕ)')
    axs[0].set_ylabel('Psi (ψ)')
    axs[0].set_xlim([-180, 180])
    axs[0].set_ylim([-180, 180])

    # Generated data plot
    axs[1].scatter(phi_fake, psi_fake, s=1, color='red')
    axs[1].set_title('Ramachandran Plot (Generated Data)')
    axs[1].set_xlabel('Phi (ϕ)')
    axs[1].set_xlim([-180, 180])
    axs[1].set_ylim([-180, 180])

    plt.tight_layout()
    plt.show()

def get_generated_samples(model, sde, num_samples, steps, shape, device_ids):
    generated_samples = generate_specified_num_samples_parallel(num_samples, sde, model, steps, shape, device_ids)
    return unflatten_structure(generated_samples)

def get_real_samples(dataloaders):
    real_data=[]
    for dataloader in dataloaders:
        for data in dataloader:
            real_data.append(unflatten_structure(data))
    real_data = torch.cat(real_data, dim=0)
    return real_data

# Main function to generate Ramachandran plots
def generate_ramachandran_plots(model, sde, num_samples, steps, shape, device_ids, dataloaders):
    real_samples = get_real_samples(dataloaders)
    generated_samples = get_generated_samples(model, sde, num_samples, steps, shape, device_ids)

    # Extract phi and psi angles for real and generated data
    phi_real, psi_real = extract_phi_psi(real_samples)
    phi_fake, psi_fake = extract_phi_psi(generated_samples)
    
    # Plot Ramachandran plots side by side
    plot_ramachandran(phi_real, psi_real, phi_fake, psi_fake)