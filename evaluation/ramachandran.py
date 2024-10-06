import numpy as np
import matplotlib.pyplot as plt
from utils.sampling_utils import generate_specified_num_samples_parallel
from data.utils import unflatten_structure
import torch 
from utils.train_utils import prepare_batch
from tqdm import tqdm
import os
import seaborn as sns

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
def calculate_dihedral_torch(p0, p1, p2, p3):
    """Calculates dihedral angles for batches of four consecutive atoms."""
    # Vectorized calculations using torch
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # Normalize b1
    b1_norm = torch.norm(b1, dim=-1, keepdim=True)
    b1 = b1 / b1_norm

    # Compute vectors perpendicular to b1
    v = b0 - (torch.sum(b0 * b1, dim=-1, keepdim=True)) * b1
    w = b2 - (torch.sum(b2 * b1, dim=-1, keepdim=True)) * b1

    # Compute dihedral angle
    x = torch.sum(v * w, dim=-1)
    y = torch.sum(torch.cross(b1, v, dim=-1) * w, dim=-1)
    
    return torch.atan2(y, x) * (180.0 / np.pi)  # Convert to degrees


def extract_phi_psi(samples):
    """Extracts phi and psi angles from the protein backbone atoms with torch-based batch computation."""
    phi_angles = []
    psi_angles = []

    # Use tqdm to track the progress
    for protein in tqdm(samples, desc="Extracting phi and psi angles"):
        # Extract backbone atoms: N, CA, C (Assuming standard atom ordering)
        N = torch.tensor(protein[:, 0, :], dtype=torch.float32)   # Shape (num_residues, 3)
        CA = torch.tensor(protein[:, 1, :], dtype=torch.float32)  # Shape (num_residues, 3)
        C = torch.tensor(protein[:, 2, :], dtype=torch.float32)   # Shape (num_residues, 3)

        # Compute phi angles (N-CA-C-N)
        phi = calculate_dihedral_torch(N[:-2], CA[:-2], C[:-2], N[1:-1])
        phi_angles.append(phi)

        # Compute psi angles (CA-C-N-CA)
        psi = calculate_dihedral_torch(CA[:-2], C[:-2], N[1:-1], CA[1:-1])
        psi_angles.append(psi)

    return torch.cat(phi_angles).cpu().numpy(), torch.cat(psi_angles).cpu().numpy()

def plot_ramachandran(phi_real, psi_real, phi_fake, psi_fake, save_dir):
    """Plots Ramachandran plots for real and generated protein samples and saves them."""
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

    # Save the figure with excellent quality (dpi=300)
    save_path = os.path.join(save_dir, 'ramachandran_plot.png')
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_ramachandran_kde(phi_real, psi_real, phi_fake, psi_fake, save_dir):
    """Plots Ramachandran plots with Kernel Density Estimation for real and generated protein samples."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Real data KDE plot
    sns.kdeplot(x=phi_real, y=psi_real, cmap="Blues", fill=True, thresh=0.05, ax=axs[0])
    axs[0].set_title('Ramachandran Plot KDE (Real Data)')
    axs[0].set_xlabel('Phi (ϕ)')
    axs[0].set_ylabel('Psi (ψ)')
    axs[0].set_xlim([-180, 180])
    axs[0].set_ylim([-180, 180])

    # Generated data KDE plot
    sns.kdeplot(x=phi_fake, y=psi_fake, cmap="Reds", fill=True, thresh=0.05, ax=axs[1])
    axs[1].set_title('Ramachandran Plot KDE (Generated Data)')
    axs[1].set_xlabel('Phi (ϕ)')
    axs[1].set_xlim([-180, 180])
    axs[1].set_ylim([-180, 180])

    plt.tight_layout()

    # Save the figure with excellent quality (dpi=300)
    save_path = os.path.join(save_dir, 'ramachandran_kde_plot.png')
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

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
def generate_ramachandran_plots(model, sde, num_samples, steps, shape, device_ids, dataloaders, eval_dir):
    real_samples = get_real_samples(dataloaders)[:64]
    #generated_samples = real_samples  # Uncomment this when you want to generate samples
    generated_samples = get_generated_samples(model, sde, num_samples, steps, shape, device_ids)

    # Extract phi and psi angles for real and generated data
    phi_real, psi_real = extract_phi_psi(real_samples)
    phi_fake, psi_fake = extract_phi_psi(generated_samples)
    
    # Create the directory for saving the plots
    save_dir = os.path.join(eval_dir, 'ramachandran_plots')
    os.makedirs(save_dir, exist_ok=True)

    # Plot Ramachandran plots and save
    plot_ramachandran(phi_real, psi_real, phi_fake, psi_fake, save_dir)
    plot_ramachandran_kde(phi_real, psi_real, phi_fake, psi_fake, save_dir)