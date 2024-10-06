import os
import plotly.graph_objects as go

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


import plotly.graph_objects as go

def simply_visualise_protein(protein):
    """
    Visualizes the 3D structure of a protein as blue points using Plotly.
    
    Parameters:
    - protein: A numpy array of shape (num_residues, num_atoms=4, 3) 
               containing the 3D coordinates of the backbone atoms of all residues.
    """
    # Ensure the input protein has the correct shape
    assert protein.shape[1] == 4 and protein.shape[2] == 3, "Protein structure must be (num_residues, 4, 3)"
    
    # Create a figure for visualization
    fig = go.Figure()
    
    # Add all atoms as blue points
    for i, residue in enumerate(protein):
        for atom_pos in residue:
            fig.add_trace(go.Scatter3d(
                x=[atom_pos[0]], y=[atom_pos[1]], z=[atom_pos[2]],
                mode='markers',
                marker=dict(size=6, color='blue', symbol='circle'),
                showlegend=False
            ))
    
    # Configure plot appearance
    fig.update_layout(
        title="3D Protein Structure Visualization",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            bgcolor='white'
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        width=800,
        height=800
    )
    
    return fig


def visualise_protein(protein):
    """
    Visualizes the 3D structure of a protein backbone using Plotly.
    
    Parameters:
    - protein: A numpy array of shape (num_residues, num_atoms=4, 3) 
               containing the 3D coordinates of the backbone atoms of all residues.
               The atom order in each residue is ['N', 'CA', 'C', 'O'].
    """
    # Ensure the input protein has the correct shape
    assert protein.shape[1] == 4 and protein.shape[2] == 3, "Protein structure must be (num_residues, 4, 3)"
    
    # Colors for each atom type
    atom_colors = {
        'N': 'blue',
        'CA': 'green',
        'C': 'gray',
        'O': 'red'
    }
    
    # Colors for bonds
    bond_colors = {
        'single': 'orange',  # N-Cα, Cα-C, C-N (peptide bond)
        'double': 'purple'   # C=O (carbonyl double bond)
    }
    
    # Create a figure for visualization
    fig = go.Figure()
    
    # Add atoms to the plot
    atom_labels = ['N', 'CA', 'C', 'O']  # Fixed order of atoms in each residue
    for i, residue in enumerate(protein):
        for j, atom_pos in enumerate(residue):
            atom_type = atom_labels[j]
            fig.add_trace(go.Scatter3d(
                x=[atom_pos[0]], y=[atom_pos[1]], z=[atom_pos[2]],
                mode='markers',
                marker=dict(size=6, color=atom_colors[atom_type], symbol='circle'),
                name=f"{atom_type} (Residue {i + 1})",
                showlegend=False
            ))
    
    # Add bonds between backbone atoms
    for i in range(protein.shape[0] - 1):
        # Within the same residue: N -> Cα -> C
        for (start_idx, end_idx, bond_type) in [(0, 1, 'single'), (1, 2, 'single'), (2, 3, 'double')]:
            start_atom = protein[i, start_idx]
            end_atom = protein[i, end_idx]
            fig.add_trace(go.Scatter3d(
                x=[start_atom[0], end_atom[0]],
                y=[start_atom[1], end_atom[1]],
                z=[start_atom[2], end_atom[2]],
                mode='lines',
                line=dict(color=bond_colors[bond_type], width=4),
                showlegend=False
            ))
        
        # Between consecutive residues: C (i) -> N (i+1) (peptide bond)
        start_atom = protein[i, 2]  # C of residue i
        end_atom = protein[i + 1, 0]  # N of residue i+1
        fig.add_trace(go.Scatter3d(
            x=[start_atom[0], end_atom[0]],
            y=[start_atom[1], end_atom[1]],
            z=[start_atom[2], end_atom[2]],
            mode='lines',
            line=dict(color=bond_colors['single'], width=4),
            showlegend=False
        ))
    
    # Configure plot appearance
    fig.update_layout(
        title="Interactive 3D Protein Backbone Visualization",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            bgcolor='white'
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        width=800,
        height=800
    )
    
    return fig

def visualise_raw_protein(protein_flattened):
    """
    Visualizes the 3D structure of a raw unflattened protein structure as connected points.
    This function connects points linearly from index 0 to the last point.
    
    Parameters:
    - protein: A numpy array of shape (num_residues, num_atoms=4, 3)
               containing the 3D coordinates of the backbone atoms.
    """
    # Flatten the protein into a continuous list of 3D points
    #protein_flattened = protein.reshape(-1, 3)  # Shape: (num_residues*num_atoms, 3)
    
    # Create a 3D scatter plot with connected lines
    fig = go.Figure(go.Scatter3d(
        x=protein_flattened[:, 0],  # X coordinates
        y=protein_flattened[:, 1],  # Y coordinates
        z=protein_flattened[:, 2],  # Z coordinates
        mode='lines+markers',  # Connect points with lines and show markers
        line=dict(color='red', width=2),  # Blue lines
        marker=dict(size=5, color='blue')  # Blue markers
    ))
    
    # Configure plot appearance
    fig.update_layout(
        title="Raw 3D Protein Structure Visualization",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            bgcolor='white'
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        width=800,
        height=800
    )
    
    return fig

def visualise_and_save_proteins(samples, eval_dir):
    """
    Visualizes and saves the 3D structure of protein samples as interactive HTML files.
    
    Parameters:
    - samples: A tensor of shape (batch_size, num_residues, num_atoms=4, 3), 
               containing multiple protein samples.
    - eval_dir: Directory where the results should be saved.
    """
    # Create the save directory if it doesn't exist
    save_dir = os.path.join(eval_dir, 'samples')
    os.makedirs(save_dir, exist_ok=True)

    samples = unflatten_structure(samples)

    # Iterate over the samples and create an interactive plot for each
    for idx in range(samples.size(0)):
        protein = samples[idx].cpu().numpy()  # Convert the current sample to a numpy array
        
        # Ensure that the protein has the correct shape
        #print(protein.shape)
        #assert protein.shape == (protein.shape[0], 4, 3), f"Sample {idx} has an incorrect shape."
        
        # Generate the interactive 3D plot using the visualise_protein function
        fig = visualise_protein(protein)
        
        # Save the plot as an HTML file with the sample index
        html_filename = os.path.join(save_dir, f'protein_sample_{idx + 1}.html')
        fig.write_html(html_filename)
        print(f"Saved: {html_filename}")
