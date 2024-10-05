def flatten_structure(structure):
    """
    Reshapes the structure from (num_residues, num_atoms=4, 3) to (num_residues*num_atoms, 3).
    """
    num_residues = structure.shape[0]
    num_atoms = structure.shape[1]
    assert num_atoms == 4, "Expected num_atoms=4, got {}".format(num_atoms)
    
    # Reshape to (num_residues * num_atoms, 3)
    flattened_structure = structure.reshape(num_residues * num_atoms, 3)
    return flattened_structure

def unflatten_structure(flattened_structure):
    """
    Reshapes the structure back from (num_residues*num_atoms, 3) to (num_residues, num_atoms=4, 3).
    """
    total_atoms = flattened_structure.shape[0]
    assert total_atoms % 4 == 0, "The total number of atoms should be divisible by 4."
    
    num_residues = total_atoms // 4
    unflattened_structure = flattened_structure.reshape(num_residues, 4, 3)
    return unflattened_structure