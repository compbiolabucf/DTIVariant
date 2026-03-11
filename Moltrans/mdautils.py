three_to_one = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
}
import MDAnalysis as mda
import os
import hashlib
def generate_hashid(input_string):
    # Create a hash object using SHA-256
    hash_object = hashlib.sha256(input_string.encode())
    # Get the hexadecimal digest of the hash
    hex_digest = hash_object.hexdigest()
    # Return the first 4 characters of the hash
    return hex_digest[:10]
def get_residues(path, cutoff_distance = 5.0, data="bind"):
    # u = mda.Universe(f"def_clos_fur_complex_{data}/{hash}/{hash}_complex.pdb")
    # print(path)
    # print(os.path.isfile(path))
    u = mda.Universe(path)
    ligand = u.select_atoms("resname UNL")
    binding_site_residues = u.select_atoms(f"protein and around {cutoff_distance} group ligand", ligand=ligand)
    residues = {(res.resname, res.resid) for res in binding_site_residues.residues}
    residues = sorted(residues, key=lambda x:x[1])
    ress = "".join([three_to_one[x[0]] for x in residues])
    indices = [y for x,y in residues]
    return ress, indices