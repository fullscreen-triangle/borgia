"""
Fixed Point Uniqueness Module

Implements the molecular encoder from Algorithm 11.1.
Maps molecular structures to unique fixed points in S-space.
Validates Theorem 9.3: Distinct molecules → distinct fixed points.

Author: Trajectory Completion Framework
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from typing import Tuple, List, Dict
import json
from pathlib import Path
import pandas as pd


class MolecularEncoder:
    """Encodes molecular structures to fixed points in entropy coordinate space."""

    def __init__(self):
        """Initialize encoder with physical constants."""
        self.c = 299792458  # Speed of light (m/s)
        self.h = 6.62607015e-34  # Planck constant (J·s)
        self.kb = 1.380649e-23  # Boltzmann constant (J/K)
        self.Z_max = 118  # Maximum atomic number (Oganesson)
        self.M_max = 1000  # Maximum partition depth (normalized)

    def parse_smarts(self, smarts: str) -> Chem.Mol:
        """Parse SMARTS pattern to RDKit molecule."""
        mol = Chem.MolFromSmarts(smarts)
        if mol is None:
            # Try as SMILES
            mol = Chem.MolFromSmiles(smarts)
        return mol

    def compute_partition_depth_atomic(self, atom: Chem.Atom) -> float:
        """
        Compute partition depth M_i for an atom.

        M = Σ log₃(k_i) where k_i are partition counts.
        Approximate from quantum numbers (n, ℓ, m, s).

        Args:
            atom: RDKit Atom object

        Returns:
            Partition depth M_i
        """
        Z = atom.GetAtomicNum()

        if Z == 0:
            return 0.0

        # Estimate principal quantum number from atomic number
        # Rough approximation based on electron shells
        if Z <= 2:
            n = 1
        elif Z <= 10:
            n = 2
        elif Z <= 18:
            n = 3
        elif Z <= 36:
            n = 4
        elif Z <= 54:
            n = 5
        elif Z <= 86:
            n = 6
        else:
            n = 7

        # Partition depth contribution: M_i ≈ Σ log₃(2n²)
        # Each shell contributes
        M_i = sum(np.log(2 * k**2) / np.log(3) for k in range(1, n + 1))

        return M_i

    def compute_partition_depth_bond(self, bond: Chem.Bond) -> float:
        """
        Compute partition depth for a bond.

        From Partition Compression Theorem.

        Args:
            bond: RDKit Bond object

        Returns:
            Bond partition depth
        """
        bond_type = bond.GetBondType()

        # Bond order mapping
        if bond_type == Chem.BondType.SINGLE:
            order = 1.0
        elif bond_type == Chem.BondType.DOUBLE:
            order = 2.0
        elif bond_type == Chem.BondType.TRIPLE:
            order = 3.0
        elif bond_type == Chem.BondType.AROMATIC:
            order = 1.5
        else:
            order = 1.0

        # M_bond = log₃(order)
        M_bond = np.log(order + 1) / np.log(3)

        return M_bond

    def assign_trit_value_atom(self, atom: Chem.Atom, mol: Chem.Mol) -> int:
        """
        Assign trit value (0, 1, 2) based on primary axis contribution.

        0: Heavy atoms (high Z) → primarily constrain S_k
        1: High-frequency modes → primarily constrain S_t
        2: Deep partition structures → primarily constrain S_e

        Args:
            atom: RDKit Atom object
            mol: Parent molecule

        Returns:
            Trit value ∈ {0, 1, 2}
        """
        Z = atom.GetAtomicNum()
        degree = atom.GetDegree()
        M_i = self.compute_partition_depth_atomic(atom)

        # Decision based on properties
        if Z > 20:  # Heavy atoms
            return 0
        elif degree == 0 or atom.GetIsAromatic():  # High frequency modes
            return 1
        else:  # Deep partition structure
            return 2

    def assign_trit_value_bond(self, bond: Chem.Bond) -> int:
        """
        Assign trit value for bond.

        σ→0, π→1, δ→2

        Args:
            bond: RDKit Bond object

        Returns:
            Trit value ∈ {0, 1, 2}
        """
        bond_type = bond.GetBondType()

        if bond_type == Chem.BondType.SINGLE:
            return 0  # σ bond
        elif bond_type == Chem.BondType.DOUBLE:
            return 1  # π bond
        elif bond_type == Chem.BondType.TRIPLE:
            return 2  # δ component
        elif bond_type == Chem.BondType.AROMATIC:
            return 1  # π character
        else:
            return 0

    def construct_trit_string(self, mol: Chem.Mol) -> List[int]:
        """
        Construct trit string for molecule (Phase 3 of Algorithm 11.1).

        Args:
            mol: RDKit Mol object

        Returns:
            List of trit values (depth-ordered)
        """
        contributions = []

        # Atomic contributions
        for atom in mol.GetAtoms():
            M_i = self.compute_partition_depth_atomic(atom)
            trit = self.assign_trit_value_atom(atom, mol)
            contributions.append((M_i, trit, 'atom', atom.GetIdx()))

        # Bond contributions
        for bond in mol.GetBonds():
            M_bond = self.compute_partition_depth_bond(bond)
            trit = self.assign_trit_value_bond(bond)
            contributions.append((M_bond, trit, 'bond', bond.GetIdx()))

        # Sort by partition depth (descending)
        contributions.sort(key=lambda x: x[0], reverse=True)

        # Extract trit sequence
        trit_string = [c[1] for c in contributions]

        return trit_string

    def project_trit_string(self, trit_string: List[int], axis: int) -> np.ndarray:
        """
        Project trit string onto specific axis (Phase 4 of Algorithm 11.1).

        Args:
            trit_string: List of trit values
            axis: Axis to project onto (0, 1, or 2)

        Returns:
            Subsequence of trits matching axis
        """
        return np.array([t for t in trit_string if t == axis])

    def trit_sequence_to_coordinate(self, trit_seq: np.ndarray, total_length: int) -> float:
        """
        Convert trit sequence to [0,1] coordinate.

        Uses ternary encoding: value = Σ trit_i / 3^(i+1)

        Args:
            trit_seq: Sequence of trits for this axis
            total_length: Total length of trit string

        Returns:
            Coordinate value ∈ [0, 1]
        """
        if len(trit_seq) == 0:
            return 0.0

        # Proportion of trits on this axis
        proportion = len(trit_seq) / total_length

        # Encode sequence as ternary fraction
        value = 0.0
        for i, trit in enumerate(trit_seq[:10]):  # Limit to 10 trits for precision
            value += trit / (3 ** (i + 1))

        # Combine proportion and value
        coordinate = 0.5 * proportion + 0.5 * value

        return float(np.clip(coordinate, 0, 1))

    def encode_molecule(self, mol: Chem.Mol) -> Dict:
        """
        Encode molecule to fixed point s* in S-space (Full Algorithm 11.1).

        Args:
            mol: RDKit Mol object

        Returns:
            Dictionary with coordinates, trit string, and metadata
        """
        if mol is None:
            return None

        # Phase 1-3: Construct trit string
        trit_string = self.construct_trit_string(mol)

        # Phase 4: Extract coordinates
        k = len(trit_string)

        proj_0 = self.project_trit_string(trit_string, 0)
        proj_1 = self.project_trit_string(trit_string, 1)
        proj_2 = self.project_trit_string(trit_string, 2)

        S_k = self.trit_sequence_to_coordinate(proj_0, k)
        S_t = self.trit_sequence_to_coordinate(proj_1, k)
        S_e = self.trit_sequence_to_coordinate(proj_2, k)

        # Phase 5: Penultimate state
        if k > 0:
            trit_minus1 = trit_string[:-1]
            gateway = trit_string[-1]
        else:
            trit_minus1 = []
            gateway = None

        # Compute total partition depth
        M_total = sum(self.compute_partition_depth_atomic(atom) for atom in mol.GetAtoms())
        M_total += sum(self.compute_partition_depth_bond(bond) for bond in mol.GetBonds())

        return {
            'coordinates': {
                'S_k': S_k,
                'S_t': S_t,
                'S_e': S_e
            },
            'trit_string': trit_string,
            'trit_string_length': k,
            'penultimate_string': trit_minus1,
            'gateway_trit': gateway,
            'partition_depth': M_total / self.M_max,
            'n_atoms': mol.GetNumAtoms(),
            'n_bonds': mol.GetNumBonds(),
            'molecular_weight': Descriptors.MolWt(mol)
        }

    def compute_distance(self, s1: Tuple[float, float, float],
                        s2: Tuple[float, float, float]) -> float:
        """
        Compute Euclidean distance between two points in S-space.

        Args:
            s1: First coordinate (S_k, S_t, S_e)
            s2: Second coordinate (S_k, S_t, S_e)

        Returns:
            Euclidean distance
        """
        return np.sqrt(sum((a - b)**2 for a, b in zip(s1, s2)))

    def validate_uniqueness(self, encoded_molecules: List[Dict]) -> Dict:
        """
        Validate Theorem 9.3: distinct molecules have distinct fixed points.

        Args:
            encoded_molecules: List of encoding results

        Returns:
            Validation statistics
        """
        n_molecules = len(encoded_molecules)

        if n_molecules < 2:
            return {'validation': 'insufficient_data'}

        # Extract coordinates
        coords = np.array([[m['coordinates']['S_k'],
                           m['coordinates']['S_t'],
                           m['coordinates']['S_e']] for m in encoded_molecules])

        # Compute pairwise distances
        distances = []
        for i in range(n_molecules):
            for j in range(i + 1, n_molecules):
                dist = np.linalg.norm(coords[i] - coords[j])
                distances.append(dist)

        distances = np.array(distances)

        # Check uniqueness
        min_dist = np.min(distances)
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)

        # All pairs should be distinguishable (distance > 0)
        all_unique = min_dist > 1e-6

        return {
            'n_molecules': n_molecules,
            'all_unique': bool(all_unique),
            'min_distance': float(min_dist),
            'mean_distance': float(mean_dist),
            'std_distance': float(std_dist),
            'pairwise_distances': distances.tolist()
        }


def main():
    """Process SMARTS patterns and validate uniqueness."""
    encoder = MolecularEncoder()

    # Load SMARTS patterns
    smarts_file = Path(__file__).parent.parent.parent / 'public' / 'daylight-smarts-tar' / 'daylight.smarts'

    with open(smarts_file, 'r') as f:
        lines = f.readlines()

    # Parse SMARTS
    molecules_encoded = []

    for line in lines[1:]:  # Skip header
        if line.strip() and not line.startswith('#'):
            parts = line.strip().split()
            if len(parts) >= 2:
                smarts = parts[0]
                name = parts[1]

                mol = encoder.parse_smarts(smarts)

                if mol is not None:
                    encoded = encoder.encode_molecule(mol)
                    if encoded:
                        encoded['smarts'] = smarts
                        encoded['name'] = name
                        molecules_encoded.append(encoded)

                        print(f"{name}: S = ({encoded['coordinates']['S_k']:.4f}, "
                              f"{encoded['coordinates']['S_t']:.4f}, "
                              f"{encoded['coordinates']['S_e']:.4f}), "
                              f"depth = {encoded['trit_string_length']}")

    print(f"\n✓ Encoded {len(molecules_encoded)} molecules")

    # Validate uniqueness
    validation = encoder.validate_uniqueness(molecules_encoded)
    print(f"\n=== Uniqueness Validation ===")
    print(f"All unique: {validation['all_unique']}")
    print(f"Min distance: {validation['min_distance']:.6f}")
    print(f"Mean distance: {validation['mean_distance']:.6f}")

    # Save results
    results = {
        'molecules': molecules_encoded,
        'validation': validation
    }

    output_file = Path(__file__).parent.parent.parent / 'results' / 'fixed_point_uniqueness.json'
    output_file.parent.mkdir(exist_ok=True, parents=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")

    # Also save as CSV for easy viewing
    df = pd.DataFrame([{
        'name': m['name'],
        'smarts': m['smarts'],
        'S_k': m['coordinates']['S_k'],
        'S_t': m['coordinates']['S_t'],
        'S_e': m['coordinates']['S_e'],
        'depth': m['trit_string_length'],
        'n_atoms': m['n_atoms']
    } for m in molecules_encoded])

    csv_file = output_file.with_suffix('.csv')
    df.to_csv(csv_file, index=False)
    print(f"✓ CSV saved to {csv_file}")

    return results


if __name__ == '__main__':
    main()
