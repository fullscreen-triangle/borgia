"""
Fixed Point Atlas Generator

Generates comprehensive fixed point atlas from:
1. Molecular encoder (SMARTS patterns)
2. Spectral oscillator data
3. Combined validation database

Implements Definition 13.1: Fixed Point Atlas structure.

Author: Trajectory Completion Framework
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List


class FixedPointAtlasGenerator:
    """Generates and manages fixed point atlases."""

    def __init__(self):
        """Initialize generator."""
        self.results_dir = Path(__file__).parent.parent.parent / 'results'
        self.results_dir.mkdir(exist_ok=True, parents=True)

    def load_molecular_data(self) -> Dict:
        """Load molecular encoder results."""
        mol_file = self.results_dir / 'fixed_point_uniqueness.json'

        if not mol_file.exists():
            return {'molecules': [], 'validation': {}}

        with open(mol_file, 'r') as f:
            return json.load(f)

    def load_spectral_data(self) -> List[Dict]:
        """Load oscillator mapping results."""
        spec_file = self.results_dir / 'oscillator_mapping_results.json'

        if not spec_file.exists():
            return []

        with open(spec_file, 'r') as f:
            return json.load(f)

    def create_atlas_entry(self, molecule: Dict, index: int) -> Dict:
        """
        Create atlas entry for a molecule.

        Args:
            molecule: Encoded molecule dictionary
            index: Index in atlas

        Returns:
            Atlas entry (Definition 13.1)
        """
        return {
            'atlas_id': f'FP_{index:04d}',
            'molecule_name': molecule.get('name', f'unknown_{index}'),
            'smarts': molecule.get('smarts', ''),
            'fixed_point': {
                'S_k': molecule['coordinates']['S_k'],
                'S_t': molecule['coordinates']['S_t'],
                'S_e': molecule['coordinates']['S_e']
            },
            'trit_string': molecule['trit_string'],
            'trit_depth': molecule['trit_string_length'],
            'penultimate_string': molecule['penultimate_string'],
            'gateway_trit': molecule['gateway_trit'],
            'partition_depth': molecule['partition_depth'],
            'molecular_properties': {
                'n_atoms': molecule['n_atoms'],
                'n_bonds': molecule['n_bonds'],
                'molecular_weight': molecule['molecular_weight']
            }
        }

    def create_spectral_entry(self, spectrum: Dict, index: int) -> Dict:
        """
        Create atlas entry for spectral data.

        Args:
            spectrum: Spectral data dictionary
            index: Index in atlas

        Returns:
            Atlas entry
        """
        return {
            'atlas_id': f'SP_{index:04d}',
            'data_source': spectrum['file'],
            'data_type': spectrum['type'],
            'fixed_point': {
                'S_k': spectrum['coordinates']['S_k'],
                'S_t': spectrum['coordinates']['S_t'],
                'S_e': spectrum['coordinates']['S_e']
            },
            'n_oscillators': spectrum['n_oscillators'],
            'categorical_resolution': spectrum['categorical_resolution'],
            'oscillator_summary': {
                'n_oscillators': spectrum['n_oscillators'],
                'mean_frequency': np.mean(spectrum['oscillators']['frequencies']),
                'max_frequency': np.max(spectrum['oscillators']['frequencies']),
                'min_frequency': np.min(spectrum['oscillators']['frequencies'])
            }
        }

    def generate_molecular_atlas(self) -> Dict:
        """
        Generate atlas from molecular data.

        Returns:
            Molecular fixed point atlas
        """
        mol_data = self.load_molecular_data()

        if not mol_data['molecules']:
            return {'error': 'no_molecular_data'}

        atlas_entries = []

        for i, molecule in enumerate(mol_data['molecules']):
            entry = self.create_atlas_entry(molecule, i)
            atlas_entries.append(entry)

        return {
            'atlas_type': 'molecular',
            'n_entries': len(atlas_entries),
            'entries': atlas_entries,
            'validation': mol_data.get('validation', {})
        }

    def generate_spectral_atlas(self) -> Dict:
        """
        Generate atlas from spectral data.

        Returns:
            Spectral fixed point atlas
        """
        spec_data = self.load_spectral_data()

        if not spec_data:
            return {'error': 'no_spectral_data'}

        atlas_entries = []

        for i, spectrum in enumerate(spec_data):
            entry = self.create_spectral_entry(spectrum, i)
            atlas_entries.append(entry)

        return {
            'atlas_type': 'spectral',
            'n_entries': len(atlas_entries),
            'entries': atlas_entries
        }

    def generate_combined_atlas(self) -> Dict:
        """
        Generate combined atlas from all sources.

        Returns:
            Combined fixed point atlas
        """
        mol_atlas = self.generate_molecular_atlas()
        spec_atlas = self.generate_spectral_atlas()

        all_entries = []

        if 'entries' in mol_atlas:
            all_entries.extend(mol_atlas['entries'])

        if 'entries' in spec_atlas:
            all_entries.extend(spec_atlas['entries'])

        return {
            'atlas_type': 'combined',
            'n_entries': len(all_entries),
            'n_molecular': len(mol_atlas.get('entries', [])),
            'n_spectral': len(spec_atlas.get('entries', [])),
            'entries': all_entries,
            'molecular_validation': mol_atlas.get('validation', {}),
            'metadata': {
                'generator': 'FixedPointAtlasGenerator',
                'framework': 'Trajectory Completion Cheminformatics'
            }
        }

    def atlas_statistics(self, atlas: Dict) -> Dict:
        """
        Compute atlas statistics.

        Args:
            atlas: Fixed point atlas

        Returns:
            Statistics dictionary
        """
        if 'entries' not in atlas or len(atlas['entries']) == 0:
            return {'error': 'empty_atlas'}

        coords = np.array([[e['fixed_point']['S_k'],
                           e['fixed_point']['S_t'],
                           e['fixed_point']['S_e']] for e in atlas['entries']])

        # Coverage in S-space
        coverage = {
            'S_k_range': [float(coords[:, 0].min()), float(coords[:, 0].max())],
            'S_t_range': [float(coords[:, 1].min()), float(coords[:, 1].max())],
            'S_e_range': [float(coords[:, 2].min()), float(coords[:, 2].max())],
            'S_k_mean': float(coords[:, 0].mean()),
            'S_t_mean': float(coords[:, 1].mean()),
            'S_e_mean': float(coords[:, 2].mean())
        }

        # Density analysis
        pairwise_dists = []
        n = len(coords)
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(coords[i] - coords[j])
                pairwise_dists.append(dist)

        pairwise_dists = np.array(pairwise_dists)

        density = {
            'min_distance': float(pairwise_dists.min()) if len(pairwise_dists) > 0 else 0,
            'mean_distance': float(pairwise_dists.mean()) if len(pairwise_dists) > 0 else 0,
            'max_distance': float(pairwise_dists.max()) if len(pairwise_dists) > 0 else 0,
            'std_distance': float(pairwise_dists.std()) if len(pairwise_dists) > 0 else 0
        }

        return {
            'n_entries': atlas['n_entries'],
            'coverage': coverage,
            'density': density
        }

    def export_to_csv(self, atlas: Dict, filename: str):
        """
        Export atlas to CSV format.

        Args:
            atlas: Fixed point atlas
            filename: Output filename
        """
        if 'entries' not in atlas:
            return

        rows = []

        for entry in atlas['entries']:
            row = {
                'atlas_id': entry['atlas_id'],
                'name': entry.get('molecule_name') or entry.get('data_source'),
                'type': entry.get('data_type', 'molecular'),
                'S_k': entry['fixed_point']['S_k'],
                'S_t': entry['fixed_point']['S_t'],
                'S_e': entry['fixed_point']['S_e']
            }

            if 'trit_depth' in entry:
                row['trit_depth'] = entry['trit_depth']
                row['gateway_trit'] = entry.get('gateway_trit')

            if 'n_oscillators' in entry:
                row['n_oscillators'] = entry['n_oscillators']

            rows.append(row)

        df = pd.DataFrame(rows)
        output_path = self.results_dir / filename
        df.to_csv(output_path, index=False)

        return output_path

    def generate_all(self) -> Dict:
        """Generate all atlas types and save."""
        print("\n=== Generating Fixed Point Atlases ===\n")

        # Generate atlases
        print("Generating molecular atlas...")
        mol_atlas = self.generate_molecular_atlas()

        if 'error' not in mol_atlas:
            print(f"  ✓ {mol_atlas['n_entries']} molecular entries")

        print("\nGenerating spectral atlas...")
        spec_atlas = self.generate_spectral_atlas()

        if 'error' not in spec_atlas:
            print(f"  ✓ {spec_atlas['n_entries']} spectral entries")

        print("\nGenerating combined atlas...")
        combined_atlas = self.generate_combined_atlas()

        print(f"  ✓ {combined_atlas['n_entries']} total entries")
        print(f"    - {combined_atlas['n_molecular']} molecular")
        print(f"    - {combined_atlas['n_spectral']} spectral")

        # Compute statistics
        print("\n=== Atlas Statistics ===")
        stats = self.atlas_statistics(combined_atlas)

        print(f"\nCoverage:")
        print(f"  S_k: [{stats['coverage']['S_k_range'][0]:.3f}, {stats['coverage']['S_k_range'][1]:.3f}]")
        print(f"  S_t: [{stats['coverage']['S_t_range'][0]:.3f}, {stats['coverage']['S_t_range'][1]:.3f}]")
        print(f"  S_e: [{stats['coverage']['S_e_range'][0]:.3f}, {stats['coverage']['S_e_range'][1]:.3f}]")

        print(f"\nDensity:")
        print(f"  Min distance: {stats['density']['min_distance']:.6f}")
        print(f"  Mean distance: {stats['density']['mean_distance']:.6f}")

        # Save atlases
        print("\n=== Saving Atlases ===")

        # JSON format
        json_file = self.results_dir / 'fixed_point_atlas.json'
        with open(json_file, 'w') as f:
            json.dump(combined_atlas, f, indent=2)
        print(f"✓ JSON: {json_file}")

        # CSV format
        csv_file = self.export_to_csv(combined_atlas, 'fixed_point_atlas.csv')
        print(f"✓ CSV: {csv_file}")

        # Statistics
        stats_file = self.results_dir / 'atlas_statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"✓ Stats: {stats_file}")

        return {
            'combined_atlas': combined_atlas,
            'statistics': stats
        }


def main():
    """Generate atlases."""
    generator = FixedPointAtlasGenerator()
    results = generator.generate_all()

    return results


if __name__ == '__main__':
    main()
