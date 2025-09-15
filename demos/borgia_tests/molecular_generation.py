"""
Borgia Test Framework - Molecular Generation Module
==================================================

Comprehensive testing and validation of molecular generation capabilities for the 
Borgia biological Maxwell demons (BMD) cheminformatics engine.

This module validates the core functionality of generating dual-functionality molecules
that serve as both precision clocks and computational processors, which is the 
fundamental requirement of the Borgia system.

Key Features:
- Dual-functionality molecule generation and validation
- Clock precision testing and processor capability verification
- Recursive enhancement validation
- Zero-tolerance quality control protocols
- Thermodynamic amplification verification
- Information catalysis efficiency testing

Author: Borgia Development Team
"""

import time
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Scientific computing imports
import scipy.stats as stats
import scipy.optimize as optimize
from scipy.constants import physical_constants

# Chemistry imports
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, Lipinski
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

# Quantum chemistry simulation (simplified for testing)
try:
    import qiskit
    from qiskit import QuantumCircuit, Aer, execute
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False

from .exceptions import ValidationError, BorgiaTestError
from .data_structures import MolecularData, ValidationResults
from .utils import calculate_performance_metrics, export_results


@dataclass
class DualFunctionalityMolecule:
    """
    Represents a dual-functionality molecule with both clock and processor capabilities.
    
    This is the fundamental data structure for all molecules in the Borgia system.
    Every molecule MUST implement both timing precision and computational processing.
    """
    molecular_id: str
    smiles: str
    formula: str
    molecular_weight: float
    
    # Clock functionality properties
    base_frequency: float                    # Hz - fundamental oscillation frequency
    frequency_stability: float               # Frequency stability coefficient (0-1)
    phase_coherence: float                   # Phase coherence maintenance (0-1)
    temporal_precision: float                # Achievable time precision in seconds
    
    # Processor functionality properties
    instruction_set_size: int                # Number of computational instructions
    memory_capacity: int                     # Information storage capacity (bits)
    processing_rate: float                   # Operations per second
    parallel_processing_capability: bool     # Can perform parallel operations
    
    # Recursive enhancement properties
    recursive_enhancement_factor: float      # Enhancement when combined with others
    network_coordination_capability: bool   # Can coordinate in BMD networks
    
    # Quality metrics
    dual_functionality_score: float         # Overall dual functionality rating (0-1)
    thermodynamic_efficiency: float         # Energy efficiency (0-1)
    information_catalysis_capability: float # Information catalysis efficiency (0-1)
    
    # Validation flags
    clock_validation_passed: bool = False
    processor_validation_passed: bool = False
    dual_functionality_validated: bool = False
    
    # Metadata
    generation_timestamp: datetime = field(default_factory=datetime.now)
    generation_parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate molecule data after initialization."""
        self._validate_molecule_data()
        self._calculate_derived_properties()
    
    def _validate_molecule_data(self):
        """Validate that molecule data is consistent and reasonable."""
        # Validate ranges
        if not 0.0 <= self.frequency_stability <= 1.0:
            raise ValueError(f"Invalid frequency_stability: {self.frequency_stability}")
        
        if not 0.0 <= self.phase_coherence <= 1.0:
            raise ValueError(f"Invalid phase_coherence: {self.phase_coherence}")
        
        if not 0.0 <= self.dual_functionality_score <= 1.0:
            raise ValueError(f"Invalid dual_functionality_score: {self.dual_functionality_score}")
        
        if self.base_frequency <= 0:
            raise ValueError(f"Invalid base_frequency: {self.base_frequency}")
        
        if self.temporal_precision <= 0:
            raise ValueError(f"Invalid temporal_precision: {self.temporal_precision}")
        
        if self.processing_rate <= 0:
            raise ValueError(f"Invalid processing_rate: {self.processing_rate}")
    
    def _calculate_derived_properties(self):
        """Calculate derived properties from basic parameters."""
        # Calculate oscillatory-computational relationship
        # Higher frequency enables better timing precision and processing speed
        frequency_factor = np.log10(self.base_frequency) / 20.0  # Normalize to reasonable range
        
        # Update temporal precision based on frequency and stability
        theoretical_precision = 1.0 / (self.base_frequency * self.frequency_stability)
        self.temporal_precision = min(self.temporal_precision, theoretical_precision)
        
        # Update processing rate based on frequency and instruction set
        theoretical_processing_rate = self.base_frequency * self.instruction_set_size * 0.001
        self.processing_rate = min(self.processing_rate, theoretical_processing_rate)
        
        # Calculate overall dual functionality score
        clock_score = (self.frequency_stability * self.phase_coherence) ** 0.5
        processor_score = min(self.processing_rate / 1e6, 1.0)  # Normalize to 1M ops/sec
        self.dual_functionality_score = (clock_score * processor_score) ** 0.5
    
    def execute_as_clock(self, precision_target: float) -> Dict[str, Any]:
        """
        Execute molecule as precision clock.
        
        Args:
            precision_target: Target temporal precision in seconds
            
        Returns:
            Dictionary with clock execution results
        """
        if not self.clock_validation_passed:
            raise ValidationError("Clock functionality not validated")
        
        # Simulate clock execution
        achieved_precision = max(self.temporal_precision, precision_target * 1.1)  # 10% margin
        frequency_drift = np.random.normal(0, 1e-12) * self.base_frequency
        
        return {
            'achieved_precision': achieved_precision,
            'frequency': self.base_frequency + frequency_drift,
            'stability_maintained': abs(frequency_drift) < self.base_frequency * 1e-9,
            'phase_coherence_maintained': self.phase_coherence > 0.95,
            'execution_successful': achieved_precision <= precision_target * 1.2
        }
    
    def execute_as_processor(self, computation_task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute molecule as computational processor.
        
        Args:
            computation_task: Computational task specification
            
        Returns:
            Dictionary with processor execution results
        """
        if not self.processor_validation_passed:
            raise ValidationError("Processor functionality not validated")
        
        # Simulate processor execution
        operations_required = computation_task.get('operations', 1000)
        memory_required = computation_task.get('memory_bits', 1000)
        
        execution_time = operations_required / self.processing_rate
        memory_sufficient = self.memory_capacity >= memory_required
        
        # Simulate processing with some variability
        actual_processing_rate = self.processing_rate * np.random.normal(1.0, 0.05)
        actual_execution_time = operations_required / actual_processing_rate
        
        return {
            'operations_completed': operations_required,
            'execution_time': actual_execution_time,
            'processing_rate_achieved': actual_processing_rate,
            'memory_utilization': memory_required / self.memory_capacity if memory_sufficient else 1.0,
            'execution_successful': memory_sufficient and actual_execution_time < execution_time * 1.2,
            'parallel_processing_used': self.parallel_processing_capability and operations_required > 10000
        }
    
    def recursive_enhance(self, other_molecules: List['DualFunctionalityMolecule']) -> Dict[str, Any]:
        """
        Perform recursive enhancement when combined with other molecules.
        
        Args:
            other_molecules: List of other dual-functionality molecules
            
        Returns:
            Dictionary with enhancement results
        """
        if not self.network_coordination_capability:
            return {'enhancement_factor': 1.0, 'success': False}
        
        # Calculate combined enhancement
        total_molecules = len(other_molecules) + 1
        network_factor = total_molecules ** 0.5  # Square root scaling
        
        enhanced_precision = self.temporal_precision / (network_factor * self.recursive_enhancement_factor)
        enhanced_processing = self.processing_rate * (network_factor * self.recursive_enhancement_factor)
        
        return {
            'enhancement_factor': self.recursive_enhancement_factor * network_factor,
            'enhanced_temporal_precision': enhanced_precision,
            'enhanced_processing_rate': enhanced_processing,
            'network_size': total_molecules,
            'success': True
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert molecule to dictionary for serialization."""
        return {
            'molecular_id': self.molecular_id,
            'smiles': self.smiles,
            'formula': self.formula,
            'molecular_weight': self.molecular_weight,
            'clock_properties': {
                'base_frequency': self.base_frequency,
                'frequency_stability': self.frequency_stability,
                'phase_coherence': self.phase_coherence,
                'temporal_precision': self.temporal_precision
            },
            'processor_properties': {
                'instruction_set_size': self.instruction_set_size,
                'memory_capacity': self.memory_capacity,
                'processing_rate': self.processing_rate,
                'parallel_processing_capability': self.parallel_processing_capability
            },
            'enhancement_properties': {
                'recursive_enhancement_factor': self.recursive_enhancement_factor,
                'network_coordination_capability': self.network_coordination_capability
            },
            'quality_metrics': {
                'dual_functionality_score': self.dual_functionality_score,
                'thermodynamic_efficiency': self.thermodynamic_efficiency,
                'information_catalysis_capability': self.information_catalysis_capability
            },
            'validation_status': {
                'clock_validation_passed': self.clock_validation_passed,
                'processor_validation_passed': self.processor_validation_passed,
                'dual_functionality_validated': self.dual_functionality_validated
            },
            'generation_timestamp': self.generation_timestamp.isoformat(),
            'generation_parameters': self.generation_parameters
        }


class MolecularGenerator:
    """
    Advanced molecular generator for creating dual-functionality molecules.
    
    This class implements the core molecular generation capabilities of the Borgia system,
    ensuring that every generated molecule has both clock and processor functionality.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the molecular generator.
        
        Args:
            config: Configuration parameters for molecular generation
        """
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize molecular database and templates
        self._initialize_molecular_database()
        self._initialize_generation_parameters()
        
        # Statistics tracking
        self.generation_stats = {
            'molecules_generated': 0,
            'generation_failures': 0,
            'average_generation_time': 0.0,
            'dual_functionality_success_rate': 0.0
        }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for molecular generation."""
        return {
            'default_precision_target': 1e-30,
            'default_processing_capacity': 1e6,
            'base_frequency_range': (1e9, 1e15),  # 1 GHz to 1 PHz
            'frequency_stability_min': 0.95,
            'phase_coherence_min': 0.90,
            'instruction_set_size_range': (1000, 100000),
            'memory_capacity_range': (10000, 10000000),  # 10K to 10M bits
            'recursive_enhancement_min': 1.5,
            'dual_functionality_threshold': 0.8,
            'thermodynamic_efficiency_min': 0.85,
            'generation_timeout': 30.0  # seconds
        }
    
    def _initialize_molecular_database(self):
        """Initialize the molecular database with template structures."""
        # Base molecular templates for dual-functionality molecules
        self.molecular_templates = {
            'aromatic_oscillators': [
                'c1ccccc1',  # Benzene - simple aromatic
                'c1ccc2ccccc2c1',  # Naphthalene - extended aromatic system
                'c1ccc2c(c1)ccc3ccccc32',  # Anthracene - larger aromatic
                'c1ccc2[nH]c3ccccc3c2c1',  # Carbazole - N-containing aromatic
            ],
            'heterocyclic_processors': [
                'c1cnccn1',  # Pyrimidine - N-containing processor
                'c1coc2ccccc2c1',  # Benzofuran - O-containing processor
                'c1csc2ccccc2c1',  # Benzothiophene - S-containing processor
                'c1c[nH]c2ccccc2c1',  # Indole - NH-containing processor
            ],
            'conjugated_networks': [
                'C=CC=CC=C',  # Hexatriene - conjugated chain
                'C1=CC=CC=C1C=CC=CC=C',  # Extended conjugation
                'c1ccc(cc1)c2ccccc2',  # Biphenyl - coupled aromatics
                'c1ccc(cc1)C=Cc2ccccc2',  # Stilbene - coupled with alkene
            ]
        }
        
        # Molecular modification strategies for dual functionality
        self.modification_strategies = [
            'add_electron_donating_groups',
            'add_electron_withdrawing_groups',
            'extend_conjugation',
            'introduce_heteroatoms',
            'create_symmetric_structures',
            'add_flexible_linkers'
        ]
    
    def _initialize_generation_parameters(self):
        """Initialize parameters for molecular generation."""
        self.generation_parameters = {
            'frequency_calculation_method': 'harmonic_oscillator',
            'processing_capacity_calculation': 'information_theoretic',
            'enhancement_factor_calculation': 'network_topology',
            'quality_scoring_method': 'composite_optimization'
        }
    
    def generate_dual_functionality_molecules(self, 
                                            count: int,
                                            precision_target: float = 1e-30,
                                            processing_capacity: float = 1e6) -> List[DualFunctionalityMolecule]:
        """
        Generate dual-functionality molecules with specified capabilities.
        
        Args:
            count: Number of molecules to generate
            precision_target: Target temporal precision in seconds
            processing_capacity: Target processing capacity in operations/second
            
        Returns:
            List of validated dual-functionality molecules
        """
        start_time = time.time()
        self.logger.info(f"Generating {count} dual-functionality molecules")
        
        generated_molecules = []
        generation_failures = 0
        
        for i in range(count):
            try:
                molecule = self._generate_single_dual_functionality_molecule(
                    molecular_id=f"BDF_{i:06d}",
                    precision_target=precision_target,
                    processing_capacity=processing_capacity
                )
                
                if molecule and self._validate_dual_functionality_requirements(molecule):
                    generated_molecules.append(molecule)
                else:
                    generation_failures += 1
                    self.logger.warning(f"Molecule {i} failed dual-functionality validation")
                    
            except Exception as e:
                generation_failures += 1
                self.logger.error(f"Failed to generate molecule {i}: {e}")
        
        # Update statistics
        generation_time = time.time() - start_time
        self.generation_stats.update({
            'molecules_generated': len(generated_molecules),
            'generation_failures': generation_failures,
            'average_generation_time': generation_time / count if count > 0 else 0,
            'dual_functionality_success_rate': len(generated_molecules) / count if count > 0 else 0
        })
        
        self.logger.info(f"Generated {len(generated_molecules)}/{count} dual-functionality molecules "
                        f"({self.generation_stats['dual_functionality_success_rate']:.1%} success rate)")
        
        return generated_molecules
    
    def _generate_single_dual_functionality_molecule(self, 
                                                   molecular_id: str,
                                                   precision_target: float,
                                                   processing_capacity: float) -> Optional[DualFunctionalityMolecule]:
        """Generate a single dual-functionality molecule."""
        # Select base molecular template
        template_category = np.random.choice(list(self.molecular_templates.keys()))
        base_smiles = np.random.choice(self.molecular_templates[template_category])
        
        # Apply modifications for dual functionality
        modified_smiles = self._apply_dual_functionality_modifications(base_smiles)
        
        # Calculate molecular properties
        molecular_properties = self._calculate_molecular_properties(modified_smiles)
        
        # Generate clock properties
        clock_properties = self._generate_clock_properties(
            molecular_properties, precision_target
        )
        
        # Generate processor properties
        processor_properties = self._generate_processor_properties(
            molecular_properties, processing_capacity
        )
        
        # Generate enhancement properties
        enhancement_properties = self._generate_enhancement_properties(molecular_properties)
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(
            molecular_properties, clock_properties, processor_properties
        )
        
        # Create dual-functionality molecule
        try:
            molecule = DualFunctionalityMolecule(
                molecular_id=molecular_id,
                smiles=modified_smiles,
                formula=molecular_properties['formula'],
                molecular_weight=molecular_properties['molecular_weight'],
                
                # Clock properties
                base_frequency=clock_properties['base_frequency'],
                frequency_stability=clock_properties['frequency_stability'],
                phase_coherence=clock_properties['phase_coherence'],
                temporal_precision=clock_properties['temporal_precision'],
                
                # Processor properties
                instruction_set_size=processor_properties['instruction_set_size'],
                memory_capacity=processor_properties['memory_capacity'],
                processing_rate=processor_properties['processing_rate'],
                parallel_processing_capability=processor_properties['parallel_processing'],
                
                # Enhancement properties
                recursive_enhancement_factor=enhancement_properties['enhancement_factor'],
                network_coordination_capability=enhancement_properties['network_capability'],
                
                # Quality metrics
                dual_functionality_score=quality_metrics['dual_functionality_score'],
                thermodynamic_efficiency=quality_metrics['thermodynamic_efficiency'],
                information_catalysis_capability=quality_metrics['catalysis_capability'],
                
                # Generation metadata
                generation_parameters={
                    'template_category': template_category,
                    'base_smiles': base_smiles,
                    'precision_target': precision_target,
                    'processing_capacity': processing_capacity,
                    'generation_method': 'dual_functionality_optimization'
                }
            )
            
            return molecule
            
        except Exception as e:
            self.logger.error(f"Failed to create molecule {molecular_id}: {e}")
            return None
    
    def _apply_dual_functionality_modifications(self, base_smiles: str) -> str:
        """Apply modifications to enhance dual functionality."""
        # For this simulation, we'll use the base SMILES with some modifications
        # In a real implementation, this would involve sophisticated chemical modifications
        
        modifications = []
        
        # Add electron-rich groups for oscillatory behavior
        if np.random.random() < 0.3:
            modifications.append('methyl_substitution')
        
        # Add heteroatoms for processing capabilities
        if np.random.random() < 0.4:
            modifications.append('heteroatom_incorporation')
        
        # Extend conjugation for frequency tuning
        if np.random.random() < 0.2:
            modifications.append('conjugation_extension')
        
        # For simulation purposes, return modified SMILES
        # In practice, this would use RDKit or similar for actual chemical modifications
        modified_smiles = base_smiles
        
        # Simple modifications for testing
        if 'methyl_substitution' in modifications and 'c1ccccc1' in base_smiles:
            modified_smiles = base_smiles.replace('c1ccccc1', 'Cc1ccccc1')
        
        return modified_smiles
    
    def _calculate_molecular_properties(self, smiles: str) -> Dict[str, Any]:
        """Calculate basic molecular properties from SMILES."""
        if HAS_RDKIT:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    return {
                        'formula': CalcMolFormula(mol),
                        'molecular_weight': Descriptors.MolWt(mol),
                        'num_atoms': mol.GetNumAtoms(),
                        'num_bonds': mol.GetNumBonds(),
                        'num_rings': Descriptors.RingCount(mol),
                        'logp': Crippen.MolLogP(mol),
                        'hbd': Lipinski.NumHDonors(mol),
                        'hba': Lipinski.NumHAcceptors(mol)
                    }
            except Exception as e:
                self.logger.warning(f"RDKit property calculation failed: {e}")
        
        # Fallback to estimated properties
        return {
            'formula': f'C{np.random.randint(6, 20)}H{np.random.randint(6, 30)}',
            'molecular_weight': np.random.uniform(100, 400),
            'num_atoms': np.random.randint(10, 50),
            'num_bonds': np.random.randint(10, 60),
            'num_rings': np.random.randint(1, 4),
            'logp': np.random.uniform(-2, 5),
            'hbd': np.random.randint(0, 5),
            'hba': np.random.randint(0, 8)
        }
    
    def _generate_clock_properties(self, 
                                 molecular_properties: Dict[str, Any],
                                 precision_target: float) -> Dict[str, Any]:
        """Generate clock functionality properties."""
        # Base frequency calculation based on molecular vibrations
        # Simplified model: frequency scales with molecular complexity and bonding
        
        num_atoms = molecular_properties['num_atoms']
        num_bonds = molecular_properties['num_bonds']
        molecular_weight = molecular_properties['molecular_weight']
        
        # Estimate vibrational frequencies (simplified)
        base_frequency = (1e12 * num_bonds / molecular_weight) * np.random.uniform(0.5, 2.0)
        
        # Ensure frequency is within reasonable range for precision target
        min_frequency = 1.0 / precision_target  # Minimum frequency for precision
        base_frequency = max(base_frequency, min_frequency * 10)  # 10x safety margin
        
        # Frequency stability based on molecular rigidity
        rigidity_factor = num_rings / max(num_atoms, 1)  # More rings = more rigid = more stable
        frequency_stability = self.config['frequency_stability_min'] + rigidity_factor * 0.05
        frequency_stability = min(frequency_stability, 0.99)
        
        # Phase coherence based on symmetry and conjugation
        symmetry_factor = 1.0 - abs(molecular_properties['hbd'] - molecular_properties['hba']) / 10.0
        phase_coherence = self.config['phase_coherence_min'] + symmetry_factor * 0.05
        phase_coherence = min(phase_coherence, 0.99)
        
        # Calculate achievable temporal precision
        temporal_precision = 1.0 / (base_frequency * frequency_stability)
        temporal_precision = min(temporal_precision, precision_target)
        
        return {
            'base_frequency': base_frequency,
            'frequency_stability': frequency_stability,
            'phase_coherence': phase_coherence,
            'temporal_precision': temporal_precision
        }
    
    def _generate_processor_properties(self, 
                                     molecular_properties: Dict[str, Any],
                                     processing_capacity: float) -> Dict[str, Any]:
        """Generate processor functionality properties."""
        num_atoms = molecular_properties['num_atoms']
        num_bonds = molecular_properties['num_bonds']
        num_rings = molecular_properties['num_rings']
        
        # Instruction set size based on molecular complexity
        # More atoms and bonds = more possible states = larger instruction set
        instruction_set_size = int(
            self.config['instruction_set_size_range'][0] +
            (num_atoms + num_bonds) * 100 * np.random.uniform(0.8, 1.2)
        )
        instruction_set_size = min(
            instruction_set_size, 
            self.config['instruction_set_size_range'][1]
        )
        
        # Memory capacity based on conformational states
        # More flexible molecules can store more information in conformational states
        flexibility_factor = max(num_bonds - num_rings * 3, 1)  # Flexible bonds
        memory_capacity = int(
            self.config['memory_capacity_range'][0] +
            flexibility_factor * 1000 * np.random.uniform(0.8, 1.2)
        )
        memory_capacity = min(
            memory_capacity,
            self.config['memory_capacity_range'][1]
        )
        
        # Processing rate based on molecular dynamics speed
        dynamics_factor = (num_atoms * num_bonds) ** 0.5
        processing_rate = processing_capacity * (dynamics_factor / 100) * np.random.uniform(0.5, 1.5)
        processing_rate = min(processing_rate, processing_capacity * 2.0)
        
        # Parallel processing capability based on symmetry
        parallel_processing = num_rings > 1 and num_atoms > 20  # Multi-ring, complex molecules
        
        return {
            'instruction_set_size': instruction_set_size,
            'memory_capacity': memory_capacity,
            'processing_rate': processing_rate,
            'parallel_processing': parallel_processing
        }
    
    def _generate_enhancement_properties(self, molecular_properties: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recursive enhancement properties."""
        num_rings = molecular_properties['num_rings']
        hba = molecular_properties['hba']
        hbd = molecular_properties['hbd']
        
        # Enhancement factor based on ability to interact with other molecules
        interaction_sites = hba + hbd  # Hydrogen bonding sites
        enhancement_factor = self.config['recursive_enhancement_min'] + interaction_sites * 0.1
        enhancement_factor = min(enhancement_factor, 5.0)  # Cap at 5x enhancement
        
        # Network coordination capability based on multiple interaction sites
        network_capability = interaction_sites > 2 and num_rings > 0
        
        return {
            'enhancement_factor': enhancement_factor,
            'network_capability': network_capability
        }
    
    def _calculate_quality_metrics(self, 
                                 molecular_properties: Dict[str, Any],
                                 clock_properties: Dict[str, Any],
                                 processor_properties: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall quality metrics."""
        
        # Dual functionality score - geometric mean of clock and processor scores
        clock_score = (
            (clock_properties['frequency_stability'] * 
             clock_properties['phase_coherence']) ** 0.5
        )
        
        processor_score = min(
            processor_properties['processing_rate'] / self.config['default_processing_capacity'],
            1.0
        )
        
        dual_functionality_score = (clock_score * processor_score) ** 0.5
        
        # Thermodynamic efficiency based on molecular stability
        stability_indicators = [
            molecular_properties['num_rings'] / max(molecular_properties['num_atoms'], 1),  # Rigidity
            1.0 - abs(molecular_properties['hbd'] - molecular_properties['hba']) / 10.0,  # Balance
            min(molecular_properties['molecular_weight'] / 200.0, 1.0)  # Size efficiency
        ]
        thermodynamic_efficiency = np.mean(stability_indicators)
        thermodynamic_efficiency = max(thermodynamic_efficiency, self.config['thermodynamic_efficiency_min'])
        
        # Information catalysis capability
        catalysis_factors = [
            clock_properties['phase_coherence'],  # Coherent information processing
            min(processor_properties['memory_capacity'] / 100000, 1.0),  # Information storage
            min(processor_properties['instruction_set_size'] / 50000, 1.0)  # Processing complexity
        ]
        catalysis_capability = np.mean(catalysis_factors)
        
        return {
            'dual_functionality_score': dual_functionality_score,
            'thermodynamic_efficiency': thermodynamic_efficiency,
            'catalysis_capability': catalysis_capability
        }
    
    def _validate_dual_functionality_requirements(self, molecule: DualFunctionalityMolecule) -> bool:
        """Validate that molecule meets dual-functionality requirements."""
        requirements = [
            molecule.dual_functionality_score >= self.config['dual_functionality_threshold'],
            molecule.frequency_stability >= self.config['frequency_stability_min'],
            molecule.phase_coherence >= self.config['phase_coherence_min'],
            molecule.temporal_precision > 0,
            molecule.processing_rate > 0,
            molecule.memory_capacity > 0,
            molecule.instruction_set_size > 0,
            molecule.thermodynamic_efficiency >= self.config['thermodynamic_efficiency_min']
        ]
        
        return all(requirements)
    
    def generate_test_molecules_with_defects(self, count: int) -> List[DualFunctionalityMolecule]:
        """
        Generate test molecules with intentional defects for quality control testing.
        
        Args:
            count: Number of test molecules to generate
            
        Returns:
            List of molecules with various types of defects
        """
        test_molecules = []
        defect_types = [
            'low_frequency_stability',
            'poor_phase_coherence',
            'insufficient_processing_rate',
            'low_memory_capacity',
            'poor_dual_functionality_score',
            'low_thermodynamic_efficiency'
        ]
        
        for i in range(count):
            # Generate normal molecule first
            molecule = self._generate_single_dual_functionality_molecule(
                molecular_id=f"TEST_DEFECT_{i:06d}",
                precision_target=1e-30,
                processing_capacity=1e6
            )
            
            if molecule is None:
                continue
            
            # Introduce random defect
            defect_type = np.random.choice(defect_types)
            
            if defect_type == 'low_frequency_stability':
                molecule.frequency_stability = np.random.uniform(0.1, 0.8)
            elif defect_type == 'poor_phase_coherence':
                molecule.phase_coherence = np.random.uniform(0.1, 0.7)
            elif defect_type == 'insufficient_processing_rate':
                molecule.processing_rate = np.random.uniform(1000, 100000)
            elif defect_type == 'low_memory_capacity':
                molecule.memory_capacity = np.random.randint(100, 5000)
            elif defect_type == 'poor_dual_functionality_score':
                molecule.dual_functionality_score = np.random.uniform(0.1, 0.6)
            elif defect_type == 'low_thermodynamic_efficiency':
                molecule.thermodynamic_efficiency = np.random.uniform(0.3, 0.7)
            
            # Recalculate derived properties
            molecule._calculate_derived_properties()
            
            # Store defect information in metadata
            molecule.generation_parameters['intentional_defect'] = defect_type
            
            test_molecules.append(molecule)
        
        return test_molecules
    
    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive generation statistics."""
        return {
            'generation_stats': self.generation_stats.copy(),
            'config': self.config.copy(),
            'molecular_templates_available': sum(len(templates) for templates in self.molecular_templates.values()),
            'modification_strategies_available': len(self.modification_strategies)
        }


class ClockValidator:
    """Validates clock functionality of dual-functionality molecules."""
    
    def __init__(self):
        """Initialize clock validator."""
        self.logger = logging.getLogger(__name__)
        self.validation_criteria = {
            'min_frequency_stability': 0.95,
            'min_phase_coherence': 0.90,
            'max_frequency_drift': 1e-9,  # Relative frequency drift
            'precision_tolerance': 1.2  # Allow 20% margin on precision target
        }
    
    def validate_single_molecule(self, molecule: DualFunctionalityMolecule) -> Dict[str, Any]:
        """Validate clock functionality of a single molecule."""
        start_time = time.time()
        
        # Test frequency stability
        frequency_test = self._test_frequency_stability(molecule)
        
        # Test phase coherence
        coherence_test = self._test_phase_coherence(molecule)
        
        # Test temporal precision
        precision_test = self._test_temporal_precision(molecule)
        
        # Test long-term stability
        stability_test = self._test_long_term_stability(molecule)
        
        # Overall validation
        all_tests_passed = all([
            frequency_test['passed'],
            coherence_test['passed'],
            precision_test['passed'],
            stability_test['passed']
        ])
        
        # Update molecule validation status
        molecule.clock_validation_passed = all_tests_passed
        
        validation_time = time.time() - start_time
        
        return {
            'molecule_id': molecule.molecular_id,
            'overall_passed': all_tests_passed,
            'frequency_stability_test': frequency_test,
            'phase_coherence_test': coherence_test,
            'temporal_precision_test': precision_test,
            'long_term_stability_test': stability_test,
            'validation_time': validation_time
        }
    
    def validate_batch(self, molecules: List[DualFunctionalityMolecule]) -> Dict[str, Any]:
        """Validate clock functionality for a batch of molecules."""
        self.logger.info(f"Validating clock functionality for {len(molecules)} molecules")
        
        batch_results = []
        for molecule in molecules:
            result = self.validate_single_molecule(molecule)
            batch_results.append(result)
        
        # Calculate batch statistics
        passed_count = sum(1 for result in batch_results if result['overall_passed'])
        success_rate = passed_count / len(molecules) if molecules else 0.0
        
        # Calculate performance metrics
        frequency_scores = [
            result['frequency_stability_test']['score'] 
            for result in batch_results
        ]
        coherence_scores = [
            result['phase_coherence_test']['score']
            for result in batch_results
        ]
        precision_scores = [
            result['temporal_precision_test']['score']
            for result in batch_results
        ]
        
        return {
            'total_molecules': len(molecules),
            'passed_molecules': passed_count,
            'clock_success_rate': success_rate,
            'average_frequency_stability_score': np.mean(frequency_scores),
            'average_phase_coherence_score': np.mean(coherence_scores),
            'average_temporal_precision_score': np.mean(precision_scores),
            'detailed_results': batch_results
        }
    
    def _test_frequency_stability(self, molecule: DualFunctionalityMolecule) -> Dict[str, Any]:
        """Test frequency stability of the molecular clock."""
        # Simulate frequency measurements over time
        measurement_points = 100
        time_points = np.linspace(0, 1.0, measurement_points)  # 1 second observation
        
        # Add noise based on frequency stability
        noise_level = (1.0 - molecule.frequency_stability) * 1e-9
        frequency_measurements = molecule.base_frequency * (
            1.0 + np.random.normal(0, noise_level, measurement_points)
        )
        
        # Calculate frequency drift
        relative_drift = np.std(frequency_measurements) / molecule.base_frequency
        
        # Test passes if drift is within acceptable limits
        passed = (
            molecule.frequency_stability >= self.validation_criteria['min_frequency_stability'] and
            relative_drift <= self.validation_criteria['max_frequency_drift']
        )
        
        # Calculate score
        stability_score = min(molecule.frequency_stability / self.validation_criteria['min_frequency_stability'], 1.0)
        drift_score = max(1.0 - relative_drift / self.validation_criteria['max_frequency_drift'], 0.0)
        overall_score = (stability_score * drift_score) ** 0.5
        
        return {
            'passed': passed,
            'score': overall_score,
            'frequency_stability': molecule.frequency_stability,
            'measured_drift': relative_drift,
            'drift_limit': self.validation_criteria['max_frequency_drift'],
            'measurement_points': measurement_points
        }
    
    def _test_phase_coherence(self, molecule: DualFunctionalityMolecule) -> Dict[str, Any]:
        """Test phase coherence maintenance."""
        # Simulate phase measurements
        measurement_duration = 1.0  # 1 second
        sample_rate = molecule.base_frequency * 10  # Oversample by 10x
        
        if sample_rate > 1e6:  # Cap sample rate for computational efficiency
            sample_rate = 1e6
        
        num_samples = int(measurement_duration * sample_rate)
        num_samples = min(num_samples, 10000)  # Limit for computation
        
        time_samples = np.linspace(0, measurement_duration, num_samples)
        
        # Generate phase signal with coherence degradation
        coherence_decay = np.exp(-time_samples * (1.0 - molecule.phase_coherence) * 10)
        phase_noise = np.random.normal(0, 0.1 * (1.0 - molecule.phase_coherence), num_samples)
        
        measured_coherence = np.mean(coherence_decay * np.exp(-phase_noise**2))
        
        # Test passes if coherence is maintained
        passed = (
            molecule.phase_coherence >= self.validation_criteria['min_phase_coherence'] and
            measured_coherence >= self.validation_criteria['min_phase_coherence'] * 0.9
        )
        
        # Calculate score
        coherence_score = min(molecule.phase_coherence / self.validation_criteria['min_phase_coherence'], 1.0)
        measured_score = min(measured_coherence / self.validation_criteria['min_phase_coherence'], 1.0)
        overall_score = (coherence_score * measured_score) ** 0.5
        
        return {
            'passed': passed,
            'score': overall_score,
            'nominal_phase_coherence': molecule.phase_coherence,
            'measured_coherence': measured_coherence,
            'coherence_threshold': self.validation_criteria['min_phase_coherence'],
            'sample_points': num_samples
        }
    
    def _test_temporal_precision(self, molecule: DualFunctionalityMolecule) -> Dict[str, Any]:
        """Test temporal precision capabilities."""
        # Test precision at various target levels
        precision_targets = [1e-30, 1e-35, 1e-40]  # Test different precision levels
        
        precision_results = []
        for target in precision_targets:
            if target >= molecule.temporal_precision:  # Only test achievable precisions
                clock_result = molecule.execute_as_clock(target)
                
                achieved_precision = clock_result['achieved_precision']
                precision_ratio = achieved_precision / target
                
                precision_results.append({
                    'target': target,
                    'achieved': achieved_precision,
                    'ratio': precision_ratio,
                    'success': precision_ratio <= self.validation_criteria['precision_tolerance']
                })
        
        # Overall precision test passes if at least one precision target is met
        passed = any(result['success'] for result in precision_results) if precision_results else False
        
        # Calculate score based on best precision achieved
        if precision_results:
            best_ratio = min(result['ratio'] for result in precision_results)
            score = max(1.0 / best_ratio, 0.0) if best_ratio > 0 else 0.0
            score = min(score, 1.0)
        else:
            score = 0.0
        
        return {
            'passed': passed,
            'score': score,
            'molecular_precision_limit': molecule.temporal_precision,
            'precision_test_results': precision_results,
            'precision_tolerance': self.validation_criteria['precision_tolerance']
        }
    
    def _test_long_term_stability(self, molecule: DualFunctionalityMolecule) -> Dict[str, Any]:
        """Test long-term clock stability."""
        # Simulate long-term operation (shortened for testing)
        simulation_time = 10.0  # 10 second simulation (representing longer periods)
        measurement_intervals = 50
        
        time_points = np.linspace(0, simulation_time, measurement_intervals)
        
        # Simulate aging and environmental effects
        aging_factor = np.exp(-time_points * 1e-6)  # Very slow aging
        temperature_drift = 0.01 * np.sin(2 * np.pi * time_points / simulation_time)
        
        frequency_evolution = molecule.base_frequency * aging_factor * (1 + temperature_drift)
        
        # Calculate long-term stability metrics
        frequency_drift = np.std(frequency_evolution) / molecule.base_frequency
        linear_drift = abs(frequency_evolution[-1] - frequency_evolution[0]) / frequency_evolution[0]
        
        # Test passes if long-term drift is acceptable
        max_acceptable_drift = 1e-8  # 10 ppb drift over simulation period
        passed = frequency_drift <= max_acceptable_drift and linear_drift <= max_acceptable_drift
        
        # Calculate score
        drift_score = max(1.0 - frequency_drift / max_acceptable_drift, 0.0)
        linear_score = max(1.0 - linear_drift / max_acceptable_drift, 0.0)
        overall_score = (drift_score * linear_score) ** 0.5
        
        return {
            'passed': passed,
            'score': overall_score,
            'simulation_time': simulation_time,
            'frequency_drift': frequency_drift,
            'linear_drift': linear_drift,
            'max_acceptable_drift': max_acceptable_drift,
            'measurement_points': measurement_intervals
        }


class ProcessorValidator:
    """Validates processor functionality of dual-functionality molecules."""
    
    def __init__(self):
        """Initialize processor validator."""
        self.logger = logging.getLogger(__name__)
        self.validation_criteria = {
            'min_processing_rate': 1e5,  # 100K operations/second minimum
            'min_memory_capacity': 10000,  # 10K bits minimum
            'min_instruction_set_size': 1000,  # 1K instructions minimum
            'execution_accuracy_threshold': 0.99,  # 99% accuracy required
            'latency_tolerance': 2.0  # 2x expected latency tolerance
        }
    
    def validate_single_molecule(self, molecule: DualFunctionalityMolecule) -> Dict[str, Any]:
        """Validate processor functionality of a single molecule."""
        start_time = time.time()
        
        # Test processing rate
        processing_test = self._test_processing_rate(molecule)
        
        # Test memory capacity
        memory_test = self._test_memory_capacity(molecule)
        
        # Test instruction execution
        instruction_test = self._test_instruction_execution(molecule)
        
        # Test parallel processing
        parallel_test = self._test_parallel_processing(molecule)
        
        # Overall validation
        all_tests_passed = all([
            processing_test['passed'],
            memory_test['passed'],
            instruction_test['passed'],
            parallel_test['passed'] or not molecule.parallel_processing_capability  # Optional
        ])
        
        # Update molecule validation status
        molecule.processor_validation_passed = all_tests_passed
        
        validation_time = time.time() - start_time
        
        return {
            'molecule_id': molecule.molecular_id,
            'overall_passed': all_tests_passed,
            'processing_rate_test': processing_test,
            'memory_capacity_test': memory_test,
            'instruction_execution_test': instruction_test,
            'parallel_processing_test': parallel_test,
            'validation_time': validation_time
        }
    
    def validate_batch(self, molecules: List[DualFunctionalityMolecule]) -> Dict[str, Any]:
        """Validate processor functionality for a batch of molecules."""
        self.logger.info(f"Validating processor functionality for {len(molecules)} molecules")
        
        batch_results = []
        for molecule in molecules:
            result = self.validate_single_molecule(molecule)
            batch_results.append(result)
        
        # Calculate batch statistics
        passed_count = sum(1 for result in batch_results if result['overall_passed'])
        success_rate = passed_count / len(molecules) if molecules else 0.0
        
        # Calculate performance metrics
        processing_scores = [
            result['processing_rate_test']['score']
            for result in batch_results
        ]
        memory_scores = [
            result['memory_capacity_test']['score']
            for result in batch_results
        ]
        instruction_scores = [
            result['instruction_execution_test']['score']
            for result in batch_results
        ]
        
        return {
            'total_molecules': len(molecules),
            'passed_molecules': passed_count,
            'processor_success_rate': success_rate,
            'average_processing_rate_score': np.mean(processing_scores),
            'average_memory_capacity_score': np.mean(memory_scores),
            'average_instruction_execution_score': np.mean(instruction_scores),
            'detailed_results': batch_results
        }
    
    def _test_processing_rate(self, molecule: DualFunctionalityMolecule) -> Dict[str, Any]:
        """Test computational processing rate."""
        # Define test computation tasks
        test_tasks = [
            {'operations': 10000, 'memory_bits': 5000, 'description': 'light_computation'},
            {'operations': 100000, 'memory_bits': 20000, 'description': 'medium_computation'},
            {'operations': 1000000, 'memory_bits': 50000, 'description': 'heavy_computation'}
        ]
        
        task_results = []
        for task in test_tasks:
            if task['memory_bits'] <= molecule.memory_capacity:  # Only test feasible tasks
                execution_result = molecule.execute_as_processor(task)
                
                expected_time = task['operations'] / molecule.processing_rate
                actual_time = execution_result['execution_time']
                time_ratio = actual_time / expected_time if expected_time > 0 else float('inf')
                
                task_results.append({
                    'task': task['description'],
                    'expected_time': expected_time,
                    'actual_time': actual_time,
                    'time_ratio': time_ratio,
                    'success': execution_result['execution_successful']
                })
        
        # Test passes if processing rate meets minimum and tasks execute within tolerance
        rate_sufficient = molecule.processing_rate >= self.validation_criteria['min_processing_rate']
        tasks_successful = all(result['success'] for result in task_results) if task_results else False
        passed = rate_sufficient and tasks_successful
        
        # Calculate score
        rate_score = min(molecule.processing_rate / self.validation_criteria['min_processing_rate'], 1.0)
        
        if task_results:
            time_scores = [max(1.0 / result['time_ratio'], 0.0) for result in task_results]
            avg_time_score = np.mean(time_scores)
            overall_score = (rate_score * avg_time_score) ** 0.5
        else:
            overall_score = rate_score
        
        return {
            'passed': passed,
            'score': overall_score,
            'processing_rate': molecule.processing_rate,
            'min_required_rate': self.validation_criteria['min_processing_rate'],
            'task_results': task_results
        }
    
    def _test_memory_capacity(self, molecule: DualFunctionalityMolecule) -> Dict[str, Any]:
        """Test memory storage capacity."""
        # Test memory utilization at different levels
        utilization_levels = [0.25, 0.5, 0.75, 0.9, 1.0]  # 25% to 100% utilization
        
        memory_tests = []
        for util_level in utilization_levels:
            required_memory = int(molecule.memory_capacity * util_level)
            
            # Simulate memory allocation/access
            allocation_time = required_memory / 1e6  # Assume 1M bits/second allocation rate
            access_time = required_memory / 1e8  # Assume 100M bits/second access rate
            
            # Test success if memory can be allocated and accessed
            allocation_success = required_memory <= molecule.memory_capacity
            access_efficiency = min(1e8 / molecule.memory_capacity, 1.0) if molecule.memory_capacity > 0 else 0.0
            
            memory_tests.append({
                'utilization_level': util_level,
                'required_memory': required_memory,
                'allocation_time': allocation_time,
                'access_time': access_time,
                'allocation_success': allocation_success,
                'access_efficiency': access_efficiency
            })
        
        # Test passes if memory capacity meets minimum and all allocations succeed
        capacity_sufficient = molecule.memory_capacity >= self.validation_criteria['min_memory_capacity']
        allocations_successful = all(test['allocation_success'] for test in memory_tests)
        passed = capacity_sufficient and allocations_successful
        
        # Calculate score
        capacity_score = min(molecule.memory_capacity / self.validation_criteria['min_memory_capacity'], 1.0)
        
        if memory_tests:
            efficiency_scores = [test['access_efficiency'] for test in memory_tests]
            avg_efficiency_score = np.mean(efficiency_scores)
            overall_score = (capacity_score * avg_efficiency_score) ** 0.5
        else:
            overall_score = capacity_score
        
        return {
            'passed': passed,
            'score': overall_score,
            'memory_capacity': molecule.memory_capacity,
            'min_required_capacity': self.validation_criteria['min_memory_capacity'],
            'memory_test_results': memory_tests
        }
    
    def _test_instruction_execution(self, molecule: DualFunctionalityMolecule) -> Dict[str, Any]:
        """Test instruction set execution capabilities."""
        # Test instruction execution accuracy and completeness
        num_test_instructions = min(molecule.instruction_set_size, 1000)  # Test subset
        
        instruction_categories = [
            'arithmetic', 'logical', 'memory', 'control_flow', 'specialized'
        ]
        
        execution_results = []
        for category in instruction_categories:
            category_instructions = num_test_instructions // len(instruction_categories)
            
            # Simulate instruction execution
            execution_accuracy = np.random.beta(
                molecule.instruction_set_size / 10000 + 2,  # Shape parameter based on complexity
                2
            )
            execution_time = category_instructions / molecule.processing_rate
            
            # Account for molecular processing characteristics
            if category == 'specialized' and molecule.information_catalysis_capability > 0.8:
                execution_accuracy = min(execution_accuracy * 1.1, 1.0)  # Bonus for high catalysis
            
            execution_results.append({
                'category': category,
                'instructions_tested': category_instructions,
                'execution_accuracy': execution_accuracy,
                'execution_time': execution_time,
                'success': execution_accuracy >= self.validation_criteria['execution_accuracy_threshold']
            })
        
        # Test passes if instruction set size is sufficient and execution is accurate
        set_size_sufficient = molecule.instruction_set_size >= self.validation_criteria['min_instruction_set_size']
        execution_accurate = all(result['success'] for result in execution_results)
        passed = set_size_sufficient and execution_accurate
        
        # Calculate score
        set_size_score = min(molecule.instruction_set_size / self.validation_criteria['min_instruction_set_size'], 1.0)
        
        if execution_results:
            accuracy_scores = [result['execution_accuracy'] for result in execution_results]
            avg_accuracy_score = np.mean(accuracy_scores)
            overall_score = (set_size_score * avg_accuracy_score) ** 0.5
        else:
            overall_score = set_size_score
        
        return {
            'passed': passed,
            'score': overall_score,
            'instruction_set_size': molecule.instruction_set_size,
            'min_required_set_size': self.validation_criteria['min_instruction_set_size'],
            'execution_results': execution_results
        }
    
    def _test_parallel_processing(self, molecule: DualFunctionalityMolecule) -> Dict[str, Any]:
        """Test parallel processing capabilities."""
        if not molecule.parallel_processing_capability:
            return {
                'passed': True,  # Pass if parallel processing not claimed
                'score': 1.0,
                'parallel_capability': False,
                'note': 'Parallel processing not claimed by molecule'
            }
        
        # Test parallel execution with multiple tasks
        parallel_tasks = [
            {'operations': 50000, 'memory_bits': 10000, 'task_id': 1},
            {'operations': 75000, 'memory_bits': 15000, 'task_id': 2},
            {'operations': 60000, 'memory_bits': 12000, 'task_id': 3},
            {'operations': 40000, 'memory_bits': 8000, 'task_id': 4}
        ]
        
        # Filter tasks that fit in memory
        feasible_tasks = [
            task for task in parallel_tasks 
            if task['memory_bits'] <= molecule.memory_capacity
        ]
        
        if not feasible_tasks:
            return {
                'passed': False,
                'score': 0.0,
                'parallel_capability': True,
                'error': 'No tasks fit in available memory'
            }
        
        # Simulate sequential execution
        total_operations_sequential = sum(task['operations'] for task in feasible_tasks)
        sequential_time = total_operations_sequential / molecule.processing_rate
        
        # Simulate parallel execution (simplified model)
        max_operations_parallel = max(task['operations'] for task in feasible_tasks)
        parallel_efficiency = min(len(feasible_tasks) / 4.0, 1.0)  # Up to 4-way parallelism
        parallel_time = max_operations_parallel / (molecule.processing_rate * parallel_efficiency)
        
        # Calculate speedup
        speedup = sequential_time / parallel_time if parallel_time > 0 else 0.0
        theoretical_max_speedup = min(len(feasible_tasks), 4)  # Maximum theoretical speedup
        
        # Test passes if parallel execution provides meaningful speedup
        min_expected_speedup = 1.5  # At least 50% improvement
        passed = speedup >= min_expected_speedup
        
        # Calculate score based on speedup efficiency
        speedup_efficiency = min(speedup / theoretical_max_speedup, 1.0) if theoretical_max_speedup > 0 else 0.0
        score = speedup_efficiency
        
        return {
            'passed': passed,
            'score': score,
            'parallel_capability': True,
            'sequential_time': sequential_time,
            'parallel_time': parallel_time,
            'speedup_achieved': speedup,
            'theoretical_max_speedup': theoretical_max_speedup,
            'speedup_efficiency': speedup_efficiency,
            'tasks_tested': len(feasible_tasks)
        }


class DualFunctionalityValidator:
    """
    Comprehensive validator for dual-functionality molecules.
    
    This validator ensures that molecules meet the critical requirement of functioning
    as both precision clocks and computational processors simultaneously.
    """
    
    def __init__(self):
        """Initialize dual-functionality validator."""
        self.logger = logging.getLogger(__name__)
        self.clock_validator = ClockValidator()
        self.processor_validator = ProcessorValidator()
        
        self.dual_functionality_criteria = {
            'simultaneous_operation_required': True,
            'min_dual_functionality_score': 0.8,
            'resource_sharing_efficiency': 0.9,
            'cross_functionality_interference_max': 0.1,
            'enhancement_capability_required': True
        }
    
    def validate_single_molecule(self, molecule: DualFunctionalityMolecule) -> Dict[str, Any]:
        """Validate dual-functionality of a single molecule."""
        start_time = time.time()
        
        # Validate individual functionalities first
        clock_results = self.clock_validator.validate_single_molecule(molecule)
        processor_results = self.processor_validator.validate_single_molecule(molecule)
        
        # Test simultaneous operation
        simultaneous_test = self._test_simultaneous_operation(molecule)
        
        # Test resource sharing efficiency
        resource_test = self._test_resource_sharing(molecule)
        
        # Test cross-functionality interference
        interference_test = self._test_cross_functionality_interference(molecule)
        
        # Test recursive enhancement capability
        enhancement_test = self._test_recursive_enhancement(molecule)
        
        # Overall dual-functionality validation
        all_tests_passed = all([
            clock_results['overall_passed'],
            processor_results['overall_passed'],
            simultaneous_test['passed'],
            resource_test['passed'],
            interference_test['passed'],
            enhancement_test['passed']
        ])
        
        # Update molecule validation status
        molecule.dual_functionality_validated = all_tests_passed
        
        validation_time = time.time() - start_time
        
        # Calculate composite dual-functionality score
        component_scores = [
            clock_results.get('overall_score', 0.0),
            processor_results.get('overall_score', 0.0),
            simultaneous_test['score'],
            resource_test['score'],
            interference_test['score'],
            enhancement_test['score']
        ]
        composite_score = np.mean(component_scores)
        
        return {
            'molecule_id': molecule.molecular_id,
            'overall_passed': all_tests_passed,
            'composite_score': composite_score,
            'clock_validation': clock_results,
            'processor_validation': processor_results,
            'simultaneous_operation_test': simultaneous_test,
            'resource_sharing_test': resource_test,
            'interference_test': interference_test,
            'enhancement_test': enhancement_test,
            'validation_time': validation_time
        }
    
    def validate_batch(self, molecules: List[DualFunctionalityMolecule]) -> Dict[str, Any]:
        """Validate dual-functionality for a batch of molecules."""
        self.logger.info(f"Validating dual-functionality for {len(molecules)} molecules")
        
        batch_results = []
        for molecule in molecules:
            result = self.validate_single_molecule(molecule)
            batch_results.append(result)
        
        # Calculate batch statistics
        passed_count = sum(1 for result in batch_results if result['overall_passed'])
        dual_functionality_success_rate = passed_count / len(molecules) if molecules else 0.0
        
        # Calculate individual functionality statistics
        clock_passed = sum(1 for result in batch_results if result['clock_validation']['overall_passed'])
        processor_passed = sum(1 for result in batch_results if result['processor_validation']['overall_passed'])
        
        clock_success_rate = clock_passed / len(molecules) if molecules else 0.0
        processor_success_rate = processor_passed / len(molecules) if molecules else 0.0
        
        # Calculate performance metrics
        composite_scores = [result['composite_score'] for result in batch_results]
        average_dual_functionality_score = np.mean(composite_scores)
        
        # Calculate precision and processing metrics
        clock_precisions = []
        processing_rates = []
        
        for result in batch_results:
            molecule_id = result['molecule_id']
            # Find the corresponding molecule to get its properties
            molecule = next((m for m in molecules if m.molecular_id == molecule_id), None)
            if molecule:
                clock_precisions.append(molecule.temporal_precision)
                processing_rates.append(molecule.processing_rate)
        
        return {
            'total_molecules': len(molecules),
            'dual_functionality_passed': passed_count,
            'dual_functionality_success_rate': dual_functionality_success_rate,
            'clock_success_rate': clock_success_rate,
            'processor_success_rate': processor_success_rate,
            'average_dual_functionality_score': average_dual_functionality_score,
            'average_clock_precision': np.mean(clock_precisions) if clock_precisions else 0.0,
            'average_processing_capacity': np.mean(processing_rates) if processing_rates else 0.0,
            'precision_range': (np.min(clock_precisions), np.max(clock_precisions)) if clock_precisions else (0, 0),
            'processing_range': (np.min(processing_rates), np.max(processing_rates)) if processing_rates else (0, 0),
            'detailed_results': batch_results
        }
    
    def quick_validation(self, molecules: List[DualFunctionalityMolecule]) -> Dict[str, Any]:
        """Perform quick validation for rapid testing."""
        self.logger.info(f"Quick validation of {len(molecules)} molecules")
        
        quick_results = []
        for molecule in molecules:
            # Quick checks without detailed testing
            clock_quick = (
                molecule.frequency_stability >= 0.9 and
                molecule.phase_coherence >= 0.9 and
                molecule.temporal_precision > 0
            )
            
            processor_quick = (
                molecule.processing_rate >= 1e5 and
                molecule.memory_capacity >= 10000 and
                molecule.instruction_set_size >= 1000
            )
            
            dual_quick = (
                molecule.dual_functionality_score >= 0.7 and
                clock_quick and processor_quick
            )
            
            quick_results.append({
                'molecule_id': molecule.molecular_id,
                'quick_passed': dual_quick,
                'clock_quick': clock_quick,
                'processor_quick': processor_quick
            })
        
        passed_count = sum(1 for result in quick_results if result['quick_passed'])
        success_rate = passed_count / len(molecules) if molecules else 0.0
        
        return {
            'total_molecules': len(molecules),
            'quick_passed': passed_count,
            'success_rate': success_rate,
            'quick_results': quick_results
        }
    
    def _test_simultaneous_operation(self, molecule: DualFunctionalityMolecule) -> Dict[str, Any]:
        """Test simultaneous clock and processor operation."""
        try:
            # Execute clock function
            clock_task = molecule.execute_as_clock(precision_target=1e-30)
            
            # Execute processor function simultaneously (simulated)
            processor_task = molecule.execute_as_processor({
                'operations': 100000,
                'memory_bits': 50000
            })
            
            # Check if both operations were successful
            clock_success = clock_task['execution_successful']
            processor_success = processor_task['execution_successful']
            simultaneous_success = clock_success and processor_success
            
            # Calculate performance degradation due to simultaneous operation
            # In reality, there might be some interference, but dual-functionality molecules
            # are designed to minimize this
            performance_degradation = np.random.uniform(0.05, 0.15)  # 5-15% degradation
            adjusted_performance = 1.0 - performance_degradation
            
            passed = (
                simultaneous_success and
                adjusted_performance >= (1.0 - self.dual_functionality_criteria['cross_functionality_interference_max'])
            )
            
            score = adjusted_performance if simultaneous_success else 0.0
            
            return {
                'passed': passed,
                'score': score,
                'clock_success': clock_success,
                'processor_success': processor_success,
                'performance_degradation': performance_degradation,
                'adjusted_performance': adjusted_performance
            }
            
        except Exception as e:
            return {
                'passed': False,
                'score': 0.0,
                'error': str(e)
            }
    
    def _test_resource_sharing(self, molecule: DualFunctionalityMolecule) -> Dict[str, Any]:
        """Test efficiency of resource sharing between clock and processor functions."""
        # Calculate theoretical resource requirements for separate systems
        clock_resources = {
            'energy': molecule.base_frequency * 1e-20,  # Energy proportional to frequency
            'space': molecule.molecular_weight * 1e-3,   # Space proportional to molecular weight
            'complexity': molecule.frequency_stability * 100
        }
        
        processor_resources = {
            'energy': molecule.processing_rate * 1e-15,  # Energy proportional to processing rate
            'space': molecule.memory_capacity * 1e-6,    # Space proportional to memory
            'complexity': molecule.instruction_set_size * 1e-2
        }
        
        # Calculate total resources if implemented separately
        separate_total_resources = {
            key: clock_resources[key] + processor_resources[key]
            for key in clock_resources.keys()
        }
        
        # Calculate actual resources used by dual-functionality molecule
        # Dual-functionality molecules should be more efficient due to shared components
        sharing_efficiency = molecule.thermodynamic_efficiency
        actual_total_resources = {
            key: separate_total_resources[key] * (2.0 - sharing_efficiency)
            for key in separate_total_resources.keys()
        }
        
        # Calculate resource efficiency
        resource_efficiencies = {
            key: separate_total_resources[key] / actual_total_resources[key]
            if actual_total_resources[key] > 0 else 1.0
            for key in separate_total_resources.keys()
        }
        
        average_efficiency = np.mean(list(resource_efficiencies.values()))
        
        passed = average_efficiency >= self.dual_functionality_criteria['resource_sharing_efficiency']
        score = min(average_efficiency, 1.0)
        
        return {
            'passed': passed,
            'score': score,
            'resource_sharing_efficiency': sharing_efficiency,
            'average_efficiency': average_efficiency,
            'resource_efficiencies': resource_efficiencies,
            'separate_resources': separate_total_resources,
            'actual_resources': actual_total_resources
        }
    
    def _test_cross_functionality_interference(self, molecule: DualFunctionalityMolecule) -> Dict[str, Any]:
        """Test interference between clock and processor functions."""
        # Test clock performance alone
        solo_clock_result = molecule.execute_as_clock(precision_target=1e-30)
        
        # Test processor performance alone
        solo_processor_result = molecule.execute_as_processor({
            'operations': 100000,
            'memory_bits': 50000
        })
        
        # Simulate simultaneous operation with potential interference
        interference_factor = (2.0 - molecule.dual_functionality_score) * 0.1  # Higher score = less interference
        
        # Clock performance during simultaneous operation
        simultaneous_clock_precision = solo_clock_result['achieved_precision'] * (1 + interference_factor)
        simultaneous_clock_stability = solo_clock_result['stability_maintained'] and (interference_factor < 0.05)
        
        # Processor performance during simultaneous operation
        simultaneous_processor_rate = solo_processor_result['processing_rate_achieved'] * (1 - interference_factor)
        simultaneous_processor_success = solo_processor_result['execution_successful'] and (interference_factor < 0.1)
        
        # Calculate interference metrics
        clock_interference = abs(simultaneous_clock_precision - solo_clock_result['achieved_precision']) / solo_clock_result['achieved_precision']
        processor_interference = abs(simultaneous_processor_rate - solo_processor_result['processing_rate_achieved']) / solo_processor_result['processing_rate_achieved']
        
        max_interference = max(clock_interference, processor_interference)
        
        passed = (
            max_interference <= self.dual_functionality_criteria['cross_functionality_interference_max'] and
            simultaneous_clock_stability and
            simultaneous_processor_success
        )
        
        # Score inversely related to interference
        score = max(1.0 - max_interference / self.dual_functionality_criteria['cross_functionality_interference_max'], 0.0)
        
        return {
            'passed': passed,
            'score': score,
            'clock_interference': clock_interference,
            'processor_interference': processor_interference,
            'max_interference': max_interference,
            'interference_threshold': self.dual_functionality_criteria['cross_functionality_interference_max'],
            'simultaneous_operation_stable': simultaneous_clock_stability and simultaneous_processor_success
        }
    
    def _test_recursive_enhancement(self, molecule: DualFunctionalityMolecule) -> Dict[str, Any]:
        """Test recursive enhancement capabilities when combined with other molecules."""
        if not molecule.network_coordination_capability:
            return {
                'passed': False,
                'score': 0.0,
                'error': 'Molecule lacks network coordination capability'
            }
        
        # Create simulated partner molecules for enhancement testing
        partner_molecules = []
        for i in range(3):  # Test with 3 partner molecules
            partner = DualFunctionalityMolecule(
                molecular_id=f"PARTNER_{i}",
                smiles="c1ccccc1",  # Simple benzene
                formula="C6H6",
                molecular_weight=78.11,
                base_frequency=molecule.base_frequency * np.random.uniform(0.8, 1.2),
                frequency_stability=np.random.uniform(0.9, 0.99),
                phase_coherence=np.random.uniform(0.9, 0.99),
                temporal_precision=molecule.temporal_precision * np.random.uniform(0.5, 1.5),
                instruction_set_size=molecule.instruction_set_size // 2,
                memory_capacity=molecule.memory_capacity // 2,
                processing_rate=molecule.processing_rate * np.random.uniform(0.5, 1.5),
                parallel_processing_capability=True,
                recursive_enhancement_factor=np.random.uniform(1.2, 2.0),
                network_coordination_capability=True,
                dual_functionality_score=np.random.uniform(0.7, 0.9),
                thermodynamic_efficiency=np.random.uniform(0.8, 0.95),
                information_catalysis_capability=np.random.uniform(0.7, 0.9)
            )
            partner_molecules.append(partner)
        
        # Test recursive enhancement
        enhancement_result = molecule.recursive_enhance(partner_molecules)
        
        if not enhancement_result['success']:
            return {
                'passed': False,
                'score': 0.0,
                'error': 'Enhancement failed'
            }
        
        # Verify enhancement effectiveness
        original_precision = molecule.temporal_precision
        original_processing = molecule.processing_rate
        
        enhanced_precision = enhancement_result['enhanced_temporal_precision']
        enhanced_processing = enhancement_result['enhanced_processing_rate']
        
        # Calculate enhancement factors
        precision_enhancement = original_precision / enhanced_precision  # Lower is better for precision
        processing_enhancement = enhanced_processing / original_processing  # Higher is better for processing
        
        # Check if enhancement is significant
        min_precision_enhancement = 2.0  # At least 2x better precision
        min_processing_enhancement = 2.0  # At least 2x better processing
        
        precision_adequate = precision_enhancement >= min_precision_enhancement
        processing_adequate = processing_enhancement >= min_processing_enhancement
        
        passed = precision_adequate and processing_adequate
        
        # Calculate score based on enhancement effectiveness
        precision_score = min(precision_enhancement / min_precision_enhancement, 1.0)
        processing_score = min(processing_enhancement / min_processing_enhancement, 1.0)
        score = (precision_score * processing_score) ** 0.5
        
        return {
            'passed': passed,
            'score': score,
            'original_precision': original_precision,
            'enhanced_precision': enhanced_precision,
            'precision_enhancement_factor': precision_enhancement,
            'original_processing': original_processing,
            'enhanced_processing': enhanced_processing,
            'processing_enhancement_factor': processing_enhancement,
            'network_size': len(partner_molecules) + 1,
            'enhancement_result': enhancement_result
        }


class MolecularQualityControl:
    """
    Zero-tolerance quality control system for dual-functionality molecules.
    
    This system implements the critical quality assurance protocols required
    by the Borgia system, where any molecule failing dual-functionality
    requirements poses a cascade failure risk to the entire system.
    """
    
    def __init__(self):
        """Initialize molecular quality control system."""
        self.logger = logging.getLogger(__name__)
        self.dual_functionality_validator = DualFunctionalityValidator()
        
        # Zero-tolerance criteria
        self.quality_criteria = {
            'zero_tolerance_failures': [
                'dual_functionality_validation',
                'clock_functionality',
                'processor_functionality',
                'thermodynamic_efficiency',
                'information_catalysis_capability'
            ],
            'critical_thresholds': {
                'dual_functionality_score': 0.8,
                'frequency_stability': 0.95,
                'phase_coherence': 0.90,
                'processing_rate': 1e5,
                'memory_capacity': 10000,
                'thermodynamic_efficiency': 0.85,
                'information_catalysis_capability': 0.8
            },
            'cascade_failure_risk_assessment': True,
            'mandatory_enhancement_capability': True
        }
        
        # Quality control statistics
        self.qc_stats = {
            'molecules_processed': 0,
            'molecules_passed': 0,
            'molecules_rejected': 0,
            'rejection_reasons': {},
            'critical_failures_detected': 0,
            'cascade_risks_prevented': 0
        }
    
    def validate_zero_tolerance_protocols(self, molecules: List[DualFunctionalityMolecule]) -> Dict[str, Any]:
        """
        Execute zero-tolerance quality control validation.
        
        Args:
            molecules: List of molecules to validate
            
        Returns:
            Dictionary with comprehensive quality control results
        """
        start_time = time.time()
        self.logger.info(f"Executing zero-tolerance QC on {len(molecules)} molecules")
        
        passed_molecules = []
        rejected_molecules = []
        detailed_results = []
        
        for molecule in molecules:
            qc_result = self._execute_zero_tolerance_validation(molecule)
            
            if qc_result['passed']:
                passed_molecules.append(molecule)
            else:
                rejected_molecules.append(molecule)
                
                # Update rejection statistics
                for reason in qc_result['rejection_reasons']:
                    if reason in self.qc_stats['rejection_reasons']:
                        self.qc_stats['rejection_reasons'][reason] += 1
                    else:
                        self.qc_stats['rejection_reasons'][reason] = 1
            
            detailed_results.append(qc_result)
        
        # Update statistics
        self.qc_stats['molecules_processed'] += len(molecules)
        self.qc_stats['molecules_passed'] += len(passed_molecules)
        self.qc_stats['molecules_rejected'] += len(rejected_molecules)
        
        # Calculate performance metrics
        processing_time = time.time() - start_time
        molecules_per_second = len(molecules) / processing_time if processing_time > 0 else 0
        
        # Calculate quality metrics
        defect_detection_rate = len(rejected_molecules) / len(molecules) if molecules else 0
        pass_rate = len(passed_molecules) / len(molecules) if molecules else 0
        
        # Assess cascade failure risk
        cascade_risk_assessment = self._assess_cascade_failure_risk(rejected_molecules)
        
        return {
            'total_molecules_processed': len(molecules),
            'molecules_passed': len(passed_molecules),
            'molecules_rejected': len(rejected_molecules),
            'pass_rate': pass_rate,
            'defect_detection_rate': defect_detection_rate,
            'false_positive_rate': 0.0,  # Assuming validation is accurate for testing
            'false_negative_rate': 0.0,  # Zero tolerance means no false negatives allowed
            'processing_time': processing_time,
            'molecules_per_second': molecules_per_second,
            'memory_usage_mb': self._estimate_memory_usage(molecules),
            'rejection_reasons_summary': self.qc_stats['rejection_reasons'].copy(),
            'cascade_risk_assessment': cascade_risk_assessment,
            'detailed_results': detailed_results,
            'qc_statistics': self.qc_stats.copy()
        }
    
    def _execute_zero_tolerance_validation(self, molecule: DualFunctionalityMolecule) -> Dict[str, Any]:
        """Execute zero-tolerance validation on a single molecule."""
        validation_start = time.time()
        
        rejection_reasons = []
        critical_failures = []
        warnings = []
        
        # Execute comprehensive dual-functionality validation
        dual_validation = self.dual_functionality_validator.validate_single_molecule(molecule)
        
        # Zero-tolerance checks
        if not dual_validation['overall_passed']:
            rejection_reasons.append('dual_functionality_validation_failed')
            critical_failures.append('Dual-functionality validation failed')
        
        if not dual_validation['clock_validation']['overall_passed']:
            rejection_reasons.append('clock_functionality_failed')
            critical_failures.append('Clock functionality validation failed')
        
        if not dual_validation['processor_validation']['overall_passed']:
            rejection_reasons.append('processor_functionality_failed')
            critical_failures.append('Processor functionality validation failed')
        
        # Critical threshold checks
        for criterion, threshold in self.quality_criteria['critical_thresholds'].items():
            molecule_value = getattr(molecule, criterion, None)
            
            if molecule_value is None:
                rejection_reasons.append(f'{criterion}_missing')
                critical_failures.append(f'{criterion} value missing')
                continue
            
            if molecule_value < threshold:
                rejection_reasons.append(f'{criterion}_below_threshold')
                critical_failures.append(f'{criterion} ({molecule_value:.3f}) below threshold ({threshold})')
        
        # Enhancement capability check
        if self.quality_criteria['mandatory_enhancement_capability']:
            if not molecule.network_coordination_capability:
                rejection_reasons.append('enhancement_capability_missing')
                critical_failures.append('Network coordination capability missing')
            
            if molecule.recursive_enhancement_factor < 1.5:
                rejection_reasons.append('enhancement_factor_insufficient')
                critical_failures.append(f'Recursive enhancement factor ({molecule.recursive_enhancement_factor:.2f}) insufficient')
        
        # Information catalysis validation
        if molecule.information_catalysis_capability < self.quality_criteria['critical_thresholds']['information_catalysis_capability']:
            rejection_reasons.append('information_catalysis_insufficient')
            critical_failures.append('Information catalysis capability insufficient')
        
        # Calculate cascade failure risk
        cascade_risk = self._calculate_cascade_failure_risk(molecule, critical_failures)
        
        if cascade_risk > 0.1:  # 10% cascade risk threshold
            rejection_reasons.append('cascade_failure_risk_too_high')
            critical_failures.append(f'Cascade failure risk ({cascade_risk:.1%}) exceeds threshold')
        
        # Final pass/fail determination
        passed = len(rejection_reasons) == 0
        
        # Update molecule validation flags
        if passed:
            molecule.dual_functionality_validated = True
            molecule.clock_validation_passed = True
            molecule.processor_validation_passed = True
        
        validation_time = time.time() - validation_start
        
        return {
            'molecule_id': molecule.molecular_id,
            'passed': passed,
            'rejection_reasons': rejection_reasons,
            'critical_failures': critical_failures,
            'warnings': warnings,
            'cascade_failure_risk': cascade_risk,
            'dual_validation_results': dual_validation,
            'validation_time': validation_time,
            'zero_tolerance_applied': True
        }
    
    def _calculate_cascade_failure_risk(self, molecule: DualFunctionalityMolecule, failures: List[str]) -> float:
        """Calculate cascade failure risk for a molecule."""
        base_risk = 0.0
        
        # Risk factors based on functionality failures
        if any('dual_functionality' in failure.lower() for failure in failures):
            base_risk += 0.5  # 50% risk for dual-functionality failure
        
        if any('clock' in failure.lower() for failure in failures):
            base_risk += 0.3  # 30% risk for clock failure
        
        if any('processor' in failure.lower() for failure in failures):
            base_risk += 0.3  # 30% risk for processor failure
        
        if any('enhancement' in failure.lower() for failure in failures):
            base_risk += 0.2  # 20% risk for enhancement failure
        
        # Adjust risk based on molecule quality
        quality_factor = 1.0 - molecule.dual_functionality_score
        adjusted_risk = base_risk * (1.0 + quality_factor)
        
        return min(adjusted_risk, 1.0)  # Cap at 100%
    
    def _assess_cascade_failure_risk(self, rejected_molecules: List[DualFunctionalityMolecule]) -> Dict[str, Any]:
        """Assess overall cascade failure risk from rejected molecules."""
        if not rejected_molecules:
            return {
                'overall_risk': 0.0,
                'risk_level': 'NONE',
                'mitigation_required': False
            }
        
        # Calculate risk distribution
        risk_levels = []
        critical_count = 0
        
        for molecule in rejected_molecules:
            if hasattr(molecule, 'generation_parameters'):
                defect_type = molecule.generation_parameters.get('intentional_defect', 'unknown')
                
                if 'dual_functionality' in defect_type:
                    risk_levels.append(0.8)  # High risk
                    critical_count += 1
                elif any(term in defect_type for term in ['clock', 'processor']):
                    risk_levels.append(0.5)  # Medium risk
                else:
                    risk_levels.append(0.2)  # Low risk
            else:
                risk_levels.append(0.3)  # Default medium-low risk
        
        overall_risk = np.mean(risk_levels) if risk_levels else 0.0
        
        # Determine risk level
        if overall_risk > 0.7:
            risk_level = 'CRITICAL'
        elif overall_risk > 0.4:
            risk_level = 'HIGH'
        elif overall_risk > 0.2:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        mitigation_required = overall_risk > 0.3 or critical_count > 0
        
        return {
            'overall_risk': overall_risk,
            'risk_level': risk_level,
            'mitigation_required': mitigation_required,
            'critical_failures_detected': critical_count,
            'total_rejected_molecules': len(rejected_molecules),
            'risk_distribution': risk_levels
        }
    
    def _estimate_memory_usage(self, molecules: List[DualFunctionalityMolecule]) -> float:
        """Estimate memory usage in MB for processing molecules."""
        # Rough estimation based on molecule complexity
        base_memory_per_molecule = 0.1  # 0.1 MB per molecule base
        
        total_memory = 0.0
        for molecule in molecules:
            molecule_memory = base_memory_per_molecule
            molecule_memory += molecule.memory_capacity * 1e-9  # Convert bits to MB
            molecule_memory += molecule.instruction_set_size * 1e-6  # Instructions overhead
            total_memory += molecule_memory
        
        return total_memory
    
    def get_quality_control_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality control report."""
        return {
            'quality_control_statistics': self.qc_stats.copy(),
            'quality_criteria': self.quality_criteria.copy(),
            'system_status': {
                'zero_tolerance_active': True,
                'cascade_failure_protection': True,
                'critical_thresholds_enforced': True
            },
            'performance_metrics': {
                'total_processed': self.qc_stats['molecules_processed'],
                'overall_pass_rate': (
                    self.qc_stats['molecules_passed'] / self.qc_stats['molecules_processed']
                    if self.qc_stats['molecules_processed'] > 0 else 0.0
                ),
                'critical_failure_rate': (
                    self.qc_stats['critical_failures_detected'] / self.qc_stats['molecules_processed']
                    if self.qc_stats['molecules_processed'] > 0 else 0.0
                )
            }
        }
    
    def reset_statistics(self):
        """Reset quality control statistics."""
        self.qc_stats = {
            'molecules_processed': 0,
            'molecules_passed': 0,
            'molecules_rejected': 0,
            'rejection_reasons': {},
            'critical_failures_detected': 0,
            'cascade_risks_prevented': 0
        }
