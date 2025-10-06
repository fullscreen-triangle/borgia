
"""
Empty Dictionary Synthesis for Dynamic Molecular Discovery

Implements dynamic molecular identification synthesis without static storage,
operating through equilibrium-seeking coordinate navigation in chemical space.
Addresses the vast chemical space problem (10⁶⁰+ possible molecules).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Optional
import hashlib
from collections import defaultdict

class EmptyDictionaryEngine:
    """
    Dynamic molecular identification synthesis without database storage.
    Operates as semantic gas molecules where queries create perturbations
    resolved through coordinate navigation.
    """

    def __init__(self):
        self.dictionary_state = {}  # Always empty - no static storage
        self.synthesis_cache = {}   # Temporary synthesis results
        self.pressure_state = 0.0   # System pressure
        self.perturbation_history = []
        self.coordinate_navigator = None

    def initialize_coordinate_navigator(self, smarts_data: List[str]):
        """Initialize coordinate navigation system for synthesis."""
        print("🧭 Initializing coordinate navigation system...")

        # Create coordinate mappings for known molecules
        self.coordinate_mappings = {}

        for i, smarts in enumerate(smarts_data):
            mol_id = f"mol_{i}"

            # Calculate semantic coordinates
            semantic_coords = self._calculate_semantic_coordinates(smarts)

            # Calculate synthesis coordinates
            synthesis_coords = self._calculate_synthesis_coordinates(smarts)

            # Store coordinate mapping (temporary for navigation)
            self.coordinate_mappings[mol_id] = {
                'smarts': smarts,
                'semantic_coords': semantic_coords,
                'synthesis_coords': synthesis_coords,
                'molecular_hash': self._calculate_molecular_hash(smarts)
            }

        print(f"✅ Initialized navigation for {len(self.coordinate_mappings)} molecules")

    def _calculate_semantic_coordinates(self, smarts: str) -> Tuple[float, float, float, float]:
        """Calculate 4D semantic coordinates (technical, emotional, action, descriptive)."""
        # Technical dimension - structural complexity
        technical = self._calculate_technical_score(smarts)

        # Emotional dimension - molecular "personality"
        emotional = self._calculate_emotional_score(smarts)

        # Action dimension - reactivity potential
        action = self._calculate_action_score(smarts)

        # Descriptive dimension - information content
        descriptive = self._calculate_descriptive_score(smarts)

        return (technical, emotional, action, descriptive)

    def _calculate_technical_score(self, smarts: str) -> float:
        """Calculate technical complexity score."""
        if not smarts:
            return 0.0

        # Count technical features
        technical_features = {
            'atoms': len([c for c in smarts if c.isupper()]),
            'bonds': smarts.count('=') + smarts.count('#') + smarts.count('-'),
            'rings': smarts.count('1') + smarts.count('2') + smarts.count('3'),
            'branches': smarts.count('(') + smarts.count('['),
            'charges': smarts.count('+') + smarts.count('-'),
            'aromatic': smarts.count(':')
        }

        # Weighted technical score
        weights = [0.25, 0.2, 0.2, 0.15, 0.1, 0.1]
        max_values = [20, 25, 10, 8, 5, 15]  # Normalization factors

        technical_score = sum(
            w * min(v / max_v, 1.0)
            for w, v, max_v in zip(weights, technical_features.values(), max_values)
        )

        return min(technical_score, 1.0)

    def _calculate_emotional_score(self, smarts: str) -> float:
        """Calculate molecular emotional score (stability, harmony)."""
        if not smarts:
            return 0.0

        # Emotional factors
        stability_factors = {
            'symmetry': self._calculate_symmetry(smarts),
            'balance': self._calculate_balance(smarts),
            'harmony': self._calculate_harmony(smarts),
            'elegance': self._calculate_elegance(smarts)
        }

        emotional_score = sum(stability_factors.values()) / len(stability_factors)
        return min(max(emotional_score, 0.0), 1.0)

    def _calculate_action_score(self, smarts: str) -> float:
        """Calculate molecular action/reactivity score."""
        if not smarts:
            return 0.0

        # Reactivity indicators
        reactive_features = {
            'double_bonds': smarts.count('=') / max(len(smarts), 1),
            'triple_bonds': smarts.count('#') / max(len(smarts), 1),
            'heteroatoms': (smarts.count('O') + smarts.count('N') + smarts.count('S')) / max(len(smarts), 1),
            'charges': (smarts.count('+') + smarts.count('-')) / max(len(smarts), 1),
            'strain': self._calculate_strain_indicators(smarts)
        }

        action_score = sum(reactive_features.values()) / len(reactive_features)
        return min(action_score, 1.0)

    def _calculate_descriptive_score(self, smarts: str) -> float:
        """Calculate descriptive information content score."""
        if not smarts:
            return 0.0

        # Information content measures
        char_entropy = self._calculate_character_entropy(smarts)
        pattern_complexity = self._calculate_pattern_complexity(smarts)
        uniqueness = self._calculate_uniqueness_score(smarts)

        descriptive_score = (char_entropy + pattern_complexity + uniqueness) / 3.0
        return min(max(descriptive_score, 0.0), 1.0)

    def _calculate_synthesis_coordinates(self, smarts: str) -> Tuple[float, float]:
        """Calculate synthesis navigation coordinates."""
        # Synthesis difficulty
        difficulty = self._calculate_synthesis_difficulty(smarts)

        # Synthesis accessibility
        accessibility = self._calculate_synthesis_accessibility(smarts)

        return (difficulty, accessibility)

    def _calculate_molecular_hash(self, smarts: str) -> str:
        """Calculate unique molecular hash for identification."""
        return hashlib.md5(smarts.encode()).hexdigest()[:16]

    def synthesize_molecular_identity(self, query: str, context: Dict = None) -> Dict:
        """
        Synthesize molecular identity through dynamic coordinate navigation.
        No static storage - everything computed on demand.
        """
        start_time = time.time()

        # Create query perturbation
        perturbation = self._create_query_perturbation(query, context)

        # Update system pressure
        self._update_system_pressure(perturbation)

        # Navigate to equilibrium
        equilibrium_state = self._navigate_to_equilibrium(perturbation)

        # Synthesize molecular identity
        identity = self._synthesize_identity_from_equilibrium(equilibrium_state, query)

        # Record synthesis
        synthesis_time = time.time() - start_time
        self._record_synthesis(query, identity, synthesis_time, perturbation)

        return identity

    def _create_query_perturbation(self, query: str, context: Dict = None) -> Dict:
        """Create system perturbation from query."""
        # Calculate query coordinates
        if self._is_smarts_query(query):
            # Direct SMARTS query
            semantic_coords = self._calculate_semantic_coordinates(query)
            synthesis_coords = self._calculate_synthesis_coordinates(query)
        else:
            # Text-based query - convert to coordinates
            semantic_coords = self._text_to_semantic_coordinates(query)
            synthesis_coords = self._estimate_synthesis_coordinates_from_text(query)

        perturbation = {
            'query': query,
            'semantic_coords': semantic_coords,
            'synthesis_coords': synthesis_coords,
            'context': context or {},
            'timestamp': time.time(),
            'perturbation_magnitude': self._calculate_perturbation_magnitude(semantic_coords)
        }

        return perturbation

    def _update_system_pressure(self, perturbation: Dict):
        """Update system pressure based on perturbation."""
        magnitude = perturbation['perturbation_magnitude']

        # Pressure increases with perturbation
        pressure_increment = magnitude * 0.1
        self.pressure_state += pressure_increment

        # Pressure decay over time
        current_time = time.time()
        if self.perturbation_history:
            last_time = self.perturbation_history[-1]['timestamp']
            time_decay = (current_time - last_time) * 0.05
            self.pressure_state = max(0.0, self.pressure_state - time_decay)

        # Record perturbation
        self.perturbation_history.append(perturbation)

        # Limit history size
        if len(self.perturbation_history) > 100:
            self.perturbation_history = self.perturbation_history[-50:]

    def _navigate_to_equilibrium(self, perturbation: Dict) -> Dict:
        """Navigate to equilibrium state through coordinate space."""
        target_coords = perturbation['semantic_coords']

        # Find nearest molecular coordinates
        nearest_molecules = self._find_nearest_molecules(target_coords)

        # Calculate equilibrium position
        equilibrium_coords = self._calculate_equilibrium_coordinates(target_coords, nearest_molecules)

        # Calculate equilibrium properties
        equilibrium_state = {
            'coordinates': equilibrium_coords,
            'nearest_molecules': nearest_molecules,
            'stability': self._calculate_equilibrium_stability(equilibrium_coords),
            'convergence_path': self._trace_convergence_path(target_coords, equilibrium_coords),
            'synthesis_feasibility': self._assess_synthesis_feasibility(equilibrium_coords)
        }

        return equilibrium_state

    def _find_nearest_molecules(self, target_coords: Tuple[float, float, float, float], k: int = 5) -> List[Dict]:
        """Find k nearest molecules in coordinate space."""
        if not self.coordinate_mappings:
            return []

        distances = []

        for mol_id, mol_data in self.coordinate_mappings.items():
            mol_coords = mol_data['semantic_coords']

            # Calculate 4D Euclidean distance
            distance = np.sqrt(sum((a - b)**2 for a, b in zip(target_coords, mol_coords)))

            distances.append({
                'mol_id': mol_id,
                'distance': distance,
                'smarts': mol_data['smarts'],
                'coordinates': mol_coords,
                'molecular_hash': mol_data['molecular_hash']
            })

        # Sort by distance and return k nearest
        distances.sort(key=lambda x: x['distance'])
        return distances[:k]

    def _calculate_equilibrium_coordinates(self, target_coords: Tuple, nearest_molecules: List[Dict]) -> Tuple:
        """Calculate equilibrium coordinates using weighted average."""
        if not nearest_molecules:
            return target_coords

        # Weight by inverse distance
        total_weight = 0.0
        weighted_coords = [0.0, 0.0, 0.0, 0.0]

        for mol in nearest_molecules:
            weight = 1.0 / max(mol['distance'], 1e-6)
            total_weight += weight

            for i, coord in enumerate(mol['coordinates']):
                weighted_coords[i] += weight * coord

        # Normalize
        if total_weight > 0:
            equilibrium_coords = tuple(coord / total_weight for coord in weighted_coords)
        else:
            equilibrium_coords = target_coords

        return equilibrium_coords

    def _synthesize_identity_from_equilibrium(self, equilibrium_state: Dict, query: str) -> Dict:
        """Synthesize molecular identity from equilibrium state."""
        coords = equilibrium_state['coordinates']
        nearest = equilibrium_state['nearest_molecules']

        # Generate synthetic molecular properties
        identity = {
            'query': query,
            'synthesized_coordinates': coords,
            'confidence': self._calculate_synthesis_confidence(equilibrium_state),
            'molecular_class': self._classify_molecular_type(coords),
            'predicted_properties': self._predict_molecular_properties(coords),
            'synthesis_pathway': self._generate_synthesis_pathway(equilibrium_state),
            'similar_molecules': [mol['smarts'] for mol in nearest[:3]],
            'uniqueness_score': self._calculate_uniqueness_from_equilibrium(equilibrium_state),
            'synthesis_feasibility': equilibrium_state['synthesis_feasibility']
        }

        return identity

    def _calculate_synthesis_confidence(self, equilibrium_state: Dict) -> float:
        """Calculate confidence in synthesized identity."""
        stability = equilibrium_state['stability']
        nearest_count = len(equilibrium_state['nearest_molecules'])

        # Base confidence from stability
        confidence = stability * 0.7

        # Boost confidence with more nearby molecules
        neighbor_boost = min(nearest_count / 10.0, 0.3)
        confidence += neighbor_boost

        return min(max(confidence, 0.0), 1.0)

    def _predict_molecular_properties(self, coords: Tuple[float, float, float, float]) -> Dict:
        """Predict molecular properties from coordinates."""
        technical, emotional, action, descriptive = coords

        properties = {
            'molecular_weight': self._estimate_molecular_weight(technical, descriptive),
            'polarity': emotional,
            'reactivity': action,
            'stability': 1.0 - action,  # Inverse relationship
            'complexity': (technical + descriptive) / 2.0,
            'drug_likeness': self._estimate_drug_likeness(coords),
            'toxicity_risk': self._estimate_toxicity_risk(coords)
        }

        return properties

    def analyze_synthesis_performance(self, test_queries: List[str]) -> Dict:
        """Analyze empty dictionary synthesis performance."""
        print("🔄 Analyzing synthesis performance...")

        results = {
            'synthesis_times': [],
            'confidence_scores': [],
            'uniqueness_scores': [],
            'synthesis_feasibility': [],
            'pressure_evolution': [],
            'coordinate_coverage': {},
            'synthesis_statistics': {}
        }

        # Test synthesis on queries
        for i, query in enumerate(test_queries):
            start_time = time.time()

            # Synthesize identity
            identity = self.synthesize_molecular_identity(query)

            synthesis_time = time.time() - start_time

            # Record metrics
            results['synthesis_times'].append(synthesis_time)
            results['confidence_scores'].append(identity['confidence'])
            results['uniqueness_scores'].append(identity['uniqueness_score'])
            results['synthesis_feasibility'].append(identity['synthesis_feasibility'])
            results['pressure_evolution'].append(self.pressure_state)

        # Analyze coordinate coverage
        results['coordinate_coverage'] = self._analyze_coordinate_coverage()

        # Calculate synthesis statistics
        results['synthesis_statistics'] = {
            'avg_synthesis_time': float(np.mean(results['synthesis_times'])),
            'avg_confidence': float(np.mean(results['confidence_scores'])),
            'avg_uniqueness': float(np.mean(results['uniqueness_scores'])),
            'avg_feasibility': float(np.mean(results['synthesis_feasibility'])),
            'total_queries_processed': len(test_queries),
            'synthesis_throughput': len(test_queries) / sum(results['synthesis_times']),
            'pressure_stability': float(np.std(results['pressure_evolution']))
        }

        return results

    def _analyze_coordinate_coverage(self) -> Dict:
        """Analyze coverage of coordinate space."""
        if not self.coordinate_mappings:
            return {}

        # Extract all coordinates
        all_coords = [mol['semantic_coords'] for mol in self.coordinate_mappings.values()]
        coords_array = np.array(all_coords)

        # Calculate coverage statistics
        coverage = {
            'coordinate_ranges': {
                'technical': {'min': float(np.min(coords_array[:, 0])), 'max': float(np.max(coords_array[:, 0]))},
                'emotional': {'min': float(np.min(coords_array[:, 1])), 'max': float(np.max(coords_array[:, 1]))},
                'action': {'min': float(np.min(coords_array[:, 2])), 'max': float(np.max(coords_array[:, 2]))},
                'descriptive': {'min': float(np.min(coords_array[:, 3])), 'max': float(np.max(coords_array[:, 3]))}
            },
            'coordinate_correlations': self._calculate_coordinate_correlations(coords_array),
            'space_density': self._calculate_space_density(coords_array),
            'coverage_volume': self._calculate_coverage_volume(coords_array)
        }

        return coverage

    # Helper methods for coordinate calculations
    def _calculate_symmetry(self, smarts: str) -> float:
        """Calculate molecular symmetry."""
        if len(smarts) < 2:
            return 0.0
        mid = len(smarts) // 2
        left = smarts[:mid]
        right = smarts[mid:][::-1]
        matches = sum(1 for a, b in zip(left, right) if a == b)
        return matches / max(len(left), 1)

    def _calculate_balance(self, smarts: str) -> float:
        """Calculate molecular balance."""
        if not smarts:
            return 0.0
        char_counts = {}
        for char in smarts:
            char_counts[char] = char_counts.get(char, 0) + 1
        variance = np.var(list(char_counts.values()))
        return 1.0 / (1.0 + variance)

    def _calculate_harmony(self, smarts: str) -> float:
        """Calculate molecular harmony."""
        if len(smarts) < 2:
            return 0.0
        transitions = {}
        for i in range(len(smarts) - 1):
            pair = smarts[i:i+2]
            transitions[pair] = transitions.get(pair, 0) + 1
        entropy = -sum(count * np.log2(count) for count in transitions.values() if count > 0)
        return min(entropy / 10.0, 1.0)

    def _calculate_elegance(self, smarts: str) -> float:
        """Calculate molecular elegance."""
        if not smarts:
            return 0.0
        complexity = len(set(smarts))
        length = len(smarts)
        return complexity / max(length, 1)

    def _calculate_strain_indicators(self, smarts: str) -> float:
        """Calculate molecular strain indicators."""
        strain_features = smarts.count('(') + smarts.count('[') + smarts.count('3') + smarts.count('4')
        return min(strain_features / max(len(smarts), 1), 1.0)

    def _calculate_character_entropy(self, smarts: str) -> float:
        """Calculate character entropy."""
        if not smarts:
            return 0.0
        char_counts = {}
        for char in smarts:
            char_counts[char] = char_counts.get(char, 0) + 1
        total = len(smarts)
        entropy = -sum((count/total) * np.log2(count/total) for count in char_counts.values())
        return min(entropy / 4.0, 1.0)  # Normalize

    def _calculate_pattern_complexity(self, smarts: str) -> float:
        """Calculate pattern complexity."""
        patterns = set()
        for length in [2, 3]:
            for i in range(len(smarts) - length + 1):
                patterns.add(smarts[i:i+length])
        return min(len(patterns) / max(len(smarts), 1), 1.0)

    def _calculate_uniqueness_score(self, smarts: str) -> float:
        """Calculate uniqueness score."""
        unique_chars = len(set(smarts))
        total_chars = len(smarts)
        return unique_chars / max(total_chars, 1)

    def _calculate_synthesis_difficulty(self, smarts: str) -> float:
        """Calculate synthesis difficulty."""
        difficulty_factors = {
            'length': len(smarts) / 50.0,
            'complexity': len(set(smarts)) / 20.0,
            'rings': (smarts.count('1') + smarts.count('2')) / 10.0,
            'heteroatoms': (smarts.count('O') + smarts.count('N') + smarts.count('S')) / 10.0
        }
        return min(sum(difficulty_factors.values()) / len(difficulty_factors), 1.0)

    def _calculate_synthesis_accessibility(self, smarts: str) -> float:
        """Calculate synthesis accessibility."""
        # Inverse of difficulty with some adjustments
        difficulty = self._calculate_synthesis_difficulty(smarts)
        accessibility = 1.0 - difficulty

        # Boost for common patterns
        common_patterns = ['CC', 'CO', 'CN', 'C=C', 'C=O']
        boost = sum(0.1 for pattern in common_patterns if pattern in smarts)
        accessibility = min(accessibility + boost, 1.0)

        return accessibility

    def _is_smarts_query(self, query: str) -> bool:
        """Check if query is a SMARTS string."""
        smarts_chars = set('CNOSPFClBrI()[]=-#:+123456789')
        return len(query) > 2 and all(c in smarts_chars or c.isalpha() for c in query)

    def _text_to_semantic_coordinates(self, text: str) -> Tuple[float, float, float, float]:
        """Convert text query to semantic coordinates."""
        # Simple text-to-coordinate mapping
        text_lower = text.lower()

        # Technical score based on technical terms
        technical_terms = ['complex', 'advanced', 'sophisticated', 'intricate']
        technical = sum(0.25 for term in technical_terms if term in text_lower)

        # Emotional score based on descriptive terms
        emotional_terms = ['stable', 'balanced', 'harmonious', 'elegant']
        emotional = sum(0.25 for term in emotional_terms if term in text_lower)

        # Action score based on activity terms
        action_terms = ['reactive', 'active', 'dynamic', 'energetic']
        action = sum(0.25 for term in action_terms if term in text_lower)

        # Descriptive score based on information content
        descriptive = min(len(set(text_lower.split())) / 10.0, 1.0)

        return (technical, emotional, action, descriptive)

    def _estimate_synthesis_coordinates_from_text(self, text: str) -> Tuple[float, float]:
        """Estimate synthesis coordinates from text."""
        difficulty_terms = ['difficult', 'complex', 'challenging', 'advanced']
        easy_terms = ['simple', 'easy', 'basic', 'straightforward']

        text_lower = text.lower()

        difficulty = sum(0.25 for term in difficulty_terms if term in text_lower)
        ease = sum(0.25 for term in easy_terms if term in text_lower)

        synthesis_difficulty = max(difficulty - ease, 0.0)
        synthesis_accessibility = 1.0 - synthesis_difficulty

        return (synthesis_difficulty, synthesis_accessibility)

    def _calculate_perturbation_magnitude(self, coords: Tuple[float, float, float, float]) -> float:
        """Calculate perturbation magnitude."""
        return np.sqrt(sum(c**2 for c in coords))

    def _calculate_equilibrium_stability(self, coords: Tuple[float, float, float, float]) -> float:
        """Calculate equilibrium stability."""
        # Stability based on coordinate balance
        variance = np.var(coords)
        stability = 1.0 / (1.0 + variance)
        return stability

    def _trace_convergence_path(self, start_coords: Tuple, end_coords: Tuple) -> List[Tuple]:
        """Trace convergence path from start to end coordinates."""
        steps = 5
        path = []

        for i in range(steps + 1):
            t = i / steps
            interpolated = tuple(
                (1 - t) * start + t * end
                for start, end in zip(start_coords, end_coords)
            )
            path.append(interpolated)

        return path

    def _assess_synthesis_feasibility(self, coords: Tuple[float, float, float, float]) -> float:
        """Assess synthesis feasibility from coordinates."""
        technical, emotional, action, descriptive = coords

        # Feasibility factors
        complexity_penalty = technical * 0.3  # High complexity reduces feasibility
        stability_bonus = emotional * 0.4     # High stability increases feasibility
        reactivity_penalty = action * 0.2     # High reactivity can reduce feasibility
        information_bonus = descriptive * 0.1 # More information helps

        feasibility = 1.0 - complexity_penalty + stability_bonus - reactivity_penalty + information_bonus
        return min(max(feasibility, 0.0), 1.0)

    def _classify_molecular_type(self, coords: Tuple[float, float, float, float]) -> str:
        """Classify molecular type from coordinates."""
        technical, emotional, action, descriptive = coords

        if technical > 0.7:
            return 'complex_synthetic'
        elif emotional > 0.7:
            return 'stable_natural'
        elif action > 0.7:
            return 'reactive_intermediate'
        elif descriptive > 0.7:
            return 'well_characterized'
        else:
            return 'simple_organic'

    def _generate_synthesis_pathway(self, equilibrium_state: Dict) -> List[str]:
        """Generate synthesis pathway steps."""
        coords = equilibrium_state['coordinates']
        technical, emotional, action, descriptive = coords

        pathway = ['starting_materials']

        if technical > 0.5:
            pathway.append('protection_steps')

        if action > 0.6:
            pathway.append('coupling_reaction')
        else:
            pathway.append('substitution_reaction')

        if emotional < 0.4:  # Low stability
            pathway.append('stabilization_step')

        if technical > 0.6:
            pathway.append('deprotection_steps')

        pathway.append('purification')
        pathway.append('target_compound')

        return pathway

    def _calculate_uniqueness_from_equilibrium(self, equilibrium_state: Dict) -> float:
        """Calculate uniqueness score from equilibrium state."""
        nearest = equilibrium_state['nearest_molecules']

        if not nearest:
            return 1.0  # Completely unique

        # Average distance to nearest molecules
        avg_distance = sum(mol['distance'] for mol in nearest) / len(nearest)

        # Convert distance to uniqueness score
        uniqueness = min(avg_distance / 2.0, 1.0)

        return uniqueness

    def _estimate_molecular_weight(self, technical: float, descriptive: float) -> float:
        """Estimate molecular weight from coordinates."""
        # Simple estimation based on complexity
        base_weight = 100.0  # Base molecular weight
        complexity_factor = (technical + descriptive) / 2.0
        estimated_weight = base_weight + complexity_factor * 400.0

        return estimated_weight

    def _estimate_drug_likeness(self, coords: Tuple[float, float, float, float]) -> float:
        """Estimate drug-likeness from coordinates."""
        technical, emotional, action, descriptive = coords

        # Drug-like molecules are moderately complex, stable, not too reactive
        ideal_technical = 0.6
        ideal_emotional = 0.7
        ideal_action = 0.3
        ideal_descriptive = 0.8

        deviations = [
            abs(technical - ideal_technical),
            abs(emotional - ideal_emotional),
            abs(action - ideal_action),
            abs(descriptive - ideal_descriptive)
        ]

        avg_deviation = sum(deviations) / len(deviations)
        drug_likeness = 1.0 - avg_deviation

        return max(drug_likeness, 0.0)

    def _estimate_toxicity_risk(self, coords: Tuple[float, float, float, float]) -> float:
        """Estimate toxicity risk from coordinates."""
        technical, emotional, action, descriptive = coords

        # High reactivity and complexity increase toxicity risk
        risk_factors = [
            action * 0.4,      # High reactivity
            technical * 0.3,   # High complexity
            (1 - emotional) * 0.2,  # Low stability
            (1 - descriptive) * 0.1  # Low information
        ]

        toxicity_risk = sum(risk_factors)
        return min(toxicity_risk, 1.0)

    def _record_synthesis(self, query: str, identity: Dict, synthesis_time: float, perturbation: Dict):
        """Record synthesis in temporary cache."""
        # Store in temporary cache (not permanent storage)
        cache_key = hashlib.md5(query.encode()).hexdigest()

        self.synthesis_cache[cache_key] = {
            'query': query,
            'identity': identity,
            'synthesis_time': synthesis_time,
            'perturbation': perturbation,
            'timestamp': time.time()
        }

        # Limit cache size
        if len(self.synthesis_cache) > 1000:
            # Remove oldest entries
            sorted_entries = sorted(
                self.synthesis_cache.items(),
                key=lambda x: x[1]['timestamp']
            )

            # Keep only the most recent 500 entries
            self.synthesis_cache = dict(sorted_entries[-500:])

    def _calculate_coordinate_correlations(self, coords_array: np.ndarray) -> Dict:
        """Calculate correlations between coordinates."""
        if len(coords_array) < 2:
            return {}

        corr_matrix = np.corrcoef(coords_array.T)

        return {
            'technical_emotional': float(corr_matrix[0, 1]),
            'technical_action': float(corr_matrix[0, 2]),
            'technical_descriptive': float(corr_matrix[0, 3]),
            'emotional_action': float(corr_matrix[1, 2]),
            'emotional_descriptive': float(corr_matrix[1, 3]),
            'action_descriptive': float(corr_matrix[2, 3])
        }

    def _calculate_space_density(self, coords_array: np.ndarray) -> float:
        """Calculate density of coordinate space occupation."""
        if len(coords_array) == 0:
            return 0.0

        # Calculate volume of bounding box
        mins = np.min(coords_array, axis=0)
        maxs = np.max(coords_array, axis=0)
        ranges = maxs - mins

        volume = np.prod(ranges)
        if volume == 0:
            return 0.0

        density = len(coords_array) / volume
        return float(density)

    def _calculate_coverage_volume(self, coords_array: np.ndarray) -> float:
        """Calculate volume covered by coordinates."""
        if len(coords_array) == 0:
            return 0.0

        mins = np.min(coords_array, axis=0)
        maxs = np.max(coords_array, axis=0)
        ranges = maxs - mins

        volume = np.prod(ranges)
        return float(volume)

def load_smarts_datasets():
    """Load SMARTS datasets from the public directory."""
    datasets = {}

    # Find the correct base directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(current_dir, '..', '..')  # Go up to gonfanolier root

    dataset_files = {
        'agrafiotis': os.path.join(base_dir, 'public', 'agrafiotis-smarts-tar', 'agrafiotis.smarts'),
        'ahmed': os.path.join(base_dir, 'public', 'ahmed-smarts-tar', 'ahmed.smarts'),
        'daylight': os.path.join(base_dir, 'public', 'daylight-smarts-tar', 'daylight.smarts'),
        'hann': os.path.join(base_dir, 'public', 'hann-smarts-tar', 'hann.smarts'),
        'walters': os.path.join(base_dir, 'public', 'walters-smarts-tar', 'walters.smarts')
    }

    all_smarts = []

    for name, file_path in dataset_files.items():
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    smarts_list = []
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            parts = line.split()
                            if parts:
                                smarts_list.append(parts[0])
                    datasets[name] = smarts_list
                    all_smarts.extend(smarts_list)
                    print(f"✅ Loaded {len(smarts_list)} SMARTS from {name}")
            else:
                print(f"⚠️ File not found: {file_path}")
        except Exception as e:
            print(f"❌ Error loading {name}: {e}")

    if not all_smarts:
        print("⚠️ No SMARTS files found, generating synthetic data...")
        all_smarts = [
            "C1=CC=CC=C1",  # Benzene
            "CC(=O)O",      # Acetic acid
            "CCO",          # Ethanol
            "C1=CC=C(C=C1)O", # Phenol
            "CC(C)O",       # Isopropanol
            "C1=CC=C2C(=C1)C=CC=C2", # Naphthalene
            "CC(=O)N",      # Acetamide
            "C1=CC=C(C=C1)N", # Aniline
        ] * 20  # Replicate for testing

    return all_smarts, datasets

def create_visualizations(results: Dict, output_dir: Path):
    """Create comprehensive visualizations of empty dictionary performance."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use('default')
    sns.set_palette("husl")

    # 1. Synthesis performance metrics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Synthesis times
    if 'synthesis_times' in results and results['synthesis_times']:
        ax1.hist(results['synthesis_times'], bins=20, alpha=0.7, color='skyblue')
        ax1.set_xlabel('Synthesis Time (seconds)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Synthesis Time Distribution')
        ax1.set_yscale('log')

    # Confidence scores
    if 'confidence_scores' in results and results['confidence_scores']:
        ax2.hist(results['confidence_scores'], bins=20, alpha=0.7, color='lightgreen')
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Synthesis Confidence Distribution')

    # Pressure evolution
    if 'pressure_evolution' in results and results['pressure_evolution']:
        ax3.plot(results['pressure_evolution'], color='red', alpha=0.7)
        ax3.set_xlabel('Query Number')
        ax3.set_ylabel('System Pressure')
        ax3.set_title('System Pressure Evolution')

    # Synthesis statistics
    if 'synthesis_statistics' in results:
        stats = results['synthesis_statistics']
        metrics = ['avg_synthesis_time', 'avg_confidence', 'avg_uniqueness', 'avg_feasibility']
        values = [stats.get(metric, 0) for metric in metrics]

        ax4.bar(range(len(metrics)), values, color='orange', alpha=0.7)
        ax4.set_xticks(range(len(metrics)))
        ax4.set_xticklabels([m.replace('avg_', '').replace('_', '\n') for m in metrics])
        ax4.set_ylabel('Value')
        ax4.set_title('Average Synthesis Metrics')

    plt.tight_layout()
    plt.savefig(output_dir / 'empty_dictionary_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Coordinate coverage analysis
    if 'coordinate_coverage' in results and results['coordinate_coverage']:
        coverage = results['coordinate_coverage']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Coordinate ranges
        if 'coordinate_ranges' in coverage:
            ranges = coverage['coordinate_ranges']
            dimensions = list(ranges.keys())
            range_sizes = [ranges[dim]['max'] - ranges[dim]['min'] for dim in dimensions]

            ax1.bar(dimensions, range_sizes, color='purple', alpha=0.7)
            ax1.set_xlabel('Coordinate Dimension')
            ax1.set_ylabel('Range Size')
            ax1.set_title('Coordinate Space Coverage')
            ax1.tick_params(axis='x', rotation=45)

        # Coordinate correlations
        if 'coordinate_correlations' in coverage:
            corr_data = coverage['coordinate_correlations']
            corr_names = list(corr_data.keys())
            corr_values = list(corr_data.values())

            ax2.bar(range(len(corr_names)), corr_values, color='teal', alpha=0.7)
            ax2.set_xticks(range(len(corr_names)))
            ax2.set_xticklabels([name.replace('_', '\n') for name in corr_names], rotation=45)
            ax2.set_ylabel('Correlation')
            ax2.set_title('Coordinate Correlations')
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(output_dir / 'coordinate_coverage_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

def save_results(results: Dict, datasets: Dict, output_dir: Path):
    """Save analysis results to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare results for JSON serialization
    json_results = {
        'empty_dictionary_summary': {
            'total_queries_processed': results.get('synthesis_statistics', {}).get('total_queries_processed', 0),
            'avg_synthesis_time': results.get('synthesis_statistics', {}).get('avg_synthesis_time', 0),
            'synthesis_throughput': results.get('synthesis_statistics', {}).get('synthesis_throughput', 0),
            'avg_confidence': results.get('synthesis_statistics', {}).get('avg_confidence', 0)
        },
        'synthesis_performance': results.get('synthesis_statistics', {}),
        'coordinate_coverage': results.get('coordinate_coverage', {}),
        'dataset_info': {
            name: len(smarts_list) for name, smarts_list in datasets.items()
        }
    }

    # Save results
    with open(output_dir / 'empty_dictionary_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"✅ Results saved to {output_dir}")

def main():
    """Main execution function for empty dictionary analysis."""
    print("🗂️ Empty Dictionary Synthesis Analysis")
    print("=" * 50)

    # Load datasets
    print("\n📊 Loading SMARTS datasets...")
    all_smarts, datasets = load_smarts_datasets()
    print(f"✅ Loaded {len(all_smarts)} total SMARTS patterns")

    # Initialize empty dictionary engine
    print("\n🔧 Initializing Empty Dictionary Engine...")
    engine = EmptyDictionaryEngine()

    # Initialize coordinate navigation
    print("\n🧭 Initializing coordinate navigation...")
    engine.initialize_coordinate_navigator(all_smarts[:100])  # Sample for efficiency

    # Generate test queries
    test_queries = [
        "C1=CC=CC=C1",  # Benzene
        "stable aromatic compound",
        "CC(=O)O",  # Acetic acid
        "simple carboxylic acid",
        "complex heterocyclic molecule",
        "CCO",  # Ethanol
        "pharmaceutical intermediate",
        "reactive alkene compound"
    ]

    # Analyze synthesis performance
    print("\n📈 Analyzing synthesis performance...")
    results = engine.analyze_synthesis_performance(test_queries)

    # Create output directory
    output_dir = Path("gonfanolier/results/hierarchy")

    # Generate visualizations
    print("\n📊 Creating visualizations...")
    create_visualizations(results, output_dir)

    # Save results
    print("\n💾 Saving results...")
    save_results(results, datasets, output_dir)

    # Print summary
    print("\n" + "=" * 50)
    print("🎯 EMPTY DICTIONARY SYNTHESIS SUMMARY")
    print("=" * 50)

    if 'synthesis_statistics' in results:
        stats = results['synthesis_statistics']
        print(f"🔄 Queries Processed: {stats.get('total_queries_processed', 0)}")
        print(f"⚡ Average Synthesis Time: {stats.get('avg_synthesis_time', 0):.4f} seconds")
        print(f"🚀 Synthesis Throughput: {stats.get('synthesis_throughput', 0):.1f} queries/second")
        print(f"🎯 Average Confidence: {stats.get('avg_confidence', 0):.3f}")
        print(f"⭐ Average Uniqueness: {stats.get('avg_uniqueness', 0):.3f}")
        print(f"🔬 Average Feasibility: {stats.get('avg_feasibility', 0):.3f}")
        print(f"📊 Pressure Stability: {stats.get('pressure_stability', 0):.4f}")

    if 'coordinate_coverage' in results:
        coverage = results['coordinate_coverage']
        print(f"📦 Coordinate Space Volume: {coverage.get('coverage_volume', 0):.6f}")
        print(f"🎯 Space Density: {coverage.get('space_density', 0):.3f}")

    print(f"📁 Results saved to: {output_dir}")

    return results

if __name__ == "__main__":
    results = main()
