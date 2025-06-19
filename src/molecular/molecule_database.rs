// Rust integration for database connections
use reqwest;
use serde_json::Value;
use std::collections::{HashMap, BTreeMap};
use ndarray::Array1;
use serde::{Serialize, Deserialize};
use crate::molecular::{OscillatoryQuantumMolecule, HierarchyLevel};
use crate::prediction::{BiologicalActivityPrediction, LongevityPrediction, ToxicityPrediction, DrugLikenessPrediction, MembraneInteractionPrediction, QuantumEfficiencyPrediction};
use crate::similarity::ComprehensiveSimilarityResult;

pub struct DatabaseConnector {
    client: reqwest::Client,
    rate_limiter: RateLimiter,
}

impl DatabaseConnector {
    pub async fn fetch_pubchem_data(&self, compound_id: &str) -> Result<MolecularData, Error> {
        let url = format!("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/JSON", compound_id);
        
        self.rate_limiter.wait().await;
        let response = self.client.get(&url).send().await?;
        let data: Value = response.json().await?;
        
        Ok(MolecularData::from_pubchem(data))
    }
    
    pub async fn enrich_molecular_representation(&self, molecule: &ProbabilisticMolecule) -> EnrichedMolecule {
        // Fetch from multiple databases in parallel
        let (pubchem_data, chembl_data, zinc_data) = tokio::join!(
            self.fetch_pubchem_data(&molecule.id),
            self.fetch_chembl_data(&molecule.id),
            self.fetch_zinc_data(&molecule.id)
        );
        
        EnrichedMolecule::merge(molecule, pubchem_data, chembl_data, zinc_data)
    }
}

/// Advanced quantum molecular database with multi-criteria search
pub struct QuantumMolecularDatabase {
    pub molecules: HashMap<String, OscillatoryQuantumMolecule>,
    pub quantum_indices: QuantumIndices,
    pub oscillatory_indices: OscillatoryIndices,
    pub hierarchy_indices: HierarchyIndices,
    pub property_indices: PropertyIndices,
    pub similarity_cache: HashMap<(String, String), ComprehensiveSimilarityResult>,
}

impl QuantumMolecularDatabase {
    pub fn new() -> Self {
        Self {
            molecules: HashMap::new(),
            quantum_indices: QuantumIndices::new(),
            oscillatory_indices: OscillatoryIndices::new(),
            hierarchy_indices: HierarchyIndices::new(),
            property_indices: PropertyIndices::new(),
            similarity_cache: HashMap::new(),
        }
    }
    
    /// Store molecule with comprehensive quantum-oscillatory analysis
    pub fn store_molecule_with_quantum_analysis(&mut self, molecule: OscillatoryQuantumMolecule) {
        let molecule_id = molecule.molecule_id.clone();
        
        // Update indices
        self.update_quantum_indices(&molecule);
        self.update_oscillatory_indices(&molecule);
        self.update_hierarchy_indices(&molecule);
        self.update_property_indices(&molecule);
        
        // Store molecule
        self.molecules.insert(molecule_id, molecule);
    }
    
    /// Advanced multi-criteria search
    pub fn advanced_multi_criteria_search(&self, criteria: &SearchCriteria) -> Vec<(String, f64)> {
        let mut results = Vec::new();
        
        for (molecule_id, molecule) in &self.molecules {
            let mut score = 0.0;
            let mut total_weight = 0.0;
            
            // Quantum criteria
            if let Some(quantum_criteria) = &criteria.quantum_criteria {
                let quantum_score = self.evaluate_quantum_criteria(molecule, quantum_criteria);
                score += quantum_criteria.weight * quantum_score;
                total_weight += quantum_criteria.weight;
            }
            
            // Oscillatory criteria
            if let Some(oscillatory_criteria) = &criteria.oscillatory_criteria {
                let oscillatory_score = self.evaluate_oscillatory_criteria(molecule, oscillatory_criteria);
                score += oscillatory_criteria.weight * oscillatory_score;
                total_weight += oscillatory_criteria.weight;
            }
            
            // Hierarchy criteria
            if let Some(hierarchy_criteria) = &criteria.hierarchy_criteria {
                let hierarchy_score = self.evaluate_hierarchy_criteria(molecule, hierarchy_criteria);
                score += hierarchy_criteria.weight * hierarchy_score;
                total_weight += hierarchy_criteria.weight;
            }
            
            // Property criteria
            if let Some(property_criteria) = &criteria.property_criteria {
                let property_score = self.evaluate_property_criteria(molecule, property_criteria);
                score += property_criteria.weight * property_score;
                total_weight += property_criteria.weight;
            }
            
            if total_weight > 0.0 {
                score /= total_weight;
                if score >= criteria.min_score_threshold {
                    results.push((molecule_id.clone(), score));
                }
            }
        }
        
        // Sort by score descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(criteria.max_results.unwrap_or(100));
        
        results
    }
    
    /// Search for molecules with specific quantum computational properties
    pub fn search_quantum_computers(&self, criteria: &QuantumSearchCriteria) -> Vec<(String, f64)> {
        let mut results = Vec::new();
        
        for (molecule_id, molecule) in &self.molecules {
            let mut match_score = 0.0;
            let mut criteria_count = 0;
            
            // ENAQT efficiency criteria
            if let Some(min_efficiency) = criteria.min_enaqt_efficiency {
                if molecule.quantum_computer.transport_efficiency >= min_efficiency {
                    match_score += 1.0;
                }
                criteria_count += 1;
            }
            
            // Membrane-like properties
            if let Some(min_membrane_score) = criteria.min_membrane_score {
                if molecule.quantum_computer.membrane_properties.amphipathic_score >= min_membrane_score {
                    match_score += 1.0;
                }
                criteria_count += 1;
            }
            
            // Radical generation (death potential)
            if let Some(max_radical_rate) = criteria.max_radical_generation_rate {
                if molecule.quantum_computer.radical_generation_rate <= max_radical_rate {
                    match_score += 1.0;
                }
                criteria_count += 1;
            }
            
            // Coherence time
            if let Some(min_coherence) = criteria.min_coherence_time {
                if molecule.quantum_computer.coherence_time >= min_coherence {
                    match_score += 1.0;
                }
                criteria_count += 1;
            }
            
            // Tunneling pathways
            if let Some(min_pathways) = criteria.min_tunneling_pathways {
                if molecule.quantum_computer.tunneling_pathways.len() >= min_pathways {
                    match_score += 1.0;
                }
                criteria_count += 1;
            }
            
            if criteria_count > 0 {
                let final_score = match_score / criteria_count as f64;
                if final_score >= criteria.min_match_threshold.unwrap_or(0.5) {
                    results.push((molecule_id.clone(), final_score));
                }
            }
        }
        
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results
    }
    
    /// Search for oscillatory synchronization partners
    pub fn find_synchronization_partners(&self, target_molecule_id: &str, max_freq_diff: f64) -> Vec<(String, f64)> {
        if let Some(target_molecule) = self.molecules.get(target_molecule_id) {
            let target_freq = target_molecule.oscillatory_state.natural_frequency;
            let mut partners = Vec::new();
            
            for (molecule_id, molecule) in &self.molecules {
                if molecule_id != target_molecule_id {
                    let freq_diff = (molecule.oscillatory_state.natural_frequency - target_freq).abs();
                    if freq_diff <= max_freq_diff {
                        let synchronization_potential = (-freq_diff / max_freq_diff).exp();
                        partners.push((molecule_id.clone(), synchronization_potential));
                    }
                }
            }
            
            partners.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            partners
        } else {
            Vec::new()
        }
    }
    
    /// Search for longevity-enhancing molecules
    pub fn search_longevity_enhancers(&self) -> Vec<(String, f64)> {
        let mut enhancers = Vec::new();
        
        for (molecule_id, molecule) in &self.molecules {
            let longevity_score = self.calculate_longevity_enhancement_score(molecule);
            if longevity_score > 0.0 {
                enhancers.push((molecule_id.clone(), longevity_score));
            }
        }
        
        enhancers.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        enhancers
    }
    
    /// Search for death-accelerating molecules (high radical generation)
    pub fn search_death_accelerators(&self) -> Vec<(String, f64)> {
        let mut accelerators = Vec::new();
        
        for (molecule_id, molecule) in &self.molecules {
            let death_score = molecule.quantum_computer.radical_generation_rate * 1e6; // Scale for visibility
            if death_score > 0.1 {
                accelerators.push((molecule_id.clone(), death_score));
            }
        }
        
        accelerators.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        accelerators
    }
    
    /// Update quantum computational indices
    fn update_quantum_indices(&mut self, molecule: &OscillatoryQuantumMolecule) {
        let molecule_id = &molecule.molecule_id;
        
        // ENAQT efficiency index
        let efficiency = molecule.quantum_computer.transport_efficiency;
        self.quantum_indices.enaqt_efficiency_index.entry(self.discretize_efficiency(efficiency))
            .or_insert_with(Vec::new)
            .push(molecule_id.clone());
        
        // Membrane properties index
        let membrane_score = molecule.quantum_computer.membrane_properties.amphipathic_score;
        self.quantum_indices.membrane_score_index.entry(self.discretize_score(membrane_score))
            .or_insert_with(Vec::new)
            .push(molecule_id.clone());
        
        // Radical generation index
        let radical_rate = molecule.quantum_computer.radical_generation_rate;
        self.quantum_indices.radical_generation_index.entry(self.discretize_radical_rate(radical_rate))
            .or_insert_with(Vec::new)
            .push(molecule_id.clone());
        
        // Coherence time index
        let coherence = molecule.quantum_computer.coherence_time;
        self.quantum_indices.coherence_time_index.entry(self.discretize_coherence_time(coherence))
            .or_insert_with(Vec::new)
            .push(molecule_id.clone());
    }
    
    /// Update oscillatory indices
    fn update_oscillatory_indices(&mut self, molecule: &OscillatoryQuantumMolecule) {
        let molecule_id = &molecule.molecule_id;
        
        // Frequency index
        let frequency = molecule.oscillatory_state.natural_frequency;
        self.oscillatory_indices.frequency_index.entry(self.discretize_frequency(frequency))
            .or_insert_with(Vec::new)
            .push(molecule_id.clone());
        
        // Synchronization potential index
        let sync_potential = molecule.synchronization_parameters.phase_locking_strength;
        self.oscillatory_indices.synchronization_index.entry(self.discretize_score(sync_potential))
            .or_insert_with(Vec::new)
            .push(molecule_id.clone());
    }
    
    /// Update hierarchy indices
    fn update_hierarchy_indices(&mut self, molecule: &OscillatoryQuantumMolecule) {
        let molecule_id = &molecule.molecule_id;
        
        // Hierarchy levels index
        for level in molecule.hierarchy_representations.keys() {
            self.hierarchy_indices.level_index.entry(*level)
                .or_insert_with(Vec::new)
                .push(molecule_id.clone());
        }
        
        // Multi-scale coupling index
        let coupling_score = self.calculate_multi_scale_coupling(molecule);
        self.hierarchy_indices.coupling_index.entry(self.discretize_score(coupling_score))
            .or_insert_with(Vec::new)
            .push(molecule_id.clone());
    }
    
    /// Update property indices
    fn update_property_indices(&mut self, molecule: &OscillatoryQuantumMolecule) {
        let molecule_id = &molecule.molecule_id;
        
        // Biological activity index
        let activity_score = molecule.property_predictions.biological_activity.activity_score;
        self.property_indices.biological_activity_index.entry(self.discretize_score(activity_score))
            .or_insert_with(Vec::new)
            .push(molecule_id.clone());
        
        // Longevity impact index
        let longevity_factor = molecule.property_predictions.longevity_impact.longevity_factor;
        self.property_indices.longevity_index.entry(self.discretize_longevity_factor(longevity_factor))
            .or_insert_with(Vec::new)
            .push(molecule_id.clone());
        
        // Drug-likeness index
        let drug_score = molecule.property_predictions.drug_likeness.drug_likeness_score;
        self.property_indices.drug_likeness_index.entry(self.discretize_score(drug_score))
            .or_insert_with(Vec::new)
            .push(molecule_id.clone());
    }
    
    /// Evaluate quantum criteria
    fn evaluate_quantum_criteria(&self, molecule: &OscillatoryQuantumMolecule, criteria: &QuantumSearchCriteria) -> f64 {
        let mut score = 0.0;
        let mut count = 0;
        
        if let Some(min_efficiency) = criteria.min_enaqt_efficiency {
            score += if molecule.quantum_computer.transport_efficiency >= min_efficiency { 1.0 } else { 0.0 };
            count += 1;
        }
        
        if let Some(min_membrane) = criteria.min_membrane_score {
            score += if molecule.quantum_computer.membrane_properties.amphipathic_score >= min_membrane { 1.0 } else { 0.0 };
            count += 1;
        }
        
        if let Some(max_radical) = criteria.max_radical_generation_rate {
            score += if molecule.quantum_computer.radical_generation_rate <= max_radical { 1.0 } else { 0.0 };
            count += 1;
        }
        
        if count > 0 { score / count as f64 } else { 0.0 }
    }
    
    /// Evaluate oscillatory criteria
    fn evaluate_oscillatory_criteria(&self, molecule: &OscillatoryQuantumMolecule, criteria: &OscillatorySearchCriteria) -> f64 {
        let mut score = 0.0;
        let mut count = 0;
        
        if let Some((min_freq, max_freq)) = criteria.frequency_range {
            let freq = molecule.oscillatory_state.natural_frequency;
            score += if freq >= min_freq && freq <= max_freq { 1.0 } else { 0.0 };
            count += 1;
        }
        
        if let Some(min_sync) = criteria.min_synchronization_potential {
            score += if molecule.synchronization_parameters.phase_locking_strength >= min_sync { 1.0 } else { 0.0 };
            count += 1;
        }
        
        if count > 0 { score / count as f64 } else { 0.0 }
    }
    
    /// Evaluate hierarchy criteria
    fn evaluate_hierarchy_criteria(&self, molecule: &OscillatoryQuantumMolecule, criteria: &HierarchySearchCriteria) -> f64 {
        let mut score = 0.0;
        let mut count = 0;
        
        if let Some(required_levels) = &criteria.required_hierarchy_levels {
            let has_all_levels = required_levels.iter()
                .all(|level| molecule.hierarchy_representations.contains_key(level));
            score += if has_all_levels { 1.0 } else { 0.0 };
            count += 1;
        }
        
        if let Some(min_coupling) = criteria.min_cross_scale_coupling {
            let coupling_score = self.calculate_multi_scale_coupling(molecule);
            score += if coupling_score >= min_coupling { 1.0 } else { 0.0 };
            count += 1;
        }
        
        if count > 0 { score / count as f64 } else { 0.0 }
    }
    
    /// Evaluate property criteria
    fn evaluate_property_criteria(&self, molecule: &OscillatoryQuantumMolecule, criteria: &PropertySearchCriteria) -> f64 {
        let mut score = 0.0;
        let mut count = 0;
        
        if let Some(min_activity) = criteria.min_biological_activity {
            score += if molecule.property_predictions.biological_activity.activity_score >= min_activity { 1.0 } else { 0.0 };
            count += 1;
        }
        
        if let Some(min_longevity) = criteria.min_longevity_factor {
            score += if molecule.property_predictions.longevity_impact.longevity_factor >= min_longevity { 1.0 } else { 0.0 };
            count += 1;
        }
        
        if let Some(max_toxicity) = criteria.max_toxicity_score {
            score += if molecule.property_predictions.toxicity_prediction.toxicity_score <= max_toxicity { 1.0 } else { 0.0 };
            count += 1;
        }
        
        if count > 0 { score / count as f64 } else { 0.0 }
    }
    
    /// Calculate longevity enhancement score
    fn calculate_longevity_enhancement_score(&self, molecule: &OscillatoryQuantumMolecule) -> f64 {
        let longevity_factor = molecule.property_predictions.longevity_impact.longevity_factor;
        let escape_mechanisms = molecule.property_predictions.longevity_impact.escape_mechanisms;
        let quantum_burden = molecule.property_predictions.longevity_impact.quantum_burden;
        
        // Positive longevity factor with high escape mechanisms and low quantum burden
        if longevity_factor > 0.0 && escape_mechanisms > 0.3 && quantum_burden < 0.1 {
            longevity_factor * escape_mechanisms * (1.0 - quantum_burden)
        } else {
            0.0
        }
    }
    
    /// Calculate multi-scale coupling strength
    fn calculate_multi_scale_coupling(&self, molecule: &OscillatoryQuantumMolecule) -> f64 {
        if molecule.hierarchy_representations.len() < 2 {
            return 0.0;
        }
        
        let total_coupling: f64 = molecule.hierarchy_representations.values()
            .map(|level| level.coupling_to_adjacent_levels.iter().sum::<f64>())
            .sum();
        
        total_coupling / (molecule.hierarchy_representations.len() as f64 * 2.0)
    }
    
    /// Discretization helper functions
    fn discretize_efficiency(&self, efficiency: f64) -> u8 {
        (efficiency * 10.0) as u8
    }
    
    fn discretize_score(&self, score: f64) -> u8 {
        (score * 10.0) as u8
    }
    
    fn discretize_radical_rate(&self, rate: f64) -> u8 {
        if rate == 0.0 { 0 }
        else if rate < 1e-9 { 1 }
        else if rate < 1e-8 { 2 }
        else if rate < 1e-7 { 3 }
        else if rate < 1e-6 { 4 }
        else { 5 }
    }
    
    fn discretize_coherence_time(&self, time: f64) -> u8 {
        if time < 1e-15 { 0 }
        else if time < 1e-14 { 1 }
        else if time < 1e-13 { 2 }
        else if time < 1e-12 { 3 }
        else if time < 1e-11 { 4 }
        else { 5 }
    }
    
    fn discretize_frequency(&self, freq: f64) -> u8 {
        if freq < 1e9 { 0 }
        else if freq < 1e10 { 1 }
        else if freq < 1e11 { 2 }
        else if freq < 1e12 { 3 }
        else if freq < 1e13 { 4 }
        else { 5 }
    }
    
    fn discretize_longevity_factor(&self, factor: f64) -> i8 {
        if factor < -0.5 { -5 }
        else if factor < -0.1 { -1 }
        else if factor < 0.1 { 0 }
        else if factor < 0.5 { 1 }
        else { 5 }
    }
}

/// Database indices for efficient searching
#[derive(Debug)]
pub struct QuantumIndices {
    pub enaqt_efficiency_index: HashMap<u8, Vec<String>>,
    pub membrane_score_index: HashMap<u8, Vec<String>>,
    pub radical_generation_index: HashMap<u8, Vec<String>>,
    pub coherence_time_index: HashMap<u8, Vec<String>>,
}

impl QuantumIndices {
    pub fn new() -> Self {
        Self {
            enaqt_efficiency_index: HashMap::new(),
            membrane_score_index: HashMap::new(),
            radical_generation_index: HashMap::new(),
            coherence_time_index: HashMap::new(),
        }
    }
}

#[derive(Debug)]
pub struct OscillatoryIndices {
    pub frequency_index: HashMap<u8, Vec<String>>,
    pub synchronization_index: HashMap<u8, Vec<String>>,
}

impl OscillatoryIndices {
    pub fn new() -> Self {
        Self {
            frequency_index: HashMap::new(),
            synchronization_index: HashMap::new(),
        }
    }
}

#[derive(Debug)]
pub struct HierarchyIndices {
    pub level_index: HashMap<u8, Vec<String>>,
    pub coupling_index: HashMap<u8, Vec<String>>,
}

impl HierarchyIndices {
    pub fn new() -> Self {
        Self {
            level_index: HashMap::new(),
            coupling_index: HashMap::new(),
        }
    }
}

#[derive(Debug)]
pub struct PropertyIndices {
    pub biological_activity_index: HashMap<u8, Vec<String>>,
    pub longevity_index: HashMap<i8, Vec<String>>,
    pub drug_likeness_index: HashMap<u8, Vec<String>>,
}

impl PropertyIndices {
    pub fn new() -> Self {
        Self {
            biological_activity_index: HashMap::new(),
            longevity_index: HashMap::new(),
            drug_likeness_index: HashMap::new(),
        }
    }
}

/// Search criteria structures
#[derive(Clone, Debug)]
pub struct SearchCriteria {
    pub quantum_criteria: Option<QuantumSearchCriteria>,
    pub oscillatory_criteria: Option<OscillatorySearchCriteria>,
    pub hierarchy_criteria: Option<HierarchySearchCriteria>,
    pub property_criteria: Option<PropertySearchCriteria>,
    pub min_score_threshold: f64,
    pub max_results: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct QuantumSearchCriteria {
    pub min_enaqt_efficiency: Option<f64>,
    pub min_membrane_score: Option<f64>,
    pub max_radical_generation_rate: Option<f64>,
    pub min_coherence_time: Option<f64>,
    pub min_tunneling_pathways: Option<usize>,
    pub min_match_threshold: Option<f64>,
    pub weight: f64,
}

#[derive(Clone, Debug)]
pub struct OscillatorySearchCriteria {
    pub frequency_range: Option<(f64, f64)>,
    pub min_synchronization_potential: Option<f64>,
    pub min_amplitude: Option<f64>,
    pub weight: f64,
}

#[derive(Clone, Debug)]
pub struct HierarchySearchCriteria {
    pub required_hierarchy_levels: Option<Vec<u8>>,
    pub min_cross_scale_coupling: Option<f64>,
    pub required_emergence_patterns: Option<Vec<String>>,
    pub weight: f64,
}

#[derive(Clone, Debug)]
pub struct PropertySearchCriteria {
    pub min_biological_activity: Option<f64>,
    pub min_longevity_factor: Option<f64>,
    pub max_toxicity_score: Option<f64>,
    pub min_drug_likeness: Option<f64>,
    pub weight: f64,
}
