use std::collections::{HashMap, HashSet};
use crate::error::{BorgiaResult, BorgiaError};
use crate::algorithms::morgan::{MolecularGraph, MorganNode};

/// SMILES (Simplified Molecular Input Line Entry System) representation
/// Enhanced with probabilistic features and advanced parsing capabilities
#[derive(Debug, Clone)]
pub struct SMILESRepresentation {
    /// Raw SMILES string
    pub smiles: String,
    /// Parsed molecular graph
    pub graph: Option<MolecularGraph>,
    /// Atom properties
    pub atoms: Vec<SMILESAtom>,
    /// Bond properties
    pub bonds: Vec<SMILESBond>,
    /// Ring information
    pub rings: Vec<SMILESRing>,
    /// Stereochemistry information
    pub stereochemistry: Vec<StereoCenter>,
}

/// SMILES atom representation
#[derive(Debug, Clone)]
pub struct SMILESAtom {
    pub id: usize,
    pub element: String,
    pub atomic_number: u8,
    pub charge: i8,
    pub hydrogen_count: u8,
    pub aromatic: bool,
    pub chirality: Option<Chirality>,
    pub isotope: Option<u16>,
}

/// SMILES bond representation
#[derive(Debug, Clone)]
pub struct SMILESBond {
    pub from_atom: usize,
    pub to_atom: usize,
    pub bond_type: BondType,
    pub stereochemistry: Option<BondStereo>,
}

/// SMILES ring information
#[derive(Debug, Clone)]
pub struct SMILESRing {
    pub id: u8,
    pub atoms: Vec<usize>,
    pub aromatic: bool,
}

/// Bond types in SMILES
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BondType {
    Single,
    Double,
    Triple,
    Aromatic,
    Up,      // Stereochemical up
    Down,    // Stereochemical down
}

impl BondType {
    pub fn to_order(&self) -> f64 {
        match self {
            BondType::Single => 1.0,
            BondType::Double => 2.0,
            BondType::Triple => 3.0,
            BondType::Aromatic => 1.5,
            BondType::Up => 1.0,
            BondType::Down => 1.0,
        }
    }
}

/// Chirality specification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Chirality {
    Clockwise,
    CounterClockwise,
    Unspecified,
}

/// Bond stereochemistry
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BondStereo {
    Up,
    Down,
    Either,
}

/// Stereochemical center
#[derive(Debug, Clone)]
pub struct StereoCenter {
    pub atom_id: usize,
    pub chirality: Chirality,
    pub neighbors: Vec<usize>,
}

/// SMILES parser for converting SMILES strings to molecular graphs
#[derive(Debug)]
pub struct SMILESParser {
    /// Current position in SMILES string
    position: usize,
    /// SMILES string being parsed
    smiles: Vec<char>,
    /// Atom counter
    atom_counter: usize,
    /// Ring closure tracking
    ring_closures: HashMap<u8, usize>,
    /// Current atom stack for branching
    atom_stack: Vec<usize>,
}

impl SMILESParser {
    pub fn new() -> Self {
        Self {
            position: 0,
            smiles: Vec::new(),
            atom_counter: 0,
            ring_closures: HashMap::new(),
            atom_stack: Vec::new(),
        }
    }

    /// Parse SMILES string into SMILESRepresentation
    pub fn parse(&mut self, smiles: &str) -> BorgiaResult<SMILESRepresentation> {
        self.smiles = smiles.chars().collect();
        self.position = 0;
        self.atom_counter = 0;
        self.ring_closures.clear();
        self.atom_stack.clear();

        let mut representation = SMILESRepresentation {
            smiles: smiles.to_string(),
            graph: None,
            atoms: Vec::new(),
            bonds: Vec::new(),
            rings: Vec::new(),
            stereochemistry: Vec::new(),
        };

        // Parse the SMILES string
        self.parse_molecule(&mut representation)?;

        // Build molecular graph
        let graph = self.build_molecular_graph(&representation)?;
        representation.graph = Some(graph);

        // Detect rings
        self.detect_rings(&mut representation)?;

        // Analyze stereochemistry
        self.analyze_stereochemistry(&mut representation)?;

        Ok(representation)
    }

    /// Parse the main molecule structure
    fn parse_molecule(&mut self, representation: &mut SMILESRepresentation) -> BorgiaResult<()> {
        let mut last_atom = None;
        let mut pending_bond = BondType::Single;

        while self.position < self.smiles.len() {
            let ch = self.smiles[self.position];

            match ch {
                // Atoms
                'C' | 'N' | 'O' | 'S' | 'P' | 'F' | 'Cl' | 'Br' | 'I' | 'H' => {
                    let atom = self.parse_atom()?;
                    representation.atoms.push(atom.clone());
                    
                    if let Some(prev_atom) = last_atom {
                        let bond = SMILESBond {
                            from_atom: prev_atom,
                            to_atom: atom.id,
                            bond_type: pending_bond,
                            stereochemistry: None,
                        };
                        representation.bonds.push(bond);
                    }
                    
                    last_atom = Some(atom.id);
                    pending_bond = BondType::Single;
                }
                
                // Aromatic atoms
                'c' | 'n' | 'o' | 's' | 'p' => {
                    let atom = self.parse_aromatic_atom()?;
                    representation.atoms.push(atom.clone());
                    
                    if let Some(prev_atom) = last_atom {
                        let bond = SMILESBond {
                            from_atom: prev_atom,
                            to_atom: atom.id,
                            bond_type: BondType::Aromatic,
                            stereochemistry: None,
                        };
                        representation.bonds.push(bond);
                    }
                    
                    last_atom = Some(atom.id);
                    pending_bond = BondType::Single;
                }

                // Bonds
                '=' => {
                    pending_bond = BondType::Double;
                    self.position += 1;
                }
                '#' => {
                    pending_bond = BondType::Triple;
                    self.position += 1;
                }
                '/' => {
                    pending_bond = BondType::Up;
                    self.position += 1;
                }
                '\\' => {
                    pending_bond = BondType::Down;
                    self.position += 1;
                }

                // Branching
                '(' => {
                    if let Some(atom_id) = last_atom {
                        self.atom_stack.push(atom_id);
                    }
                    self.position += 1;
                }
                ')' => {
                    if let Some(atom_id) = self.atom_stack.pop() {
                        last_atom = Some(atom_id);
                    }
                    self.position += 1;
                }

                // Ring closures
                '0'..='9' => {
                    let ring_id = ch.to_digit(10).unwrap() as u8;
                    self.handle_ring_closure(ring_id, last_atom, pending_bond, representation)?;
                    pending_bond = BondType::Single;
                    self.position += 1;
                }

                // Brackets for complex atoms
                '[' => {
                    let atom = self.parse_bracketed_atom()?;
                    representation.atoms.push(atom.clone());
                    
                    if let Some(prev_atom) = last_atom {
                        let bond = SMILESBond {
                            from_atom: prev_atom,
                            to_atom: atom.id,
                            bond_type: pending_bond,
                            stereochemistry: None,
                        };
                        representation.bonds.push(bond);
                    }
                    
                    last_atom = Some(atom.id);
                    pending_bond = BondType::Single;
                }

                // Skip whitespace
                ' ' | '\t' | '\n' | '\r' => {
                    self.position += 1;
                }

                // Unknown character
                _ => {
                    return Err(BorgiaError::ParseError(
                        format!("Unknown character '{}' at position {}", ch, self.position)
                    ));
                }
            }
        }

        Ok(())
    }

    /// Parse a simple atom
    fn parse_atom(&mut self) -> BorgiaResult<SMILESAtom> {
        let ch = self.smiles[self.position];
        self.position += 1;

        let (element, atomic_number) = match ch {
            'C' => ("C".to_string(), 6),
            'N' => ("N".to_string(), 7),
            'O' => ("O".to_string(), 8),
            'S' => ("S".to_string(), 16),
            'P' => ("P".to_string(), 15),
            'F' => ("F".to_string(), 9),
            'H' => ("H".to_string(), 1),
            _ => {
                // Handle two-letter elements
                if self.position < self.smiles.len() {
                    let next_ch = self.smiles[self.position];
                    let two_letter = format!("{}{}", ch, next_ch);
                    match two_letter.as_str() {
                        "Cl" => {
                            self.position += 1;
                            ("Cl".to_string(), 17)
                        }
                        "Br" => {
                            self.position += 1;
                            ("Br".to_string(), 35)
                        }
                        _ => return Err(BorgiaError::ParseError(
                            format!("Unknown element '{}'", ch)
                        )),
                    }
                } else {
                    return Err(BorgiaError::ParseError(
                        format!("Unknown element '{}'", ch)
                    ));
                }
            }
        };

        let atom = SMILESAtom {
            id: self.atom_counter,
            element,
            atomic_number,
            charge: 0,
            hydrogen_count: 0,
            aromatic: false,
            chirality: None,
            isotope: None,
        };

        self.atom_counter += 1;
        Ok(atom)
    }

    /// Parse an aromatic atom
    fn parse_aromatic_atom(&mut self) -> BorgiaResult<SMILESAtom> {
        let ch = self.smiles[self.position];
        self.position += 1;

        let (element, atomic_number) = match ch {
            'c' => ("C".to_string(), 6),
            'n' => ("N".to_string(), 7),
            'o' => ("O".to_string(), 8),
            's' => ("S".to_string(), 16),
            'p' => ("P".to_string(), 15),
            _ => return Err(BorgiaError::ParseError(
                format!("Unknown aromatic element '{}'", ch)
            )),
        };

        let atom = SMILESAtom {
            id: self.atom_counter,
            element,
            atomic_number,
            charge: 0,
            hydrogen_count: 0,
            aromatic: true,
            chirality: None,
            isotope: None,
        };

        self.atom_counter += 1;
        Ok(atom)
    }

    /// Parse a bracketed atom with properties
    fn parse_bracketed_atom(&mut self) -> BorgiaResult<SMILESAtom> {
        self.position += 1; // Skip '['

        let mut isotope = None;
        let mut element = String::new();
        let mut aromatic = false;
        let mut chirality = None;
        let mut hydrogen_count = 0;
        let mut charge = 0;

        // Parse isotope
        while self.position < self.smiles.len() && self.smiles[self.position].is_ascii_digit() {
            if isotope.is_none() {
                isotope = Some(0);
            }
            isotope = Some(isotope.unwrap() * 10 + self.smiles[self.position].to_digit(10).unwrap() as u16);
            self.position += 1;
        }

        // Parse element
        if self.position < self.smiles.len() {
            let ch = self.smiles[self.position];
            if ch.is_ascii_uppercase() {
                element.push(ch);
                self.position += 1;
                
                // Check for second character
                if self.position < self.smiles.len() && self.smiles[self.position].is_ascii_lowercase() {
                    element.push(self.smiles[self.position]);
                    self.position += 1;
                }
            } else if ch.is_ascii_lowercase() {
                element.push(ch.to_ascii_uppercase());
                aromatic = true;
                self.position += 1;
            }
        }

        // Parse chirality
        if self.position < self.smiles.len() && self.smiles[self.position] == '@' {
            self.position += 1;
            if self.position < self.smiles.len() && self.smiles[self.position] == '@' {
                chirality = Some(Chirality::CounterClockwise);
                self.position += 1;
            } else {
                chirality = Some(Chirality::Clockwise);
            }
        }

        // Parse hydrogen count
        if self.position < self.smiles.len() && self.smiles[self.position] == 'H' {
            self.position += 1;
            hydrogen_count = 1;
            
            if self.position < self.smiles.len() && self.smiles[self.position].is_ascii_digit() {
                hydrogen_count = self.smiles[self.position].to_digit(10).unwrap() as u8;
                self.position += 1;
            }
        }

        // Parse charge
        while self.position < self.smiles.len() {
            match self.smiles[self.position] {
                '+' => {
                    charge += 1;
                    self.position += 1;
                }
                '-' => {
                    charge -= 1;
                    self.position += 1;
                }
                ']' => {
                    self.position += 1;
                    break;
                }
                _ => {
                    self.position += 1;
                }
            }
        }

        let atomic_number = self.get_atomic_number(&element)?;

        let atom = SMILESAtom {
            id: self.atom_counter,
            element,
            atomic_number,
            charge,
            hydrogen_count,
            aromatic,
            chirality,
            isotope,
        };

        self.atom_counter += 1;
        Ok(atom)
    }

    /// Handle ring closure
    fn handle_ring_closure(
        &mut self,
        ring_id: u8,
        current_atom: Option<usize>,
        bond_type: BondType,
        representation: &mut SMILESRepresentation,
    ) -> BorgiaResult<()> {
        if let Some(current) = current_atom {
            if let Some(&previous_atom) = self.ring_closures.get(&ring_id) {
                // Close the ring
                let bond = SMILESBond {
                    from_atom: previous_atom,
                    to_atom: current,
                    bond_type,
                    stereochemistry: None,
                };
                representation.bonds.push(bond);
                self.ring_closures.remove(&ring_id);
            } else {
                // Open the ring
                self.ring_closures.insert(ring_id, current);
            }
        }
        Ok(())
    }

    /// Get atomic number from element symbol
    fn get_atomic_number(&self, element: &str) -> BorgiaResult<u8> {
        match element {
            "H" => Ok(1),
            "He" => Ok(2),
            "Li" => Ok(3),
            "Be" => Ok(4),
            "B" => Ok(5),
            "C" => Ok(6),
            "N" => Ok(7),
            "O" => Ok(8),
            "F" => Ok(9),
            "Ne" => Ok(10),
            "Na" => Ok(11),
            "Mg" => Ok(12),
            "Al" => Ok(13),
            "Si" => Ok(14),
            "P" => Ok(15),
            "S" => Ok(16),
            "Cl" => Ok(17),
            "Ar" => Ok(18),
            "K" => Ok(19),
            "Ca" => Ok(20),
            "Br" => Ok(35),
            "I" => Ok(53),
            _ => Err(BorgiaError::ParseError(
                format!("Unknown element: {}", element)
            )),
        }
    }

    /// Build molecular graph from parsed representation
    fn build_molecular_graph(&self, representation: &SMILESRepresentation) -> BorgiaResult<MolecularGraph> {
        let mut graph = MolecularGraph::new();

        // Add atoms as nodes
        for atom in &representation.atoms {
            graph.add_node(atom.id, atom.atomic_number);
        }

        // Add bonds as edges
        for bond in &representation.bonds {
            graph.add_edge(bond.from_atom, bond.to_atom, bond.bond_type.to_order());
        }

        Ok(graph)
    }

    /// Detect ring structures
    fn detect_rings(&self, representation: &mut SMILESRepresentation) -> BorgiaResult<()> {
        // This is a simplified ring detection
        // In practice, you'd want more sophisticated algorithms
        
        let mut visited = HashSet::new();
        let mut ring_counter = 0u8;

        for bond in &representation.bonds {
            if !visited.contains(&(bond.from_atom, bond.to_atom)) {
                if let Some(ring_atoms) = self.find_ring_containing_bond(bond, representation) {
                    let ring = SMILESRing {
                        id: ring_counter,
                        atoms: ring_atoms.clone(),
                        aromatic: self.is_ring_aromatic(&ring_atoms, representation),
                    };
                    representation.rings.push(ring);
                    
                    // Mark all bonds in this ring as visited
                    for i in 0..ring_atoms.len() {
                        let from = ring_atoms[i];
                        let to = ring_atoms[(i + 1) % ring_atoms.len()];
                        visited.insert((from, to));
                        visited.insert((to, from));
                    }
                    
                    ring_counter += 1;
                }
            }
        }

        Ok(())
    }

    /// Find ring containing a specific bond
    fn find_ring_containing_bond(
        &self,
        _bond: &SMILESBond,
        _representation: &SMILESRepresentation,
    ) -> Option<Vec<usize>> {
        // Simplified implementation
        // In practice, you'd implement proper cycle detection
        None
    }

    /// Check if a ring is aromatic
    fn is_ring_aromatic(&self, ring_atoms: &[usize], representation: &SMILESRepresentation) -> bool {
        ring_atoms.iter().all(|&atom_id| {
            representation.atoms.iter()
                .find(|atom| atom.id == atom_id)
                .map(|atom| atom.aromatic)
                .unwrap_or(false)
        })
    }

    /// Analyze stereochemistry
    fn analyze_stereochemistry(&self, representation: &mut SMILESRepresentation) -> BorgiaResult<()> {
        for atom in &representation.atoms {
            if let Some(chirality) = atom.chirality {
                let neighbors = self.get_atom_neighbors(atom.id, representation);
                let stereo_center = StereoCenter {
                    atom_id: atom.id,
                    chirality,
                    neighbors,
                };
                representation.stereochemistry.push(stereo_center);
            }
        }
        Ok(())
    }

    /// Get neighbors of an atom
    fn get_atom_neighbors(&self, atom_id: usize, representation: &SMILESRepresentation) -> Vec<usize> {
        let mut neighbors = Vec::new();
        for bond in &representation.bonds {
            if bond.from_atom == atom_id {
                neighbors.push(bond.to_atom);
            } else if bond.to_atom == atom_id {
                neighbors.push(bond.from_atom);
            }
        }
        neighbors
    }
}

/// SMILES generator for converting molecular graphs to SMILES strings
#[derive(Debug)]
pub struct SMILESGenerator {
    /// Options for SMILES generation
    pub canonical: bool,
    pub include_stereochemistry: bool,
    pub include_aromaticity: bool,
}

impl Default for SMILESGenerator {
    fn default() -> Self {
        Self {
            canonical: true,
            include_stereochemistry: true,
            include_aromaticity: true,
        }
    }
}

impl SMILESGenerator {
    pub fn new(canonical: bool, include_stereochemistry: bool, include_aromaticity: bool) -> Self {
        Self {
            canonical,
            include_stereochemistry,
            include_aromaticity,
        }
    }

    /// Generate SMILES string from molecular graph
    pub fn generate(&self, graph: &MolecularGraph) -> BorgiaResult<String> {
        // This is a simplified implementation
        // In practice, you'd implement proper SMILES generation with:
        // - Canonical ordering
        // - Ring closure detection
        // - Stereochemistry preservation
        // - Aromaticity detection
        
        let mut smiles = String::new();
        let mut visited = HashSet::new();
        
        // Start with first node
        if let Some((&start_node, _)) = graph.nodes.iter().next() {
            self.generate_from_node(start_node, &mut smiles, &mut visited, graph, None)?;
        }
        
        Ok(smiles)
    }

    /// Generate SMILES from a specific node
    fn generate_from_node(
        &self,
        node_id: usize,
        smiles: &mut String,
        visited: &mut HashSet<usize>,
        graph: &MolecularGraph,
        parent: Option<usize>,
    ) -> BorgiaResult<()> {
        if visited.contains(&node_id) {
            return Ok(());
        }
        
        visited.insert(node_id);
        
        if let Some(node) = graph.nodes.get(&node_id) {
            // Add atom symbol
            let symbol = self.get_element_symbol(node.atomic_number);
            smiles.push_str(&symbol);
            
            // Process neighbors
            let mut neighbors: Vec<_> = node.neighbors.iter()
                .filter(|&&neighbor| Some(neighbor) != parent)
                .collect();
            
            // Sort neighbors for canonical SMILES
            if self.canonical {
                neighbors.sort();
            }
            
            for (i, &&neighbor) in neighbors.iter().enumerate() {
                if !visited.contains(&neighbor) {
                    if i > 0 {
                        smiles.push('(');
                    }
                    
                    // Add bond symbol if needed
                    if let Some(bond_order) = self.get_bond_order(node_id, neighbor, graph) {
                        match bond_order {
                            2.0 => smiles.push('='),
                            3.0 => smiles.push('#'),
                            1.5 => {}, // Aromatic, no symbol needed
                            _ => {}, // Single bond, no symbol needed
                        }
                    }
                    
                    self.generate_from_node(neighbor, smiles, visited, graph, Some(node_id))?;
                    
                    if i > 0 {
                        smiles.push(')');
                    }
                }
            }
        }
        
        Ok(())
    }

    /// Get element symbol from atomic number
    fn get_element_symbol(&self, atomic_number: u8) -> String {
        match atomic_number {
            1 => "H".to_string(),
            6 => "C".to_string(),
            7 => "N".to_string(),
            8 => "O".to_string(),
            9 => "F".to_string(),
            15 => "P".to_string(),
            16 => "S".to_string(),
            17 => "Cl".to_string(),
            35 => "Br".to_string(),
            53 => "I".to_string(),
            _ => format!("[{}]", atomic_number),
        }
    }

    /// Get bond order between two atoms
    fn get_bond_order(&self, from: usize, to: usize, graph: &MolecularGraph) -> Option<f64> {
        for (edge_from, edge_to, bond_order) in &graph.edges {
            if (*edge_from == from && *edge_to == to) || (*edge_from == to && *edge_to == from) {
                return Some(*bond_order);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_smiles_parsing() {
        let mut parser = SMILESParser::new();
        let result = parser.parse("CCO");
        
        assert!(result.is_ok());
        let representation = result.unwrap();
        assert_eq!(representation.atoms.len(), 3);
        assert_eq!(representation.bonds.len(), 2);
    }

    #[test]
    fn test_aromatic_smiles_parsing() {
        let mut parser = SMILESParser::new();
        let result = parser.parse("c1ccccc1");
        
        assert!(result.is_ok());
        let representation = result.unwrap();
        assert_eq!(representation.atoms.len(), 6);
        assert!(representation.atoms.iter().all(|atom| atom.aromatic));
    }

    #[test]
    fn test_bracketed_atom_parsing() {
        let mut parser = SMILESParser::new();
        let result = parser.parse("[CH3+]");
        
        assert!(result.is_ok());
        let representation = result.unwrap();
        assert_eq!(representation.atoms.len(), 1);
        assert_eq!(representation.atoms[0].charge, 1);
        assert_eq!(representation.atoms[0].hydrogen_count, 3);
    }

    #[test]
    fn test_smiles_generation() {
        let mut graph = MolecularGraph::new();
        graph.add_node(0, 6); // C
        graph.add_node(1, 6); // C
        graph.add_node(2, 8); // O
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(1, 2, 1.0);

        let generator = SMILESGenerator::default();
        let result = generator.generate(&graph);
        
        assert!(result.is_ok());
        let smiles = result.unwrap();
        assert!(!smiles.is_empty());
    }
}
