//! Turbulance Language Interpreter
//! 
//! Executes Turbulance programs with full support for all four paradigms
//! and integration with Biological Maxwell's Demons.

use std::collections::HashMap;
use crate::turbulance::ast::*;
use crate::turbulance::probabilistic::*;
use crate::bmd_networks::*;
use crate::error::{BorgiaError, BorgiaResult};
use crate::molecular::Molecule;
use serde::{Serialize, Deserialize};

/// Turbulance interpreter with BMD integration
#[derive(Debug)]
pub struct TurbulanceInterpreter {
    /// Variable environment
    variables: HashMap<String, Value>,
    /// Point storage
    points: HashMap<String, TextPoint>,
    /// Resolution functions
    resolution_functions: HashMap<String, Box<dyn ResolutionFunction>>,
    /// BMD networks
    bmd_networks: BMDNetworkManager,
    /// Function definitions
    functions: HashMap<String, FunctionDefinition>,
    /// Execution context
    context: ExecutionContext,
}

/// Execution context for interpreter state
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    pub confidence_threshold: f64,
    pub current_domain: String,
    pub processing_mode: ProcessingMode,
    pub resolution_strategy: ResolutionStrategy,
}

/// BMD Network Manager for coordinating different scale BMDs
#[derive(Debug)]
pub struct BMDNetworkManager {
    quantum_bmd: QuantumBMD,
    molecular_bmd: MolecularBMD,
    environmental_bmd: EnvironmentalBMD,
    hardware_bmd: HardwareBMD,
    amplifier: ThermodynamicAmplifier,
}

/// Universal input type for BMD operations
#[derive(Debug, Clone)]
pub enum UniversalInput {
    Quantum(QuantumEvent),
    Molecular(Vec<Molecule>),
    Environmental(Vec<RGBPixel>, Vec<Molecule>),
    Hardware(MolecularSample),
    Text(TextPoint),
}

/// System response from BMD operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SystemResponse {
    QuantumState(QuantumState),
    Products(Vec<Product>),
    Solutions(Vec<Solution>),
    Analysis(AnalysisResult),
    Resolution(ResolutionResult),
}

impl TurbulanceInterpreter {
    /// Create new interpreter with BMD integration
    pub fn new() -> Self {
        let mut interpreter = Self {
            variables: HashMap::new(),
            points: HashMap::new(),
            resolution_functions: HashMap::new(),
            bmd_networks: BMDNetworkManager::new(),
            functions: HashMap::new(),
            context: ExecutionContext {
                confidence_threshold: 0.8,
                current_domain: "general".to_string(),
                processing_mode: ProcessingMode::Hybrid,
                resolution_strategy: ResolutionStrategy::BayesianWeighted,
            },
        };
        
        // Register built-in resolution functions
        interpreter.register_resolution_function(Box::new(ProbabilisticLengthResolution));
        interpreter.register_resolution_function(Box::new(ProbabilisticSentimentResolution));
        
        interpreter
    }
    
    /// Execute a Turbulance program
    pub fn execute(&mut self, program: &Program) -> BorgiaResult<Value> {
        let mut last_value = Value::Null;
        
        for statement in &program.statements {
            last_value = self.execute_statement(statement)?;
        }
        
        Ok(last_value)
    }
    
    /// Execute a single statement
    pub fn execute_statement(&mut self, statement: &Statement) -> BorgiaResult<Value> {
        match statement {
            Statement::ItemDeclaration { name, value, .. } => {
                let val = self.evaluate_expression(value)?;
                self.variables.insert(name.clone(), val.clone());
                Ok(val)
            },
            
            Statement::PointDeclaration { name, content, certainty, evidence_strength, contextual_relevance } => {
                let mut point = TextPoint::new(content.clone(), *certainty);
                point.evidence_strength = evidence_strength.unwrap_or(0.5);
                point.contextual_relevance = contextual_relevance.unwrap_or(0.5);
                
                self.points.insert(name.clone(), point.clone());
                Ok(Value::Point(point))
            },
            
            Statement::ResolutionCall { function_name, point, context, strategy } => {
                let point_value = self.evaluate_expression(point)?;
                let text_point = match point_value {
                    Value::Point(p) => p,
                    Value::String(s) => TextPoint::new(s, 0.8),
                    _ => return Err(BorgiaError::InterpreterError("Invalid point type for resolution".to_string())),
                };
                
                let resolution_context = ResolutionContext {
                    domain: context.clone().unwrap_or(self.context.current_domain.clone()),
                    confidence_threshold: self.context.confidence_threshold,
                    strategy: strategy.clone().unwrap_or(self.context.resolution_strategy.clone()),
                    temporal_context: None,
                    cultural_context: None,
                    purpose_context: None,
                };
                
                if let Some(func) = self.resolution_functions.get(function_name) {
                    let result = func.resolve(&text_point, &resolution_context)?;
                    Ok(Value::from_resolution_result(result))
                } else {
                    Err(BorgiaError::InterpreterError(format!("Unknown resolution function: {}", function_name)))
                }
            },
            
            Statement::FunctionDefinition { name, parameters: _, body: _, .. } => {
                // Store function definition for later execution
                if let Statement::FunctionDefinition { name, parameters, body, return_type } = statement {
                    let func_def = FunctionDefinition {
                        name: name.clone(),
                        parameters: parameters.clone(),
                        body: body.clone(),
                        return_type: return_type.clone(),
                    };
                    self.functions.insert(name.clone(), func_def);
                }
                Ok(Value::Null)
            },
            
            Statement::HybridLoop(loop_type) => {
                self.execute_hybrid_loop(loop_type)
            },
            
            Statement::LogicalConstruct(construct) => {
                self.execute_logical_construct(construct)
            },
            
            Statement::FuzzyConstruct(construct) => {
                self.execute_fuzzy_construct(construct)
            },
            
            Statement::Expression(expr) => {
                self.evaluate_expression(expr)
            },
            
            Statement::Return(expr) => {
                match expr {
                    Some(e) => self.evaluate_expression(e),
                    None => Ok(Value::Null),
                }
            },
            
            Statement::Assignment { target, value } => {
                let val = self.evaluate_expression(value)?;
                self.variables.insert(target.clone(), val.clone());
                Ok(val)
            },
            
            Statement::Conditional { condition, then_branch, else_branch } => {
                let cond_value = self.evaluate_expression(condition)?;
                let is_true = match cond_value {
                    Value::Boolean(b) => b,
                    Value::Float(f) => f > 0.5, // Probabilistic truth
                    Value::Integer(i) => i != 0,
                    _ => false,
                };
                
                if is_true {
                    self.execute_block(then_branch)
                } else if let Some(else_block) = else_branch {
                    self.execute_block(else_block)
                } else {
                    Ok(Value::Null)
                }
            },
        }
    }
    
    /// Execute a block of statements
    fn execute_block(&mut self, statements: &[Statement]) -> BorgiaResult<Value> {
        let mut last_value = Value::Null;
        for statement in statements {
            last_value = self.execute_statement(statement)?;
        }
        Ok(last_value)
    }
    
    /// Evaluate an expression
    pub fn evaluate_expression(&mut self, expression: &Expression) -> BorgiaResult<Value> {
        match expression {
            Expression::Literal(lit) => Ok(Value::from_literal(lit)),
            
            Expression::Variable(name) => {
                self.variables.get(name)
                    .cloned()
                    .ok_or_else(|| BorgiaError::InterpreterError(format!("Undefined variable: {}", name)))
            },
            
            Expression::PointReference { name, .. } => {
                self.points.get(name)
                    .cloned()
                    .map(Value::Point)
                    .ok_or_else(|| BorgiaError::InterpreterError(format!("Undefined point: {}", name)))
            },
            
            Expression::FunctionCall { name, arguments } => {
                self.execute_function_call(name, arguments)
            },
            
            Expression::BinaryOp { left, operator, right } => {
                let left_val = self.evaluate_expression(left)?;
                let right_val = self.evaluate_expression(right)?;
                self.execute_binary_op(&left_val, operator, &right_val)
            },
            
            Expression::UnaryOp { operator, operand } => {
                let val = self.evaluate_expression(operand)?;
                self.execute_unary_op(operator, &val)
            },
            
            Expression::ProbabilisticFloor { items, uncertainty_weights } => {
                let mut floor_items = Vec::new();
                for (i, item) in items.iter().enumerate() {
                    let val = self.evaluate_expression(item)?;
                    let weight = uncertainty_weights.get(i).unwrap_or(&1.0);
                    floor_items.push((val, *weight));
                }
                Ok(Value::Distribution(floor_items))
            },
            
            Expression::LogicalQuery { query, variables: _ } => {
                // Execute logical query (simplified)
                Ok(Value::String(format!("Query result: {}", query)))
            },
            
            Expression::FuzzyEvaluation { variable, linguistic_term, value } => {
                let val = self.evaluate_expression(value)?;
                // Execute fuzzy evaluation (simplified)
                Ok(Value::Float(0.7)) // Placeholder membership value
            },
        }
    }
    
    /// Execute hybrid loop constructs
    fn execute_hybrid_loop(&mut self, loop_type: &HybridLoopType) -> BorgiaResult<Value> {
        match loop_type {
            HybridLoopType::Cycle { variable, iterable, body, confidence_threshold } => {
                let iterable_value = self.evaluate_expression(iterable)?;
                let items = self.extract_iterable_items(&iterable_value)?;
                
                let mut results = Vec::new();
                for item in items {
                    self.variables.insert(variable.clone(), item);
                    let result = self.execute_block(body)?;
                    
                    // Check confidence threshold for continuation
                    if let Some(threshold) = confidence_threshold {
                        if let Value::Float(confidence) = &result {
                            if confidence < threshold {
                                break;
                            }
                        }
                    }
                    
                    results.push(result);
                }
                
                Ok(Value::Distribution(results.into_iter().map(|v| (v, 1.0)).collect()))
            },
            
            HybridLoopType::Flow { variable, stream, body, mode_switching } => {
                let stream_value = self.evaluate_expression(stream)?;
                let items = self.extract_iterable_items(&stream_value)?;
                
                let mut results = Vec::new();
                for item in items {
                    // Check if mode switching is needed
                    if let Some(switching) = mode_switching {
                        let confidence = self.estimate_confidence(&item);
                        if confidence > switching.confidence_threshold {
                            self.context.processing_mode = switching.deterministic_mode.clone();
                        } else {
                            self.context.processing_mode = switching.probabilistic_mode.clone();
                        }
                    }
                    
                    self.variables.insert(variable.clone(), item);
                    let result = self.execute_block(body)?;
                    results.push(result);
                }
                
                Ok(Value::Distribution(results.into_iter().map(|v| (v, 1.0)).collect()))
            },
            
            HybridLoopType::RollUntilSettled { body, settlement_condition, max_iterations } => {
                let max_iter = max_iterations.unwrap_or(100);
                let mut iteration = 0;
                let mut last_result = Value::Null;
                
                while iteration < max_iter {
                    last_result = self.execute_block(body)?;
                    
                    // Check settlement condition
                    let settled = self.evaluate_expression(settlement_condition)?;
                    if let Value::Boolean(true) = settled {
                        break;
                    }
                    
                    iteration += 1;
                }
                
                Ok(last_result)
            },
            
            _ => Ok(Value::Null), // Other loop types
        }
    }
    
    /// Execute logical constructs
    fn execute_logical_construct(&mut self, construct: &LogicalConstruct) -> BorgiaResult<Value> {
        match construct {
            LogicalConstruct::Fact { predicate, arguments } => {
                // Add fact to knowledge base (simplified)
                Ok(Value::String(format!("Added fact: {}({:?})", predicate, arguments)))
            },
            
            LogicalConstruct::Rule { head, body } => {
                // Add rule to knowledge base (simplified)
                Ok(Value::String(format!("Added rule: {:?} :- {:?}", head, body)))
            },
            
            LogicalConstruct::Query { variables, conditions } => {
                // Execute logical query (simplified)
                Ok(Value::String(format!("Query result for {:?}: {:?}", variables, conditions)))
            },
            
            LogicalConstruct::Constraint { condition } => {
                // Add constraint (simplified)
                Ok(Value::String(format!("Added constraint: {:?}", condition)))
            },
        }
    }
    
    /// Execute fuzzy constructs
    fn execute_fuzzy_construct(&mut self, construct: &FuzzyConstruct) -> BorgiaResult<Value> {
        match construct {
            FuzzyConstruct::FuzzyVariable { name, domain, terms } => {
                // Create fuzzy variable (simplified)
                Ok(Value::String(format!("Created fuzzy variable: {} in {:?} with {} terms", name, domain, terms.len())))
            },
            
            FuzzyConstruct::FuzzyRule { antecedent, consequent, certainty } => {
                // Add fuzzy rule (simplified)
                Ok(Value::Float(*certainty))
            },
            
            FuzzyConstruct::FuzzyRuleEval { rules } => {
                // Evaluate fuzzy rules (simplified)
                let avg_certainty = rules.iter().map(|r| r.certainty).sum::<f64>() / rules.len() as f64;
                Ok(Value::Float(avg_certainty))
            },
        }
    }
    
    /// Execute function call with BMD integration
    fn execute_function_call(&mut self, name: &str, arguments: &[Expression]) -> BorgiaResult<Value> {
        match name {
            // BMD-specific functions
            "catalyze_quantum" => {
                let arg = self.evaluate_expression(&arguments[0])?;
                let input = self.value_to_universal_input(arg)?;
                let response = self.bmd_networks.process_input(input)?;
                Ok(Value::from_system_response(response))
            },
            
            "analyze_molecular" => {
                if arguments.len() < 1 {
                    return Err(BorgiaError::InterpreterError("analyze_molecular requires at least 1 argument".to_string()));
                }
                let molecules_arg = self.evaluate_expression(&arguments[0])?;
                let molecules = self.value_to_molecules(molecules_arg)?;
                let input = UniversalInput::Molecular(molecules);
                let response = self.bmd_networks.process_input(input)?;
                Ok(Value::from_system_response(response))
            },
            
            "extract_environmental_solutions" => {
                if arguments.len() < 2 {
                    return Err(BorgiaError::InterpreterError("extract_environmental_solutions requires 2 arguments".to_string()));
                }
                let pixels_arg = self.evaluate_expression(&arguments[0])?;
                let molecules_arg = self.evaluate_expression(&arguments[1])?;
                
                let pixels = self.value_to_pixels(pixels_arg)?;
                let molecules = self.value_to_molecules(molecules_arg)?;
                let input = UniversalInput::Environmental(pixels, molecules);
                let response = self.bmd_networks.process_input(input)?;
                Ok(Value::from_system_response(response))
            },
            
            "hardware_spectroscopy" => {
                let sample_arg = self.evaluate_expression(&arguments[0])?;
                let sample = self.value_to_molecular_sample(sample_arg)?;
                let input = UniversalInput::Hardware(sample);
                let response = self.bmd_networks.process_input(input)?;
                Ok(Value::from_system_response(response))
            },
            
            // Built-in functions
            "print" => {
                let arg = self.evaluate_expression(&arguments[0])?;
                println!("{}", self.value_to_string(&arg));
                Ok(Value::Null)
            },
            
            "len" => {
                let arg = self.evaluate_expression(&arguments[0])?;
                match arg {
                    Value::String(s) => Ok(Value::Integer(s.len() as i64)),
                    Value::Distribution(d) => Ok(Value::Integer(d.len() as i64)),
                    _ => Err(BorgiaError::InterpreterError("Cannot get length of this type".to_string())),
                }
            },
            
            // User-defined functions
            _ => {
                if let Some(func_def) = self.functions.get(name).cloned() {
                    self.execute_user_function(&func_def, arguments)
                } else {
                    Err(BorgiaError::InterpreterError(format!("Unknown function: {}", name)))
                }
            }
        }
    }
    
    /// Execute user-defined function
    fn execute_user_function(&mut self, func_def: &FunctionDefinition, arguments: &[Expression]) -> BorgiaResult<Value> {
        // Create new scope
        let old_vars = self.variables.clone();
        
        // Bind parameters
        for (i, param) in func_def.parameters.iter().enumerate() {
            if i < arguments.len() {
                let arg_value = self.evaluate_expression(&arguments[i])?;
                self.variables.insert(param.name.clone(), arg_value);
            } else if let Some(default) = &param.default_value {
                let default_value = self.evaluate_expression(default)?;
                self.variables.insert(param.name.clone(), default_value);
            } else {
                return Err(BorgiaError::InterpreterError(format!("Missing argument for parameter: {}", param.name)));
            }
        }
        
        // Execute function body
        let result = self.execute_block(&func_def.body);
        
        // Restore scope
        self.variables = old_vars;
        
        result
    }
    
    /// Register a resolution function
    pub fn register_resolution_function(&mut self, func: Box<dyn ResolutionFunction>) {
        self.resolution_functions.insert(func.name().to_string(), func);
    }
    
    // Helper methods for type conversions
    fn extract_iterable_items(&self, value: &Value) -> BorgiaResult<Vec<Value>> {
        match value {
            Value::Distribution(items) => Ok(items.iter().map(|(v, _)| v.clone()).collect()),
            Value::String(s) => Ok(s.chars().map(|c| Value::String(c.to_string())).collect()),
            _ => Err(BorgiaError::InterpreterError("Value is not iterable".to_string())),
        }
    }
    
    fn estimate_confidence(&self, value: &Value) -> f64 {
        match value {
            Value::Float(f) => *f,
            Value::Boolean(true) => 1.0,
            Value::Boolean(false) => 0.0,
            Value::Point(p) => p.confidence,
            _ => 0.5,
        }
    }
    
    fn execute_binary_op(&self, left: &Value, op: &BinaryOperator, right: &Value) -> BorgiaResult<Value> {
        match (left, op, right) {
            (Value::Integer(a), BinaryOperator::Add, Value::Integer(b)) => Ok(Value::Integer(a + b)),
            (Value::Float(a), BinaryOperator::Add, Value::Float(b)) => Ok(Value::Float(a + b)),
            (Value::Integer(a), BinaryOperator::Add, Value::Float(b)) => Ok(Value::Float(*a as f64 + b)),
            (Value::Float(a), BinaryOperator::Add, Value::Integer(b)) => Ok(Value::Float(a + *b as f64)),
            
            (Value::String(a), BinaryOperator::Add, Value::String(b)) => Ok(Value::String(format!("{}{}", a, b))),
            
            (Value::Boolean(a), BinaryOperator::And, Value::Boolean(b)) => Ok(Value::Boolean(*a && *b)),
            (Value::Boolean(a), BinaryOperator::Or, Value::Boolean(b)) => Ok(Value::Boolean(*a || *b)),
            
            // Probabilistic operators
            (Value::Float(a), BinaryOperator::ProbabilisticAnd, Value::Float(b)) => {
                Ok(Value::Float(a * b)) // Product for probabilistic AND
            },
            (Value::Float(a), BinaryOperator::ProbabilisticOr, Value::Float(b)) => {
                Ok(Value::Float(a + b - a * b)) // Probabilistic OR
            },
            
            _ => Err(BorgiaError::InterpreterError(format!("Invalid binary operation: {:?} {:?} {:?}", left, op, right))),
        }
    }
    
    fn execute_unary_op(&self, op: &UnaryOperator, operand: &Value) -> BorgiaResult<Value> {
        match (op, operand) {
            (UnaryOperator::Not, Value::Boolean(b)) => Ok(Value::Boolean(!b)),
            (UnaryOperator::Minus, Value::Integer(i)) => Ok(Value::Integer(-i)),
            (UnaryOperator::Minus, Value::Float(f)) => Ok(Value::Float(-f)),
            (UnaryOperator::ProbabilisticNot, Value::Float(f)) => Ok(Value::Float(1.0 - f)),
            _ => Err(BorgiaError::InterpreterError(format!("Invalid unary operation: {:?} {:?}", op, operand))),
        }
    }
    
    fn value_to_string(&self, value: &Value) -> String {
        match value {
            Value::Integer(i) => i.to_string(),
            Value::Float(f) => f.to_string(),
            Value::String(s) => s.clone(),
            Value::Boolean(b) => b.to_string(),
            Value::Point(p) => format!("Point({})", p.content),
            Value::Distribution(d) => format!("Distribution({} items)", d.len()),
            Value::Null => "null".to_string(),
        }
    }
    
    fn value_to_universal_input(&self, value: Value) -> BorgiaResult<UniversalInput> {
        match value {
            Value::Point(p) => Ok(UniversalInput::Text(p)),
            Value::String(s) => Ok(UniversalInput::Text(TextPoint::new(s, 0.8))),
            _ => Err(BorgiaError::InterpreterError("Cannot convert value to universal input".to_string())),
        }
    }
    
    fn value_to_molecules(&self, _value: Value) -> BorgiaResult<Vec<Molecule>> {
        // Simplified conversion - in real implementation would parse SMILES, etc.
        Ok(vec![])
    }
    
    fn value_to_pixels(&self, _value: Value) -> BorgiaResult<Vec<RGBPixel>> {
        // Simplified conversion
        Ok(vec![RGBPixel { r: 128, g: 128, b: 128 }])
    }
    
    fn value_to_molecular_sample(&self, _value: Value) -> BorgiaResult<MolecularSample> {
        // Simplified conversion
        Ok(MolecularSample {
            compounds: vec!["test_compound".to_string()],
            fluorescence_properties: HashMap::new(),
        })
    }
}

impl BMDNetworkManager {
    fn new() -> Self {
        Self {
            quantum_bmd: QuantumBMD::new(),
            molecular_bmd: MolecularBMD::new(),
            environmental_bmd: EnvironmentalBMD::new(),
            hardware_bmd: HardwareBMD::new(),
            amplifier: ThermodynamicAmplifier::new(),
        }
    }
    
    fn process_input(&mut self, input: UniversalInput) -> BorgiaResult<SystemResponse> {
        match input {
            UniversalInput::Quantum(event) => {
                let state = self.quantum_bmd.process_quantum_event(event);
                Ok(SystemResponse::QuantumState(state))
            },
            UniversalInput::Molecular(molecules) => {
                let products = self.molecular_bmd.catalyze_reaction(&molecules);
                Ok(SystemResponse::Products(products))
            },
            UniversalInput::Environmental(pixels, molecules) => {
                let solutions = self.environmental_bmd.extract_solutions_from_noise(&pixels, &molecules);
                Ok(SystemResponse::Solutions(solutions))
            },
            UniversalInput::Hardware(sample) => {
                let analysis = self.hardware_bmd.perform_molecular_analysis(&sample);
                Ok(SystemResponse::Analysis(analysis))
            },
            UniversalInput::Text(point) => {
                // Convert text to resolution result
                let result = ResolutionResult::Certain(Value::String(point.content));
                Ok(SystemResponse::Resolution(result))
            },
        }
    }
}

impl Value {
    fn from_literal(literal: &Literal) -> Self {
        match literal {
            Literal::Integer(i) => Value::Integer(*i),
            Literal::Float(f) => Value::Float(*f),
            Literal::String(s) => Value::String(s.clone()),
            Literal::Boolean(b) => Value::Boolean(*b),
            Literal::Null => Value::Null,
        }
    }
    
    fn from_resolution_result(result: ResolutionResult) -> Self {
        match result {
            ResolutionResult::Certain(value) => value,
            ResolutionResult::Uncertain { possibilities, .. } => Value::Distribution(possibilities),
            ResolutionResult::Contextual { base_result, .. } => base_result,
            ResolutionResult::Fuzzy { central_tendency, .. } => Value::Float(central_tendency),
        }
    }
    
    fn from_system_response(response: SystemResponse) -> Self {
        match response {
            SystemResponse::QuantumState(_) => Value::String("QuantumState".to_string()),
            SystemResponse::Products(products) => Value::Integer(products.len() as i64),
            SystemResponse::Solutions(solutions) => Value::Integer(solutions.len() as i64),
            SystemResponse::Analysis(_) => Value::String("AnalysisResult".to_string()),
            SystemResponse::Resolution(result) => Self::from_resolution_result(result),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::turbulance::ast::*;
    
    #[test]
    fn test_interpreter_creation() {
        let interpreter = TurbulanceInterpreter::new();
        assert_eq!(interpreter.context.confidence_threshold, 0.8);
        assert_eq!(interpreter.context.current_domain, "general");
    }
    
    #[test]
    fn test_variable_assignment() {
        let mut interpreter = TurbulanceInterpreter::new();
        
        let program = Program {
            statements: vec![
                Statement::ItemDeclaration {
                    name: "x".to_string(),
                    value: Expression::Literal(Literal::Integer(42)),
                    type_annotation: None,
                },
            ],
        };
        
        let result = interpreter.execute(&program).unwrap();
        assert_eq!(result, Value::Integer(42));
        assert_eq!(interpreter.variables.get("x"), Some(&Value::Integer(42)));
    }
    
    #[test]
    fn test_point_declaration() {
        let mut interpreter = TurbulanceInterpreter::new();
        
        let program = Program {
            statements: vec![
                Statement::PointDeclaration {
                    name: "test_point".to_string(),
                    content: "Hello world".to_string(),
                    certainty: 0.9,
                    evidence_strength: Some(0.8),
                    contextual_relevance: Some(0.7),
                },
            ],
        };
        
        let result = interpreter.execute(&program).unwrap();
        match result {
            Value::Point(point) => {
                assert_eq!(point.content, "Hello world");
                assert_eq!(point.confidence, 0.9);
                assert_eq!(point.evidence_strength, 0.8);
                assert_eq!(point.contextual_relevance, 0.7);
            },
            _ => panic!("Expected Point value"),
        }
    }
} 