//! Abstract Syntax Tree for Turbulance Language
//! 
//! Supports all four revolutionary paradigms:
//! 1. Points and Resolutions (probabilistic language processing)
//! 2. Positional Semantics (position-dependent meaning)
//! 3. Perturbation Validation (robustness testing)
//! 4. Hybrid Processing (probabilistic loops)

use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Top-level program structure
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Program {
    pub statements: Vec<Statement>,
}

/// Statement types in Turbulance
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Statement {
    /// Variable declaration: `item x = value`
    ItemDeclaration {
        name: String,
        value: Expression,
        type_annotation: Option<TypeAnnotation>,
    },
    
    /// Point declaration: `point name = { content: "text", certainty: 0.8 }`
    PointDeclaration {
        name: String,
        content: String,
        certainty: f64,
        evidence_strength: Option<f64>,
        contextual_relevance: Option<f64>,
    },
    
    /// Resolution function call: `resolve function_name(point) given context("domain")`
    ResolutionCall {
        function_name: String,
        point: Expression,
        context: Option<String>,
        strategy: Option<ResolutionStrategy>,
    },
    
    /// Function definition: `funxn name(params) { body }`
    FunctionDefinition {
        name: String,
        parameters: Vec<Parameter>,
        body: Vec<Statement>,
        return_type: Option<TypeAnnotation>,
    },
    
    /// Hybrid loop constructs
    HybridLoop(HybridLoopType),
    
    /// Logical programming constructs
    LogicalConstruct(LogicalConstruct),
    
    /// Fuzzy logic constructs
    FuzzyConstruct(FuzzyConstruct),
    
    /// Expression statement
    Expression(Expression),
    
    /// Return statement
    Return(Option<Expression>),
    
    /// Assignment
    Assignment {
        target: String,
        value: Expression,
    },
    
    /// Conditional statement
    Conditional {
        condition: Expression,
        then_branch: Vec<Statement>,
        else_branch: Option<Vec<Statement>>,
    },
}

/// Expression types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expression {
    /// Literal values
    Literal(Literal),
    
    /// Variable reference
    Variable(String),
    
    /// Point reference with positional context
    PointReference {
        name: String,
        positional_context: Option<PositionalContext>,
    },
    
    /// Function call
    FunctionCall {
        name: String,
        arguments: Vec<Expression>,
    },
    
    /// Binary operations
    BinaryOp {
        left: Box<Expression>,
        operator: BinaryOperator,
        right: Box<Expression>,
    },
    
    /// Unary operations
    UnaryOp {
        operator: UnaryOperator,
        operand: Box<Expression>,
    },
    
    /// Probabilistic floor collection
    ProbabilisticFloor {
        items: Vec<Expression>,
        uncertainty_weights: Vec<f64>,
    },
    
    /// Logical query
    LogicalQuery {
        query: String,
        variables: Vec<String>,
    },
    
    /// Fuzzy evaluation
    FuzzyEvaluation {
        variable: String,
        linguistic_term: String,
        value: Box<Expression>,
    },
}

/// Literal values
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Literal {
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),
    Null,
}

/// Binary operators
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Equal,
    NotEqual,
    Less,
    Greater,
    LessEqual,
    GreaterEqual,
    And,
    Or,
    // Probabilistic operators
    ProbabilisticAnd,
    ProbabilisticOr,
    ProbabilisticImplies,
}

/// Unary operators
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum UnaryOperator {
    Not,
    Minus,
    ProbabilisticNot,
}

/// Hybrid loop types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HybridLoopType {
    /// Cycle loop: basic iteration with confidence-based continuation
    Cycle {
        variable: String,
        iterable: Expression,
        body: Vec<Statement>,
        confidence_threshold: Option<f64>,
    },
    
    /// Drift loop: probabilistic exploration with weighted sampling
    Drift {
        variable: String,
        distribution: Expression,
        body: Vec<Statement>,
        drift_parameters: DriftParameters,
    },
    
    /// Flow loop: streaming processing with adaptive modes
    Flow {
        variable: String,
        stream: Expression,
        body: Vec<Statement>,
        mode_switching: Option<ModeSwitching>,
    },
    
    /// Roll-until-settled: iterative convergence
    RollUntilSettled {
        body: Vec<Statement>,
        settlement_condition: Expression,
        max_iterations: Option<u32>,
    },
}

/// Drift parameters for probabilistic exploration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DriftParameters {
    pub exploration_rate: f64,
    pub convergence_threshold: f64,
    pub sampling_strategy: SamplingStrategy,
}

/// Sampling strategies for drift loops
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SamplingStrategy {
    Uniform,
    Gaussian { mean: f64, std_dev: f64 },
    Weighted { weights: Vec<f64> },
}

/// Mode switching for flow loops
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModeSwitching {
    pub confidence_threshold: f64,
    pub deterministic_mode: ProcessingMode,
    pub probabilistic_mode: ProcessingMode,
}

/// Processing modes
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProcessingMode {
    Deterministic,
    Probabilistic,
    Hybrid,
}

/// Logical programming constructs
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LogicalConstruct {
    /// Fact declaration: `fact gene("BRCA1").`
    Fact {
        predicate: String,
        arguments: Vec<Expression>,
    },
    
    /// Rule declaration: `rule head :- body.`
    Rule {
        head: LogicalTerm,
        body: Vec<LogicalTerm>,
    },
    
    /// Query: `query all X where predicate(X)`
    Query {
        variables: Vec<String>,
        conditions: Vec<LogicalTerm>,
    },
    
    /// Constraint: `constraint condition.`
    Constraint {
        condition: LogicalTerm,
    },
}

/// Logical terms
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LogicalTerm {
    Atom(String),
    Variable(String),
    Compound {
        functor: String,
        args: Vec<LogicalTerm>,
    },
    Negation(Box<LogicalTerm>),
}

/// Fuzzy logic constructs
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FuzzyConstruct {
    /// Fuzzy variable definition
    FuzzyVariable {
        name: String,
        domain: (f64, f64),
        terms: HashMap<String, MembershipFunction>,
    },
    
    /// Fuzzy rule definition
    FuzzyRule {
        antecedent: FuzzyExpression,
        consequent: FuzzyExpression,
        certainty: f64,
    },
    
    /// Fuzzy rule evaluation
    FuzzyRuleEval {
        rules: Vec<FuzzyRule>,
    },
}

/// Fuzzy expressions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FuzzyExpression {
    Term {
        variable: String,
        linguistic_term: String,
    },
    And(Box<FuzzyExpression>, Box<FuzzyExpression>),
    Or(Box<FuzzyExpression>, Box<FuzzyExpression>),
    Not(Box<FuzzyExpression>),
    Very(Box<FuzzyExpression>),
    Somewhat(Box<FuzzyExpression>),
    Extremely(Box<FuzzyExpression>),
}

/// Fuzzy rule structure
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FuzzyRule {
    pub antecedent: FuzzyExpression,
    pub consequent: FuzzyExpression,
    pub certainty: f64,
}

/// Membership function types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MembershipFunction {
    Triangular { a: f64, b: f64, c: f64 },
    Trapezoidal { a: f64, b: f64, c: f64, d: f64 },
    Gaussian { mean: f64, std_dev: f64 },
    Sigmoid { a: f64, c: f64 },
}

/// Resolution strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResolutionStrategy {
    MaximumLikelihood,
    Conservative,
    BayesianWeighted,
    FullDistribution,
    Exploratory,
}

/// Positional context for semantic analysis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PositionalContext {
    pub sentence_position: usize,
    pub word_position: usize,
    pub semantic_role: SemanticRole,
    pub position_weight: f64,
    pub order_dependency: f64,
}

/// Semantic roles for positional analysis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SemanticRole {
    Subject,
    Predicate,
    Object,
    Modifier,
    Determiner,
    Preposition,
    Conjunction,
    Temporal,
    Epistemic,
}

/// Type annotations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TypeAnnotation {
    Integer,
    Float,
    String,
    Boolean,
    Point,
    ProbabilisticFloor,
    FuzzySet,
    LogicalTerm,
    Custom(String),
}

/// Function parameters
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Parameter {
    pub name: String,
    pub type_annotation: Option<TypeAnnotation>,
    pub default_value: Option<Expression>,
}

/// AST visitor trait for traversal and transformation
pub trait ASTVisitor<T> {
    fn visit_program(&mut self, program: &Program) -> T;
    fn visit_statement(&mut self, statement: &Statement) -> T;
    fn visit_expression(&mut self, expression: &Expression) -> T;
    fn visit_hybrid_loop(&mut self, loop_type: &HybridLoopType) -> T;
    fn visit_logical_construct(&mut self, construct: &LogicalConstruct) -> T;
    fn visit_fuzzy_construct(&mut self, construct: &FuzzyConstruct) -> T;
}

/// Mutable AST visitor for transformations
pub trait ASTMutVisitor<T> {
    fn visit_program_mut(&mut self, program: &mut Program) -> T;
    fn visit_statement_mut(&mut self, statement: &mut Statement) -> T;
    fn visit_expression_mut(&mut self, expression: &mut Expression) -> T;
} 