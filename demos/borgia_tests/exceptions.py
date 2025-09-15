"""
Borgia Test Framework - Exception Classes
========================================

Custom exception classes for the Borgia biological Maxwell demons (BMD)
cheminformatics test framework. Provides hierarchical exception handling
for different types of validation and testing errors.

Author: Borgia Development Team
"""

from typing import Optional, Any, Dict, List


class BorgiaTestError(Exception):
    """Base exception class for Borgia test framework errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """
        Initialize Borgia test error.
        
        Args:
            message: Human-readable error message
            error_code: Optional error code for programmatic handling
            details: Optional dictionary with additional error details
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self) -> str:
        """String representation of the error."""
        error_str = f"BorgiaTestError: {self.message}"
        if self.error_code:
            error_str += f" (Code: {self.error_code})"
        if self.details:
            error_str += f" Details: {self.details}"
        return error_str


class ValidationError(BorgiaTestError):
    """Exception raised when validation tests fail."""
    
    def __init__(self, message: str, validation_type: Optional[str] = None, 
                 failed_criteria: Optional[List[str]] = None, **kwargs):
        """
        Initialize validation error.
        
        Args:
            message: Error message
            validation_type: Type of validation that failed
            failed_criteria: List of specific criteria that failed
        """
        super().__init__(message, **kwargs)
        self.validation_type = validation_type
        self.failed_criteria = failed_criteria or []


class BenchmarkError(BorgiaTestError):
    """Exception raised when benchmark tests fail."""
    
    def __init__(self, message: str, benchmark_name: Optional[str] = None,
                 performance_data: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize benchmark error.
        
        Args:
            message: Error message
            benchmark_name: Name of the failed benchmark
            performance_data: Performance data at time of failure
        """
        super().__init__(message, **kwargs)
        self.benchmark_name = benchmark_name
        self.performance_data = performance_data or {}


class ConfigurationError(BorgiaTestError):
    """Exception raised when configuration is invalid."""
    
    def __init__(self, message: str, config_section: Optional[str] = None,
                 invalid_parameters: Optional[List[str]] = None, **kwargs):
        """
        Initialize configuration error.
        
        Args:
            message: Error message
            config_section: Configuration section with error
            invalid_parameters: List of invalid parameter names
        """
        super().__init__(message, **kwargs)
        self.config_section = config_section
        self.invalid_parameters = invalid_parameters or []


class HardwareError(BorgiaTestError):
    """Exception raised when hardware integration fails."""
    
    def __init__(self, message: str, hardware_component: Optional[str] = None,
                 system_requirements: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize hardware error.
        
        Args:
            message: Error message
            hardware_component: Hardware component that failed
            system_requirements: System requirements not met
        """
        super().__init__(message, **kwargs)
        self.hardware_component = hardware_component
        self.system_requirements = system_requirements or {}


class MolecularGenerationError(ValidationError):
    """Exception raised when molecular generation fails."""
    
    def __init__(self, message: str, generation_parameters: Optional[Dict[str, Any]] = None,
                 failed_molecules: Optional[int] = None, **kwargs):
        """
        Initialize molecular generation error.
        
        Args:
            message: Error message
            generation_parameters: Parameters used for generation
            failed_molecules: Number of molecules that failed generation
        """
        super().__init__(message, validation_type="molecular_generation", **kwargs)
        self.generation_parameters = generation_parameters or {}
        self.failed_molecules = failed_molecules


class BMDNetworkError(ValidationError):
    """Exception raised when BMD network coordination fails."""
    
    def __init__(self, message: str, network_layer: Optional[str] = None,
                 coordination_failure: Optional[str] = None, **kwargs):
        """
        Initialize BMD network error.
        
        Args:
            message: Error message
            network_layer: BMD network layer (quantum/molecular/environmental)
            coordination_failure: Type of coordination failure
        """
        super().__init__(message, validation_type="bmd_network", **kwargs)
        self.network_layer = network_layer
        self.coordination_failure = coordination_failure


class InformationCatalysisError(ValidationError):
    """Exception raised when information catalysis validation fails."""
    
    def __init__(self, message: str, catalysis_type: Optional[str] = None,
                 amplification_achieved: Optional[float] = None, **kwargs):
        """
        Initialize information catalysis error.
        
        Args:
            message: Error message
            catalysis_type: Type of catalysis that failed
            amplification_achieved: Amplification factor achieved (if measurable)
        """
        super().__init__(message, validation_type="information_catalysis", **kwargs)
        self.catalysis_type = catalysis_type
        self.amplification_achieved = amplification_achieved


class CascadeFailureError(BorgiaTestError):
    """Exception raised when cascade failure is detected."""
    
    def __init__(self, message: str, failure_origin: Optional[str] = None,
                 affected_systems: Optional[List[str]] = None,
                 risk_level: Optional[str] = None, **kwargs):
        """
        Initialize cascade failure error.
        
        Args:
            message: Error message
            failure_origin: System component where failure originated
            affected_systems: List of systems affected by cascade
            risk_level: Risk level (LOW/MEDIUM/HIGH/CRITICAL)
        """
        super().__init__(message, error_code="CASCADE_FAILURE", **kwargs)
        self.failure_origin = failure_origin
        self.affected_systems = affected_systems or []
        self.risk_level = risk_level


class QualityControlError(ValidationError):
    """Exception raised when quality control validation fails."""
    
    def __init__(self, message: str, qc_stage: Optional[str] = None,
                 rejection_reasons: Optional[List[str]] = None,
                 zero_tolerance_violation: bool = False, **kwargs):
        """
        Initialize quality control error.
        
        Args:
            message: Error message
            qc_stage: Quality control stage that failed
            rejection_reasons: List of reasons for rejection
            zero_tolerance_violation: Whether this violates zero-tolerance policy
        """
        super().__init__(message, validation_type="quality_control", **kwargs)
        self.qc_stage = qc_stage
        self.rejection_reasons = rejection_reasons or []
        self.zero_tolerance_violation = zero_tolerance_violation


class DataIntegrityError(BorgiaTestError):
    """Exception raised when data integrity is compromised."""
    
    def __init__(self, message: str, data_type: Optional[str] = None,
                 corruption_detected: bool = False, **kwargs):
        """
        Initialize data integrity error.
        
        Args:
            message: Error message
            data_type: Type of data with integrity issues
            corruption_detected: Whether data corruption was detected
        """
        super().__init__(message, error_code="DATA_INTEGRITY", **kwargs)
        self.data_type = data_type
        self.corruption_detected = corruption_detected


class SystemResourceError(BorgiaTestError):
    """Exception raised when system resources are insufficient."""
    
    def __init__(self, message: str, resource_type: Optional[str] = None,
                 required_amount: Optional[float] = None,
                 available_amount: Optional[float] = None, **kwargs):
        """
        Initialize system resource error.
        
        Args:
            message: Error message
            resource_type: Type of resource (memory, cpu, storage, etc.)
            required_amount: Amount of resource required
            available_amount: Amount of resource available
        """
        super().__init__(message, error_code="INSUFFICIENT_RESOURCES", **kwargs)
        self.resource_type = resource_type
        self.required_amount = required_amount
        self.available_amount = available_amount


class TimeoutError(BorgiaTestError):
    """Exception raised when operations timeout."""
    
    def __init__(self, message: str, operation_name: Optional[str] = None,
                 timeout_duration: Optional[float] = None, **kwargs):
        """
        Initialize timeout error.
        
        Args:
            message: Error message
            operation_name: Name of operation that timed out
            timeout_duration: Duration in seconds before timeout
        """
        super().__init__(message, error_code="TIMEOUT", **kwargs)
        self.operation_name = operation_name
        self.timeout_duration = timeout_duration


# Exception hierarchy mapping for easy reference
EXCEPTION_HIERARCHY = {
    'BorgiaTestError': {
        'ValidationError': {
            'MolecularGenerationError': {},
            'BMDNetworkError': {},
            'InformationCatalysisError': {},
            'QualityControlError': {}
        },
        'BenchmarkError': {},
        'ConfigurationError': {},
        'HardwareError': {},
        'CascadeFailureError': {},
        'DataIntegrityError': {},
        'SystemResourceError': {},
        'TimeoutError': {}
    }
}


def get_exception_class(error_type: str) -> type:
    """
    Get exception class by name.
    
    Args:
        error_type: Name of exception class
        
    Returns:
        Exception class
        
    Raises:
        ValueError: If error_type is not found
    """
    exception_classes = {
        'BorgiaTestError': BorgiaTestError,
        'ValidationError': ValidationError,
        'BenchmarkError': BenchmarkError,
        'ConfigurationError': ConfigurationError,
        'HardwareError': HardwareError,
        'MolecularGenerationError': MolecularGenerationError,
        'BMDNetworkError': BMDNetworkError,
        'InformationCatalysisError': InformationCatalysisError,
        'CascadeFailureError': CascadeFailureError,
        'QualityControlError': QualityControlError,
        'DataIntegrityError': DataIntegrityError,
        'SystemResourceError': SystemResourceError,
        'TimeoutError': TimeoutError
    }
    
    if error_type not in exception_classes:
        raise ValueError(f"Unknown exception type: {error_type}")
    
    return exception_classes[error_type]


def handle_exception_gracefully(func):
    """
    Decorator for graceful exception handling in test functions.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function with exception handling
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except BorgiaTestError:
            # Re-raise Borgia-specific errors
            raise
        except Exception as e:
            # Wrap other exceptions in BorgiaTestError
            raise BorgiaTestError(
                f"Unexpected error in {func.__name__}: {str(e)}",
                error_code="UNEXPECTED_ERROR",
                details={
                    'function': func.__name__,
                    'original_exception': type(e).__name__,
                    'original_message': str(e)
                }
            ) from e
    
    return wrapper
