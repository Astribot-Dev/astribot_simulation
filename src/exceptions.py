"""Custom exceptions for Astribot Simulation."""


class AstribotSimulationError(Exception):
    """Base exception for Astribot Simulation."""
    pass


class ConfigurationError(AstribotSimulationError):
    """Configuration related errors."""
    pass


class SimulatorNotFoundError(AstribotSimulationError):
    """Simulator backend not available."""
    pass


class ROSInterfaceError(AstribotSimulationError):
    """ROS communication errors."""
    pass
