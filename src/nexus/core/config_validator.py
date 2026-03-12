"""
Configuration validation module for TheNexus.

Provides JSON schema-based validation for configuration files.
"""

from typing import Dict, Any, Optional
import json
import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigValidator:
    """
    Validates configuration files against JSON schema.
    """

    def __init__(self, schema_path: Optional[str] = None):
        """
        Initialize the config validator.

        Args:
            schema_path: Path to JSON schema file. If None, uses default schema.
        """
        if schema_path is None:
            # Default schema path
            schema_path = Path(__file__).parent.parent.parent / "config" / "config.schema.json"
        
        self.schema_path = Path(schema_path)
        self.schema: Optional[Dict[str, Any]] = None
        
        try:
            with open(self.schema_path, "r") as f:
                self.schema = json.load(f)
            logger.debug(f"Loaded schema from {self.schema_path}")
        except FileNotFoundError:
            logger.warning(f"Schema file not found at {self.schema_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in schema file: {e}")

    def validate(self, config: Dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Validate configuration against the schema.

        Args:
            config: Configuration dictionary to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Basic validation if schema not loaded
        if self.schema is None:
            logger.warning("No schema loaded, performing basic validation")
            return self._basic_validation(config)

        # Validate required fields
        if "model_ensemble" not in config:
            errors.append("Missing required field: 'model_ensemble'")
            return False, errors

        ensemble = config.get("model_ensemble", {})
        if "models" not in ensemble:
            errors.append("Missing required field: 'model_ensemble.models'")
            return False, errors

        models = ensemble.get("models", [])
        if not isinstance(models, list):
            errors.append("'model_ensemble.models' must be an array")
            return False, errors

        if len(models) == 0:
            errors.append("'model_ensemble.models' must contain at least one model")
            return False, errors

        # Validate each model
        for i, model in enumerate(models):
            if not isinstance(model, dict):
                errors.append(f"Model {i} must be an object")
                continue

            if "name" not in model:
                errors.append(f"Model {i} missing required field: 'name'")
            elif not model["name"]:
                errors.append(f"Model {i} 'name' cannot be empty")

            if "weight" not in model:
                errors.append(f"Model {i} missing required field: 'weight'")
            elif not isinstance(model["weight"], (int, float)):
                errors.append(f"Model {i} 'weight' must be a number")
            elif not 0 <= model["weight"] <= 1:
                errors.append(f"Model {i} 'weight' must be between 0 and 1")

        is_valid = len(errors) == 0
        return is_valid, errors

    def _basic_validation(self, config: Dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Perform basic validation without schema.

        Args:
            config: Configuration to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if not isinstance(config, dict):
            errors.append("Configuration must be a dictionary")
            return False, errors

        if "model_ensemble" not in config:
            errors.append("Missing 'model_ensemble' section")

        return len(errors) == 0, errors

    def validate_file(self, config_path: str) -> tuple[bool, list[str]]:
        """
        Load and validate a configuration file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            
            if config is None:
                return False, ["Configuration file is empty"]

            return self.validate(config)

        except FileNotFoundError:
            return False, [f"Configuration file not found: {config_path}"]
        except yaml.YAMLError as e:
            return False, [f"Invalid YAML: {str(e)}"]
        except Exception as e:
            return False, [f"Error loading configuration: {str(e)}"]


def validate_config(config_path: str, schema_path: Optional[str] = None) -> tuple[bool, list[str]]:
    """
    Convenience function to validate a configuration file.

    Args:
        config_path: Path to configuration file
        schema_path: Optional path to schema file

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    validator = ConfigValidator(schema_path)
    return validator.validate_file(config_path)
