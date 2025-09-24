"""
Configuration Parser for BEVFormer Training
Handles YAML configuration loading, inheritance, and validation
"""

import os
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
import re


class ConfigParser:
    """
    Configuration parser with support for:
    - YAML configuration files
    - Configuration inheritance (_base_)
    - Environment variable substitution
    - Configuration validation
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration parser

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = {}

        if config_path:
            self.config = self.load_config(config_path)

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file with inheritance support

        Args:
            config_path: Path to configuration file

        Returns:
            Loaded and processed configuration dictionary
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Handle inheritance
        if '_base_' in config:
            config = self._handle_inheritance(config, config_path.parent)

        # Handle environment variable substitution
        config = self._substitute_env_vars(config)

        # Validate configuration
        self._validate_config(config)

        return config

    def _handle_inheritance(self, config: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
        """
        Handle configuration inheritance using _base_ key

        Args:
            config: Configuration dictionary
            base_dir: Base directory for resolving relative paths

        Returns:
            Merged configuration
        """
        base_configs = config.pop('_base_')
        if isinstance(base_configs, str):
            base_configs = [base_configs]

        # Load base configurations
        merged_config = {}
        for base_config in base_configs:
            base_path = base_dir / base_config
            if not base_path.exists():
                # Try absolute path
                base_path = Path(base_config)

            if not base_path.exists():
                raise FileNotFoundError(f"Base configuration not found: {base_config}")

            base_cfg = self.load_config(str(base_path))
            merged_config = self._deep_merge(merged_config, base_cfg)

        # Merge current config on top of base configs
        merged_config = self._deep_merge(merged_config, config)

        return merged_config

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries

        Args:
            base: Base configuration
            override: Override configuration

        Returns:
            Merged configuration
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def _substitute_env_vars(self, config: Any) -> Any:
        """
        Substitute environment variables in configuration
        Supports ${VAR_NAME} and ${VAR_NAME:default_value} syntax

        Args:
            config: Configuration object (dict, list, str, etc.)

        Returns:
            Configuration with substituted environment variables
        """
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str):
            # Pattern to match ${VAR_NAME} or ${VAR_NAME:default}
            pattern = r'\$\{([^}]+)\}'

            def replace_var(match):
                var_expr = match.group(1)
                if ':' in var_expr:
                    var_name, default_value = var_expr.split(':', 1)
                    return os.environ.get(var_name, default_value)
                else:
                    var_name = var_expr
                    if var_name in os.environ:
                        return os.environ[var_name]
                    else:
                        raise ValueError(f"Environment variable '{var_name}' not found and no default provided")

            return re.sub(pattern, replace_var, config)
        else:
            return config

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration structure and required fields

        Args:
            config: Configuration to validate
        """
        required_sections = ['model', 'training', 'data']

        for section in required_sections:
            if section not in config:
                raise ValueError(f"Required configuration section '{section}' not found")

        # Validate model section
        model_config = config['model']
        if 'embed_dims' not in model_config:
            raise ValueError("'embed_dims' not found in model configuration")

        # Validate training section
        training_config = config['training']
        required_training_fields = ['epochs', 'batch_size', 'optimizer']
        for field in required_training_fields:
            if field not in training_config:
                raise ValueError(f"Required training field '{field}' not found")

        # Validate data section
        data_config = config['data']
        if 'data_root' not in data_config:
            raise ValueError("'data_root' not found in data configuration")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation

        Args:
            key: Configuration key (e.g., 'model.embed_dims')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation

        Args:
            key: Configuration key (e.g., 'model.embed_dims')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def update_from_args(self, args: Dict[str, Any]) -> None:
        """
        Update configuration from command line arguments

        Args:
            args: Dictionary of command line arguments
        """
        # Map common command line arguments to configuration keys
        arg_mapping = {
            'lr': 'training.optimizer.lr',
            'learning_rate': 'training.optimizer.lr',
            'batch_size': 'training.batch_size',
            'epochs': 'training.epochs',
            'data_root': 'data.data_root',
            'experiment_name': 'experiment.name',
        }

        for arg_key, config_key in arg_mapping.items():
            if arg_key in args and args[arg_key] is not None:
                self.set(config_key, args[arg_key])

    def save_config(self, save_path: str) -> None:
        """
        Save current configuration to file

        Args:
            save_path: Path to save configuration
        """
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access"""
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style setting"""
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        """Check if key exists in configuration"""
        return self.get(key) is not None


def load_config(config_path: str, **kwargs) -> ConfigParser:
    """
    Convenience function to load configuration

    Args:
        config_path: Path to configuration file
        **kwargs: Additional arguments to override in configuration

    Returns:
        ConfigParser instance
    """
    config_parser = ConfigParser(config_path)

    # Update with any provided kwargs
    if kwargs:
        config_parser.update_from_args(kwargs)

    return config_parser


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration values

    Returns:
        Default configuration dictionary
    """
    return {
        'model': {
            'type': 'BEVFormer',
            'embed_dims': 256,
            'encoder_layers': 6,
            'decoder_layers': 6,
            'num_query': 900,
            'bev_h': 200,
            'bev_w': 200,
        },
        'training': {
            'epochs': 24,
            'batch_size': 1,
            'optimizer': {
                'type': 'AdamW',
                'lr': 2.0e-4,
                'weight_decay': 0.01,
            },
        },
        'data': {
            'data_root': '../data/nuscenes',
            'train_pkl': 'nuscenes_infos_temporal_train.pkl',
            'val_pkl': 'nuscenes_infos_temporal_val.pkl',
        },
        'logging': {
            'tensorboard': {
                'enabled': True,
                'log_dir': './logs',
                'log_interval': 50,
            }
        },
        'checkpoint': {
            'save_dir': './checkpoints',
            'save_interval': 1,
        }
    }


# Example usage and testing
if __name__ == "__main__":
    # Test configuration loading
    try:
        # Load a configuration file
        config_path = "../configs/bevformer_tiny_r50.yaml"
        if os.path.exists(config_path):
            config = load_config(config_path)
            print(f"Loaded configuration: {config['experiment.name']}")
            print(f"Model embed_dims: {config['model.embed_dims']}")
            print(f"Training epochs: {config['training.epochs']}")
            print(f"Batch size: {config['training.batch_size']}")
        else:
            print(f"Configuration file not found: {config_path}")
            print("Using default configuration:")
            config = ConfigParser()
            config.config = get_default_config()
            print(f"Default embed_dims: {config['model.embed_dims']}")

    except Exception as e:
        print(f"Error loading configuration: {e}")
        import traceback
        traceback.print_exc()