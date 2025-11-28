#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config_loader.py — Central configuration loader for ML Sidecar.

This module provides:
    load_settings(path="./config/settings.yaml")

Features:
    - Loads YAML configuration safely
    - Provides a clear error message if config is missing or malformed
    - Ensures UTF-8 compatibility
"""

import yaml
import os


def load_settings(path: str = "./config/settings.yaml") -> dict:
    """
    Load application-wide configuration from a YAML file.

    Parameters
    ----------
    path : str
        Path to the YAML configuration file. Defaults to ./config/settings.yaml

    Returns
    -------
    dict
        Parsed YAML configuration as a Python dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If the YAML file cannot be parsed.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Configuration file not found: {path}\n"
            f"Expected location: {os.path.abspath(path)}"
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in configuration file '{path}': {e}")
    except Exception as e:
        raise ValueError(f"Failed to load configuration file '{path}': {e}")