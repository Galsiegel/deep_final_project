"""
Utility functions for Massive API operations.

This module contains helper functions for working with the Massive API,
"""

import os


def load_api_key():
    """
    Load the API key from the API_KEYS file in the Massive directory.
    
    The function looks for API_KEYS in the parent directory (Massive/)
    relative to this tools directory.
    
    Returns:
        str: The API key string
        
    Raises:
        FileNotFoundError: If the API_KEYS file is not found
        Exception: If there's an error reading the file
    """
    # Get the path to the Massive directory (parent of tools/)
    massive_dir = os.path.dirname(os.path.dirname(__file__))
    api_keys_path = os.path.join(massive_dir, 'API_KEYS')
    
    try:
        with open(api_keys_path, 'r') as f:
            content = f.read().strip()
            # Parse the Python-style assignment: MASSIVE_API_KEY = "key"
            if '=' in content:
                # Extract the value after the equals sign
                key_value = content.split('=', 1)[1].strip()
                # Remove quotes if present
                key_value = key_value.strip('"\'')
                return key_value
            else:
                # If it's just the key without assignment, return as-is
                return content
    except FileNotFoundError:
        raise FileNotFoundError(f"API_KEYS file not found at {api_keys_path}")
    except Exception as e:
        raise Exception(f"Error reading API key from {api_keys_path}: {e}")

