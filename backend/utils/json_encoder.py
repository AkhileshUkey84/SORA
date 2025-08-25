# Custom JSON encoder for numpy types
# utils/json_encoder.py
"""
Custom JSON encoder to handle numpy types and other serialization issues
"""

import json
import numpy as np
from datetime import datetime, date
from decimal import Decimal


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types and other common serialization issues"""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, date):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif hasattr(obj, 'isoformat') and callable(obj.isoformat):
            # Handle other datetime-like objects
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            # Handle custom objects by converting to dict
            return obj.__dict__
        
        # Let the base class handle the error
        return super().default(obj)


def convert_numpy_types(data):
    """
    Recursively convert numpy types in nested data structures
    """
    if isinstance(data, dict):
        return {key: convert_numpy_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(convert_numpy_types(item) for item in data)
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data


def safe_json_dumps(data, **kwargs):
    """
    Safe JSON serialization that handles numpy types
    """
    # First convert numpy types recursively
    converted_data = convert_numpy_types(data)
    
    # Use custom encoder as fallback
    return json.dumps(converted_data, cls=NumpyJSONEncoder, **kwargs)
