import os
import sys

# NumPy 2.0+ compatibility shim for legacy packages
try:
    import numpy as np
    if not hasattr(np, 'sctypes'):
        np.sctypes = {
            'int': [np.int8, np.int16, np.int32, np.int64],
            'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
            'float': [getattr(np, t) for t in ['float16', 'float32', 'float64', 'longdouble'] if hasattr(np, t)],
            'complex': [getattr(np, t) for t in ['complex64', 'complex128', 'clongdouble'] if hasattr(np, t)],
            'others': [bool, object, bytes, str, np.void]
        }
except ImportError:
    pass

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from tools.train_rec import main

if __name__ == '__main__':
    main()

