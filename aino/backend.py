import numpy as np
from typing import Any

try:
    import cupy as cp
    cp.cuda.Device(0).use()
    xp = cp
    USING_GPU = True
    print("GPU Detected! Using Cupy")
except (ImportError, Exception):
    xp = np
    USING_GPU = False
    print("Running on Standard CPU (NumPy).")

def to_cpu(array: Any) -> np.ndarray:
    """
    Transfers an array from GPU (CuPy) memory back to CPU (NumPy) memory.

    If GPU acceleration is active and the input is a CuPy array, this function
    safely converts and returns it as a NumPy array. If the input is already
    on the CPU or GPU acceleration is disabled, it returns the array unaltered.

    Args:
        array (Any): The data array to be transferred.

    Returns:
        np.ndarray: The array residing in CPU memory, ready for NumPy or Matplotlib operations.
    """
    if USING_GPU and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return array