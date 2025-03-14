#GPU utility functions
# gpu_utils.py - GPU detection and configuration utilities

import os
import tensorflow as tf
from typing import List, Optional, Tuple
import logging

def get_available_gpus() -> List[tf.config.PhysicalDevice]:
    """Detect and return available GPU devices.
    
    This function lists all physical GPU devices that TensorFlow can access.
    
    Returns:
        List of available GPU devices
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    return gpus

def configure_gpu_memory_growth(gpus: Optional[List[tf.config.PhysicalDevice]] = None) -> bool:
    """Configure GPUs to use memory growth instead of allocating all memory upfront.
    
    By enabling memory growth, TensorFlow will only allocate GPU memory as needed,
    which can help avoid out-of-memory errors when sharing GPUs with other processes.
    
    Args:
        gpus: List of GPU devices to configure (if None, will detect automatically)
        
    Returns:
        True if GPUs were configured successfully, False otherwise
    """
    if gpus is None:
        gpus = get_available_gpus()
        
    if not gpus:
        logging.info("No GPUs detected.")
        return False
        
    try:
        # Enable memory growth for all GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.info(f"Successfully configured {len(gpus)} GPU(s) with memory growth enabled.")
        return True
    except RuntimeError as e:
        logging.error(f"Error configuring GPU memory growth: {e}")
        return False

def set_gpu_memory_limit(memory_limit_mb: int, 
                        gpu_index: int = 0) -> bool:
    """Limit the amount of GPU memory that TensorFlow can allocate.
    
    This is useful when you want to reserve some GPU memory for other processes
    or when you want to simulate a lower-memory environment.
    
    Args:
        memory_limit_mb: Memory limit in megabytes
        gpu_index: Index of the GPU to configure (default: 0)
        
    Returns:
        True if the memory limit was set successfully, False otherwise
    """
    gpus = get_available_gpus()
    
    if not gpus or gpu_index >= len(gpus):
        logging.warning(f"GPU with index {gpu_index} not ava