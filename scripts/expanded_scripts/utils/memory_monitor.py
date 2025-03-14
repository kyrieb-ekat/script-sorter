# memory_monitor.py - Memory monitoring and optimization utilities

import os
import psutil
import tensorflow as tf
import numpy as np
from typing import Tuple, Optional, Union, Dict, Any
import logging

class MemoryMonitor(tf.keras.callbacks.Callback):
    """Callback for monitoring memory usage during model training.
    
    This callback tracks RAM and GPU memory usage at the end of each epoch
    and optionally reports it to the console or log files.
    
    Attributes:
        print_interval: How often to print memory stats (in epochs)
        log_to_file: Whether to log memory stats to a file
        log_file: Path to the log file (if log_to_file is True)
    """
    
    def __init__(self, print_interval: int = 1, 
                log_to_file: bool = False,
                log_file: Optional[str] = None):
        """Initialize the memory monitor.
        
        Args:
            print_interval: Print memory stats every N epochs
            log_to_file: Whether to log memory stats to a file
            log_file: Path to the log file (required if log_to_file is True)
        """
        super(MemoryMonitor, self).__init__()
        self.print_interval = print_interval
        self.log_to_file = log_to_file
        
        if log_to_file and log_file is None:
            raise ValueError("log_file must be provided when log_to_file is True")
            
        self.log_file = log_file
        
        # Initialize log file if needed
        if self.log_to_file:
            with open(self.log_file, 'w') as f:
                f.write("epoch,ram_usage_mb,gpu_memory_mb\n")
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None):
        """Record memory usage at the end of each epoch.
        
        Args:
            epoch: Current epoch number
            logs: Metrics logged by the parent process
        """
        if (epoch + 1) % self.print_interval == 0:
            # Get RAM usage
            process = psutil.Process(os.getpid())
            ram_usage = process.memory_info().rss / 1024 / 1024  # MB
            
            # Try to get GPU memory usage
            gpu_memory = None
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    mem_info = tf.config.experimental.get_memory_info('GPU:0')
                    gpu_memory = mem_info['current'] / 1024 / 1024  # MB
                except (AttributeError, ValueError, RuntimeError) as e:
                    # We'll just report that we couldn't get GPU memory info
                    pass
            
            # Print memory usage
            memory_str = f"\nEpoch {epoch + 1} Memory Usage - RAM: {ram_usage:.2f} MB"
            if gpu_memory is not None:
                memory_str += f", GPU: {gpu_memory:.2f} MB"
            print(memory_str)
            
            # Log to file if requested
            if self.log_to_file:
                with open(self.log_file, 'a') as f:
                    f.write(f"{epoch + 1},{ram_usage:.2f},{gpu_memory if gpu_memory else 'N/A'}\n")

def get_available_memory() -> int:
    """Get the amount of available system memory in bytes.
    
    For GPU systems, this function attempts to estimate the amount of
    available GPU memory. If that fails, it falls back to system RAM.
    
    Returns:
        Available memory in bytes
    """
    # Try to get GPU memory if available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Try to get GPU device properties
            gpu = tf.config.experimental.get_device_details(gpus[0])
            if 'memory_limit' in gpu:
                return gpu['memory_limit']
        except (AttributeError, ValueError, RuntimeError):
            # Fall back to an estimate
            try:
                mem_info = tf.config.experimental.get_memory_info('GPU:0')
                total = mem_info['current'] + mem_info['peak']
                available = total - mem_info['current']
                return available
            except (AttributeError, ValueError, RuntimeError):
                # Final fallback - assume a typical GPU has 4GB of memory
                return 4 * 1024 * 1024 * 1024
    
    # If no GPU or couldn't estimate GPU memory, use system RAM
    return psutil.virtual_memory().available

def optimize_batch_size(
    available_memory: int,
    image_size: Tuple[int, int],
    precision: int = 32,
    min_batch: int = 4,
    max_batch: int = 128,
    safety_factor: float = 0.8
) -> int:
    """Calculate optimal batch size based on available memory.
    
    This function estimates the maximum batch size that will fit in memory
    based on the image size and available memory, with a safety margin.
    
    Args:
        available_memory: Available memory in bytes
        image_size: Tuple of (height, width) for input images
        precision: Bit precision (16, 32, or 64)
        min_batch: Minimum batch size to return
        max_batch: Maximum batch size to return
        safety_factor: Fraction of available memory to use (0.0 to 1.0)
        
    Returns:
        Optimal batch size
    """
    # Calculate bytes per image
    bytes_per_pixel = precision / 8
    image_bytes = image_size[0] * image_size[1] * 3 * bytes_per_pixel  # 3 channels
    
    # Estimate model overhead - this varies but a rough estimate is
    # between 200MB-1GB depending on model size
    model_overhead = 500 * 1024 * 1024  # 500 MB
    
    # Account for gradient memory, optimizer states, and other overhead
    # During backpropagation, we need roughly 3x the memory of forward pass
    effective_image_bytes = image_bytes * 4  
    
    # Apply safety factor to available memory
    safe_memory = available_memory * safety_factor - model_overhead
    
    # Calculate optimal batch size
    optimal_batch = int(safe_memory / effective_image_bytes)
    
    # Clamp between min and max
    optimal_batch = max(min_batch, min(optimal_batch, max_batch))
    
    # Round down to nearest power of 2 for better GPU utilization
    power_of_2 = 2 ** int(np.log2(optimal_batch))
    if power_of_2 >= min_batch:
        return power_of_2
        
    return optimal_batch

def log_memory_stats() -> Dict[str, Union[float, str]]:
    """Log current memory statistics.
    
    Returns:
        Dictionary with memory statistics
    """
    # System memory
    vm = psutil.virtual_memory()
    system_total = vm.total / (1024 * 1024)  # MB
    system_available = vm.available / (1024 * 1024)  # MB
    system_used = vm.used / (1024 * 1024)  # MB
    system_percent = vm.percent
    
    # Process memory
    process = psutil.Process(os.getpid())
    process_memory = process.memory_info().rss / (1024 * 1024)  # MB
    
    # GPU memory if available
    gpu_total = "N/A"
    gpu_used = "N/A"
    gpu_free = "N/A"
    gpu_percent = "N/A"
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            mem_info = tf.config.experimental.get_memory_info('GPU:0')
            gpu_used = mem_info['current'] / (1024 * 1024)  # MB
            gpu_peak = mem_info['peak'] / (1024 * 1024)  # MB
            # Note: total and free are estimates here
            gpu_total = gpu_used + gpu_peak
            gpu_free = gpu_total - gpu_used
            gpu_percent = (gpu_used / gpu_total) * 100 if gpu_total > 0 else 0
        except (AttributeError, ValueError, RuntimeError):
            pass
    
    # Log all stats
    stats = {
        'system_total_mb': system_total,
        'system_available_mb': system_available,
        'system_used_mb': system_used,
        'system_used_percent': system_percent,
        'process_memory_mb': process_memory,
        'gpu_total_mb': gpu_total,
        'gpu_used_mb': gpu_used,
        'gpu_free_mb': gpu_free,
        'gpu_used_percent': gpu_percent
    }
    
    # Print a summary
    print("\nMemory Stats:")
    print(f"  System: {system_used:.0f}MB / {system_total:.0f}MB ({system_percent:.1f}%)")
    print(f"  Process: {process_memory:.0f}MB")
    print(f"  GPU: {gpu_used if isinstance(gpu_used, str) else f'{gpu_used:.0f}MB'} / "
          f"{gpu_total if isinstance(gpu_total, str) else f'{gpu_total:.0f}MB'}")
    
    return stats