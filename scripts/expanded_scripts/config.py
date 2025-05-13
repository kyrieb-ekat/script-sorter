# config.py - Central configuration file that all modules can import

import os
from typing import Tuple, List, Dict, Any

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
VAL_DIR = os.path.join(DATASET_DIR, 'validation')
TEST_DIR = os.path.join(DATASET_DIR, 'test')
MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
FEATURES_DB = os.path.join(DATASET_DIR, 'features_db.pkl')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
VISUALIZATION_DIR = os.path.join(BASE_DIR, 'visualizations')
FEEDBACK_DB = os.path.join(BASE_DIR, 'user_feedback.json')

# Ensure critical directories exist
for directory in [DATASET_DIR, MODELS_DIR, LOGS_DIR, VISUALIZATION_DIR]:
    os.makedirs(directory, exist_ok=True)

# Model parameters
IMAGE_SIZE: Tuple[int, int] = (224, 224)
BATCH_SIZE: int = 32  # Will be optimized dynamically
NUM_CLASSES: int = 5
EXPECTED_CLASSES: List[str] = ['0', '1', '2', '3', '4']

# Class descriptions for user interface
CLASS_DESCRIPTIONS: Dict[str, str] = {
    '0': 'Adiastematic neumes (without staff lines)',
    '1': 'Early diastematic neumes (with primitive staff)',
    '2': 'Square notation (without staff lines)',
    '3': 'Square notation (with staff lines)',
    '4': 'Reject class (damaged, non-manuscript, or other)'
}

# Training parameters
NUM_FOLDS: int = 5
MIN_SAMPLES_PER_CLASS: int = 50
LEARNING_RATE: float = 0.0001
EPOCHS: int = 20

# Feature extraction parameters
FEATURE_NAMES: List[str] = [
    'staffline_confidence',
    'text_ratio',
    'top_margin',
    'bottom_margin',
    'left_margin',
    'right_margin'
]

# Confidence threshold for user disambiguation
CONFIDENCE_THRESHOLD: float = 0.7  # Lower this as model performance improves

# Augmentation parameters
AUGMENTATION_FACTOR: int = 3  # Number of augmented versions to create per original image

# Environment and performance settings
MAX_THREADS: int = 16  # Maximum number of threads for parallel processing
MEMORY_LIMIT_FACTOR: float = 0.8  # Fraction of available memory to use

# Function to reload configuration from environment variables
def load_from_env() -> None:
    """Update configuration from environment variables if present."""
    if 'MANUSCRIPT_BATCH_SIZE' in os.environ:
        global BATCH_SIZE
        BATCH_SIZE = int(os.environ['MANUSCRIPT_BATCH_SIZE'])
        
    if 'MANUSCRIPT_CONFIDENCE_THRESHOLD' in os.environ:
        global CONFIDENCE_THRESHOLD
        CONFIDENCE_THRESHOLD = float(os.environ['MANUSCRIPT_CONFIDENCE_THRESHOLD'])
        
    if 'MANUSCRIPT_DATASET_DIR' in os.environ:
        global DATASET_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, FEATURES_DB
        DATASET_DIR = os.environ['MANUSCRIPT_DATASET_DIR']
        TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
        VAL_DIR = os.path.join(DATASET_DIR, 'validation')
        TEST_DIR = os.path.join(DATASET_DIR, 'test')
        FEATURES_DB = os.path.join(DATASET_DIR, 'features_db.pkl')
        
    print("Configuration loaded.")

# Load from environment if running directly
if __name__ == "__main__":
    load_from_env()
    print("Current configuration:")
    for key, value in globals().items():
        if key.isupper() and not key.startswith('_'):
            print(f"{key} = {value}")