# User feedback collection
import os
import json
import datetime
from typing import Dict, Any, Optional
from pathlib import Path

# Import from your project modules
import config

def save_user_feedback(
    img_path: str,
    selected_class: str,
    feedback_db_path: str = 'user_feedback.json',
    additional_info: Optional[Dict[str, Any]] = None
) -> bool:
    """Save user feedback for later model improvement.
    
    Args:
        img_path: Path to the image that was classified
        selected_class: The class selected by the user
        feedback_db_path: Path to the feedback database file
        additional_info: Optional additional information to store
        
    Returns:
        Boolean indicating if the feedback was successfully saved
    """
    # Load existing feedback database
    feedback_db = {}
    if os.path.exists(feedback_db_path):
        with open(feedback_db_path, 'r') as f:
            try:
                feedback_db = json.load(f)
            except json.JSONDecodeError:
                feedback_db = {}
    
    # Add new feedback
    feedback_entry = {
        'selected_class': selected_class,
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    # Add any additional information if provided
    if additional_info:
        feedback_entry.update(additional_info)
        
    feedback_db[img_path] = feedback_entry
    
    # Save updated database
    try:
        with open(feedback_db_path, 'w') as f:
            json.dump(feedback_db, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving feedback: {e}")
        return False