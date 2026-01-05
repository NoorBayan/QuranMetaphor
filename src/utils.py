# src/utils.py

import torch
import numpy as np
import random
import os
from sklearn.utils.class_weight import compute_class_weight

# ==============================================================================
# 1. CONSTANTS & CONFIGURATION
# ==============================================================================

# Label Mappings (Aligned with Section 5.1 definitions)
# Used to convert string labels from CSV to Integers
LABEL_MAPS = {
    'Type':    {'Explicit': 0, 'Implicit': 1, 'Empty': 2, 'None': 2, 'Null': 2},
    'Origin':  {'Primary': 0, 'Derivative': 1, 'Empty': 2, 'None': 2, 'Null': 2},
    'Context': {'Candidate': 0, 'Implied': 1, 'Absolute': 2, 'Empty': 3, 'None': 3, 'Null': 3}
}

# Number of output neurons for each head
NUM_LABELS = {
    'Type': 3,    # Explicit, Implicit, Null
    'Origin': 3,  # Primary, Derivative, Null
    'Context': 4  # Candidate, Implied, Absolute, Null
}

# ==============================================================================
# 2. REPRODUCIBILITY HELPER
# ==============================================================================
def set_seed(seed=42):
    """
    Sets seeds for all random number generators to ensure reproducibility.
    (Compliance with Section 6.1 Experimental Protocol)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Force deterministic algorithms (may slow down training slightly but ensures exact results)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"🌱 Random Seed set to: {seed}")

# ==============================================================================
# 3. CLASS IMBALANCE HANDLING
# ==============================================================================
def compute_class_weights_for_loss(train_df, label_maps, device):
    """
    Calculates inverse frequency weights for each task to handle class imbalance.
    Returns a dictionary of tensors to be passed to CrossEntropyLoss.
    
    Args:
        train_df (pd.DataFrame): The training split dataframe.
        label_maps (dict): The global label mapping.
        device (torch.device): 'cuda' or 'cpu'.
        
    Returns:
        dict: {'Type': Tensor, 'Origin': Tensor, 'Context': Tensor}
    """
    weights_dict = {}
    
    for task in ['Type', 'Origin', 'Context']:
        # 1. Get String Labels & Map to Ints
        y_str = train_df[task].astype(str).str.strip()
        y_int = y_str.map(lambda x: label_maps[task].get(x, label_maps[task]['Empty'])).values
        
        # 2. Compute Balanced Weights using Sklearn
        # formula: n_samples / (n_classes * np.bincount(y))
        classes_present = np.unique(y_int)
        raw_weights = compute_class_weight(class_weight='balanced', classes=classes_present, y=y_int)
        
        # 3. Create Tensor of correct size (handling missing classes in mini-splits)
        # We need a tensor of size [Num_Classes] (e.g., 3 or 4)
        num_classes = len(set(label_maps[task].values())) # Count unique int values
        final_tensor = torch.ones(num_classes) # Default to 1.0
        
        for cls_idx, w in zip(classes_present, raw_weights):
            if cls_idx < num_classes:
                final_tensor[cls_idx] = w
        
        weights_dict[task] = final_tensor.to(device, dtype=torch.float)
        
    print(f"⚖️ Computed Class Weights for {list(weights_dict.keys())}")
    return weights_dict

# ==============================================================================
# 4. DEVICE HELPER
# ==============================================================================
def get_device():
    """Returns the appropriate PyTorch device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
