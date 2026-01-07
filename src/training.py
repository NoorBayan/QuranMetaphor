# src/training.py

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, f1_score
from tqdm.auto import tqdm
import copy
import numpy as np

# ==============================================================================
# HELPER: STRICT METRICS (Section 6.4)
# ==============================================================================
def calculate_strict_metrics(true_labels, pred_labels, label_map):
    """
    Computes Macro-F1 over POSITIVE classes only (Excluding 'Null').
    
    Args:
        true_labels (list): Ground truth.
        pred_labels (list): Predictions.
        label_map (dict): Mapping e.g. {'Explicit': 0, 'Null': 2}
        
    Returns:
        dict: Classification report.
        float: Strict Macro-F1 score.
    """
    # Inverse map: Index -> Name
    inv_map = {v: k for k, v in label_map.items()}
    labels_idx = list(inv_map.keys())
    target_names = list(inv_map.values())

    # 1. Identify Null class index
    # We look for keywords like 'Empty', 'None', 'Null'
    null_indices = [v for k, v in label_map.items() if k in ['Empty', 'None', 'Null']]
    null_idx = null_indices[0] if null_indices else -1

    # 2. Define Positive Classes (All except Null)
    positive_labels = [idx for idx in labels_idx if idx != null_idx]

    # 3. Compute Macro F1 on POSITIVE classes only
    strict_macro_f1 = f1_score(
        true_labels,
        pred_labels,
        labels=positive_labels,
        average='macro',
        zero_division=0
    )

    # 4. Generate Full Report for logging
    report = classification_report(
        true_labels,
        pred_labels,
        labels=labels_idx,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )

    return report, strict_macro_f1


# ==============================================================================
# CLASS: METAPHOR TRAINER
# ==============================================================================
class MetaphorTrainer:
    """
    Handles the training and evaluation loop compliant with Phase 3 methodology.
    """
    def __init__(self, model, device, class_weights_dict=None):
        """
        Args:
            model (nn.Module): The initialized PyTorch model (Full or Ablated).
            device (torch.device): 'cuda' or 'cpu'.
            class_weights_dict (dict): Dictionary containing tensors for loss weighting.
                                     Keys: 'Type', 'Origin', 'Context'.
                                     Values: torch.Tensor of weights.
        """
        self.model = model.to(device)
        self.device = device
        self.weights = class_weights_dict if class_weights_dict else {}
        
        # Initialize Loss Functions (Weighted)
        # Defaults to None (Uniform weight) if specific weights not provided
        self.crit_type = nn.CrossEntropyLoss(weight=self.weights.get('Type'))
        self.crit_origin = nn.CrossEntropyLoss(weight=self.weights.get('Origin'))
        self.crit_context = nn.CrossEntropyLoss(weight=self.weights.get('Context'))

    def train(self, train_loader, val_loader, epochs=15, lr=3e-5, weight_decay=0.1):
        """
        Main training loop.
        """
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Scheduler for stability (Warmup 10%)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

        best_val_f1 = 0.0
        best_model_wts = copy.deepcopy(self.model.state_dict())
        history = []

        print(f"🔥 Starting Training on {self.device}...")

        for epoch in range(epochs):
            # --- Training Phase ---
            self.model.train()
            total_loss = 0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            
            for batch in progress_bar:
                # Move batch to device
                ids = batch['input_ids'].to(self.device)
                mask = batch['attention_mask'].to(self.device)
                meta_mask = batch['metaphor_mask'].to(self.device) # Input 's'
                
                l_type = batch['labels']['type'].to(self.device)
                l_origin = batch['labels']['origin'].to(self.device)
                l_context = batch['labels']['context'].to(self.device)

                optimizer.zero_grad()

                # Forward Pass
                # Note: Returns 4 values (attn is 4th), we ignore attn here
                out_type, out_origin, out_context, _ = self.model(ids, mask, meta_mask)

                # Loss Calculation (Summation as per Eq. 2)
                loss = self.crit_type(out_type, l_type) + \
                       self.crit_origin(out_origin, l_origin) + \
                       self.crit_context(out_context, l_context)

                loss.backward()

                # Gradient Clipping (Stability)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})

            # --- Validation Phase ---
            avg_loss = total_loss / len(train_loader)
            val_results = self.evaluate(val_loader, label_maps=val_loader.dataset.label_maps)
            curr_f1 = val_results['Avg_Strict_F1']

            print(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f} | Val Strict F1={curr_f1:.4f}")

            # Checkpointing
            if curr_f1 > best_val_f1:
                best_val_f1 = curr_f1
                best_model_wts = copy.deepcopy(self.model.state_dict())
                print("  💾 Checkpoint Saved (Best Metric)")
            
            history.append({'epoch': epoch+1, 'loss': avg_loss, 'val_f1': curr_f1})

        print(f"\n✅ Training Complete. Best Validation F1: {best_val_f1:.4f}")
        
        # Load best weights
        self.model.load_state_dict(best_model_wts)
        return self.model, history

    def evaluate(self, loader, label_maps):
        """
        Runs evaluation with LOGICAL CONSISTENCY ENFORCEMENT.
        """
        self.model.eval()
        
        preds = {'Type': [], 'Origin': [], 'Context': []}
        trues = {'Type': [], 'Origin': [], 'Context': []}

        # 1. Get Null Indices from Label Maps to apply logic
        # Assuming structure: {'Explicit':0, ..., 'Null': 2}
        type_null_idx = label_maps['Type'].get('Null', label_maps['Type'].get('Empty', 2))
        origin_null_idx = label_maps['Origin'].get('Null', label_maps['Origin'].get('Empty', 2))
        context_null_idx = label_maps['Context'].get('Null', label_maps['Context'].get('Empty', 3))

        with torch.no_grad():
            for batch in loader:
                ids = batch['input_ids'].to(self.device)
                mask = batch['attention_mask'].to(self.device)
                meta_mask = batch['metaphor_mask'].to(self.device)

                l_type, l_origin, l_context, _ = self.model(ids, mask, meta_mask)

                # Get Raw Predictions
                p_type = torch.argmax(l_type, dim=1)
                p_origin = torch.argmax(l_origin, dim=1)
                p_context = torch.argmax(l_context, dim=1)

                # ===================================================
                # 🔥 LOGICAL CONSISTENCY FIX (POST-PROCESSING)
                # ===================================================
                # Rule: If Type is Null, force Origin and Context to be Null
                
                # Create a mask where Type prediction is Null
                is_null_mask = (p_type == type_null_idx)
                
                # Override inconsistent predictions
                p_origin[is_null_mask] = origin_null_idx
                p_context[is_null_mask] = context_null_idx
                # ===================================================

                # Save finalized predictions
                preds['Type'].extend(p_type.cpu().numpy())
                preds['Origin'].extend(p_origin.cpu().numpy())
                preds['Context'].extend(p_context.cpu().numpy())

                trues['Type'].extend(batch['labels']['type'].cpu().numpy())
                trues['Origin'].extend(batch['labels']['origin'].cpu().numpy())
                trues['Context'].extend(batch['labels']['context'].cpu().numpy())

        # ... (Rest of the function remains the same: Compute Metrics) ...
        
        # Compute Metrics
        res_type, f1_type = calculate_strict_metrics(trues['Type'], preds['Type'], label_maps['Type'])
        res_origin, f1_origin = calculate_strict_metrics(trues['Origin'], preds['Origin'], label_maps['Origin'])
        res_context, f1_context = calculate_strict_metrics(trues['Context'], preds['Context'], label_maps['Context'])

        avg_strict_f1 = (f1_type + f1_origin + f1_context) / 3.0

        return {
            'Type': res_type, 
            'Origin': res_origin, 
            'Context': res_context,
            'Strict_F1_Type': f1_type,
            'Strict_F1_Origin': f1_origin,
            'Strict_F1_Context': f1_context,
            'Avg_Strict_F1': avg_strict_f1
        }
