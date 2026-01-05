# src/modeling.py

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

# ==============================================================================
# SECTION 1: INTERACTION LAYER (The Core Novelty)
# ==============================================================================

class QarinaInteractionLayer(nn.Module):
    """
    Implementation of the Qarīna-Aware Interaction Layer (Section 5.2.2).
    
    Functionality:
    - Computes Span-based Cross-Attention.
    - Query (Q): Derived from the Metaphor Span (s) via Mean Pooling.
    - Key (K) & Value (V): Derived from the Full Context (X).
    - Unsupervised: No labels are used to guide this attention; it learns implicitly.
    """
    def __init__(self, hidden_size):
        super(QarinaInteractionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.scale = hidden_size ** -0.5  # Scaling factor (1 / sqrt(d_k))

    def forward(self, context_states, metaphor_mask):
        """
        Args:
            context_states (Tensor): [Batch, Seq_Len, Hidden_Dim] - Encoder Output
            metaphor_mask (Tensor): [Batch, Seq_Len] - Binary mask for span 's'
            
        Returns:
            refined_context (Tensor): [Batch, Hidden_Dim] - Context weighted by rhetorical relevance.
            attn_weights (Tensor): [Batch, 1, Seq_Len] - Attention distribution for explainability.
        """
        # 1. Prepare Query (Q): Span Representation
        # -------------------------------------------------------
        # Expand mask for multiplication: [Batch, Seq_Len, 1]
        mask_expanded = metaphor_mask.unsqueeze(-1)
        
        # Aggregate span embeddings (Sum)
        sum_span = torch.sum(context_states * mask_expanded, dim=1)
        
        # Count tokens in span (avoid division by zero with clamp)
        len_span = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        
        # Compute Mean: [Batch, Hidden_Dim] -> Reshape to [Batch, 1, Hidden_Dim] for Q
        Q = (sum_span / len_span).unsqueeze(1)

        # 2. Prepare Key (K) & Value (V): Full Context
        # -------------------------------------------------------
        K = context_states  # [Batch, Seq_Len, Hidden_Dim]
        V = context_states  # [Batch, Seq_Len, Hidden_Dim]

        # 3. Scaled Dot-Product Attention (Eq. 1)
        # -------------------------------------------------------
        # Scores = (Q * K^T) / sqrt(d)
        # [B, 1, H] * [B, H, L] -> [B, 1, L]
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply Softmax to get probabilities
        attn_weights = torch.nn.functional.softmax(attention_scores, dim=-1)

        # 4. Weighted Aggregation
        # -------------------------------------------------------
        # Context = Weights * V
        # [B, 1, L] * [B, L, H] -> [B, 1, H]
        refined_context = torch.matmul(attn_weights, V)

        # Remove the singleton sequence dimension -> [Batch, Hidden_Dim]
        return refined_context.squeeze(1), attn_weights


# ==============================================================================
# SECTION 2: FULL MULTI-TASK MODEL (Phase 2)
# ==============================================================================

class QuranMetaphorModel(nn.Module):
    """
    The 'Deep Rhetoric' Multi-Task Architecture (Figure 2 in Paper).
    Components:
      1. Shared Pre-trained Encoder (e.g., MARBERT)
      2. Qarīna-Aware Interaction Layer
      3. Three Independent Classification Heads (Type, Origin, Context)
    """
    def __init__(self, model_name, num_labels_map, dropout_prob=0.1):
        """
        Args:
            model_name (str): HuggingFace model hub path (e.g., 'UBC-NLP/MARBERTv2')
            num_labels_map (dict): Dictionary defining output dim for each task.
                                   e.g. {'Type': 3, 'Origin': 3, 'Context': 4}
            dropout_prob (float): Dropout rate for regularization.
        """
        super(QuranMetaphorModel, self).__init__()
        
        # Load Config & Encoder
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.config.hidden_size
        
        # Interaction Layer
        self.interaction_layer = QarinaInteractionLayer(self.hidden_size)
        
        # Regularization
        self.dropout = nn.Dropout(dropout_prob)
        
        # Multi-Task Heads
        # Note: We use keys matching the dataset labels: 'Type', 'Origin', 'Context'
        self.type_head = nn.Linear(self.hidden_size, num_labels_map['Type'])
        self.origin_head = nn.Linear(self.hidden_size, num_labels_map['Origin'])
        self.context_head = nn.Linear(self.hidden_size, num_labels_map['Context'])

    def forward(self, input_ids, attention_mask, metaphor_mask):
        """
        Forward pass complying with (X, s) formulation.
        
        Args:
            input_ids, attention_mask: Represent X (Context)
            metaphor_mask: Represents s (Span)
            
        Returns:
            logits_type, logits_origin, logits_context: Prediction scores
            attn_weights: For interpretability/analysis
        """
        # A. Encode Context
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        
        # B. Interaction (Qarina-Awareness)
        refined_rep, attn_weights = self.interaction_layer(last_hidden_state, metaphor_mask)
        
        # Dropout
        final_rep = self.dropout(refined_rep)
        
        # C. Classification
        logits_type = self.type_head(final_rep)
        logits_origin = self.origin_head(final_rep)
        logits_context = self.context_head(final_rep)
        
        return logits_type, logits_origin, logits_context, attn_weights


# ==============================================================================
# SECTION 3: ABLATED MODEL (Phase 4 - Ablation Study)
# ==============================================================================

class QuranMetaphorModel_Ablated(nn.Module):
    """
    Baseline Model for Ablation Study (Section 7.3).
    
    DIFFERENCE:
    - Removes QarinaInteractionLayer.
    - Uses simple Mean Pooling of the metaphor span.
    - Architecture otherwise identical (Same encoder, same heads).
    """
    def __init__(self, model_name, num_labels_map, dropout_prob=0.1):
        super(QuranMetaphorModel_Ablated, self).__init__()
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.config.hidden_size
        
        self.dropout = nn.Dropout(dropout_prob)
        
        self.type_head = nn.Linear(self.hidden_size, num_labels_map['Type'])
        self.origin_head = nn.Linear(self.hidden_size, num_labels_map['Origin'])
        self.context_head = nn.Linear(self.hidden_size, num_labels_map['Context'])

    def forward(self, input_ids, attention_mask, metaphor_mask):
        # A. Encode
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        
        # B. Mean Pooling (Instead of Interaction)
        # ----------------------------------------
        mask_expanded = metaphor_mask.unsqueeze(-1)
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        
        # Pure span representation (no context interaction)
        span_representation = sum_embeddings / sum_mask
        
        # Dropout
        pooled_output = self.dropout(span_representation)
        
        # C. Classification
        l_type = self.type_head(pooled_output)
        l_origin = self.origin_head(pooled_output)
        l_context = self.context_head(pooled_output)
        
        # Return None as 4th element to match Full Model signature (simplifies training loop)
        return l_type, l_origin, l_context, None
