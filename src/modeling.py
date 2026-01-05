# src/modeling.py

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

# ==============================================================================
# SECTION 1: INTERACTION LAYER (IMPROVED with Residual Connection)
# ==============================================================================

class QarinaInteractionLayer(nn.Module):
    """
    Implementation of the Qarīna-Aware Interaction Layer (Section 5.2.2).
    """
    def __init__(self, hidden_size):
        super(QarinaInteractionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.scale = hidden_size ** -0.5

        # Optional: LayerNorm to stabilize the addition (Residual Add)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, context_states, metaphor_mask):
        """
        Returns:
            refined_context (Tensor): [Batch, Hidden_Dim]
            attn_weights (Tensor): [Batch, 1, Seq_Len]
        """
        # 1. Prepare Query (Q): Span Representation
        mask_expanded = metaphor_mask.unsqueeze(-1)
        sum_span = torch.sum(context_states * mask_expanded, dim=1)
        len_span = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        Q = (sum_span / len_span).unsqueeze(1) # [Batch, 1, Hidden_Dim]

        # 2. Prepare Key (K) & Value (V): Full Context
        K = context_states
        V = context_states

        # 3. Attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = torch.nn.functional.softmax(attention_scores, dim=-1)

        # 4. Weighted Context
        weighted_context = torch.matmul(attn_weights, V) # [Batch, 1, Hidden_Dim]

        # =======================================================
        # 🔥 CRITICAL IMPROVEMENT: RESIDUAL CONNECTION
        # =======================================================
        # Instead of replacing the span with context, we ADD them.
        # Output = Span (Q) + Context_Influence (weighted_context)
        # This ensures the model never loses the core meaning of the metaphor word.
        
        refined_rep = Q + weighted_context
        
        # Remove seq dimension -> [Batch, Hidden_Dim]
        refined_rep = refined_rep.squeeze(1)

        # Apply LayerNorm for stability
        refined_rep = self.layer_norm(refined_rep)

        return refined_rep, attn_weights


# ==============================================================================
# SECTION 2: FULL MULTI-TASK MODEL (Updated)
# ==============================================================================

class QuranMetaphorModel(nn.Module):
    def __init__(self, model_name, num_labels_map, dropout_prob=0.1):
        super(QuranMetaphorModel, self).__init__()
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.config.hidden_size
        
        # Interaction Layer
        self.interaction_layer = QarinaInteractionLayer(self.hidden_size)
        
        # Regularization
        self.dropout = nn.Dropout(dropout_prob)
        
        # Multi-Task Heads
        self.type_head = nn.Linear(self.hidden_size, num_labels_map['Type'])
        self.origin_head = nn.Linear(self.hidden_size, num_labels_map['Origin'])
        self.context_head = nn.Linear(self.hidden_size, num_labels_map['Context'])

    def forward(self, input_ids, attention_mask, metaphor_mask):
        # A. Encode
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        
        # B. Interaction (With Residual)
        refined_rep, attn_weights = self.interaction_layer(last_hidden_state, metaphor_mask)
        
        # Dropout
        final_rep = self.dropout(refined_rep)
        
        # C. Classify
        logits_type = self.type_head(final_rep)
        logits_origin = self.origin_head(final_rep)
        logits_context = self.context_head(final_rep)
        
        return logits_type, logits_origin, logits_context, attn_weights


# ==============================================================================
# SECTION 3: ABLATED MODEL (Unchanged)
# ==============================================================================
class QuranMetaphorModel_Ablated(nn.Module):
    def __init__(self, model_name, num_labels_map, dropout_prob=0.1):
        super(QuranMetaphorModel_Ablated, self).__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.hidden_size = self.config.hidden_size
        
        self.dropout = nn.Dropout(dropout_prob)
        
        self.type_head = nn.Linear(self.hidden_size, num_labels_map['Type'])
        self.origin_head = nn.Linear(self.hidden_size, num_labels_map['Origin'])
        self.context_head = nn.Linear(self.hidden_size, num_labels_map['Context'])

    def forward(self, input_ids, attention_mask, metaphor_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        
        mask_expanded = metaphor_mask.unsqueeze(-1)
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        span_representation = sum_embeddings / sum_mask
        
        pooled_output = self.dropout(span_representation)
        
        l_type = self.type_head(pooled_output)
        l_origin = self.origin_head(pooled_output)
        l_context = self.context_head(pooled_output)
        
        return l_type, l_origin, l_context, None
