# src/data_engine.py

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class QuranMetaphorDataset(Dataset):
    """
    Dataset class responsible for processing Qur'anic text and handling 
    the methodology constraints (Surrogate Spans & Unsupervised Masking).
    """
    def __init__(self, df, tokenizer, max_len, label_maps):
        """
        Args:
            df (pd.DataFrame): Dataframe containing text and annotations.
            tokenizer (AutoTokenizer): Transformer tokenizer.
            max_len (int): Maximum sequence length (e.g., 128).
            label_maps (dict): Dictionary mapping labels to integers.
        """
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_maps = label_maps

    def __len__(self):
        return len(self.df)

    def get_span_mask(self, text, span, encoding, is_metaphor_span=False):
        """
        Generates a binary mask for the span.
        
        METHODOLOGY LOGIC:
        1. Surrogate Span (Section 5.1): If 'is_metaphor_span' is True but span is Empty,
           mask the [CLS] token (index 0) to provide a valid input (X, s).
        2. Leakage Prevention: If 'is_metaphor_span' is False (Qarina) and span is Empty,
           return a zero mask (target is absent).
        """
        mask = [0] * self.max_len

        # --- Case 1: Handle Empty/Null Spans ---
        # Checks for various forms of 'empty' in pandas
        if span in ['Empty', 'None', 'nan', '3', '', None]:
            if is_metaphor_span:
                # Methodology: Assign Surrogate Span ([CLS] at index 0)
                mask[0] = 1.0
            return torch.tensor(mask, dtype=torch.float)

        # --- Case 2: Find Span in Text ---
        # Note: Ideally, text should be normalized. Here we rely on simple find.
        start_char = text.find(span)

        # Fallback if span not found due to normalization issues
        if start_char == -1:
            if is_metaphor_span:
                # Fallback to Surrogate Span to prevent crash
                mask[0] = 1.0
            return torch.tensor(mask, dtype=torch.float)

        end_char = start_char + len(span)
        
        # Convert char offsets to token indices
        token_start = encoding.char_to_token(0, start_char)
        token_end = encoding.char_to_token(0, end_char - 1)

        if token_start is not None and token_end is not None:
            for i in range(token_start, token_end + 1):
                if i < self.max_len:
                    mask[i] = 1.0
        else:
            # Tokenization boundary mismatch fallback
            if is_metaphor_span:
                mask[0] = 1.0

        return torch.tensor(mask, dtype=torch.float)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['Text_Ayah']).strip()
        
        # 1. Tokenization
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True
        )

        # 2. Generate Masks (Methodology-Aware)
        
        # Metaphor Span: INPUT to the model (s)
        metaphor_mask = self.get_span_mask(
            text, str(row['Metaphor_Span']).strip(), encoding, is_metaphor_span=True
        )

        # Qarina Span: TARGET for analysis (Not Input)
        target_qarina_mask = self.get_span_mask(
            text, str(row['Qarina_Span']).strip(), encoding, is_metaphor_span=False
        )

        # 3. Label Processing
        # Safe lookup using .get() with a default value for safety
        def get_label(col_name, val):
            # Assumes label_maps structure matches Phase 1 logic
            # Default to '2' (Null/Empty) if not found, usually appropriate for 3-class tasks
            # You should adjust default based on your label map logic in utils.py
            val = str(val).strip()
            return self.label_maps[col_name].get(val, 2) 

        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
            
            # The Critical Input Feature
            'metaphor_mask': metaphor_mask,
            
            # The Evaluation Target (Zeroed out if Empty)
            'target_qarina_mask': target_qarina_mask,

            'labels': {
                'type': torch.tensor(get_label('Type', row['Type']), dtype=torch.long),
                'origin': torch.tensor(get_label('Origin', row['Origin']), dtype=torch.long),
                # Using 3 as default for context if mapping is {..., 'Absolute': 2, 'Null': 3}
                'context': torch.tensor(self.label_maps['Context'].get(str(row['Context']).strip(), 3), dtype=torch.long)
            }
        }


def load_and_preprocess_data(file_path, encoding='utf-16'):
    """
    Helper function to load CSV and perform basic cleaning.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"dataset file not found at: {file_path}")

    df = pd.read_csv(file_path, sep='\t', encoding=encoding)
    
    # Basic cleaning
    df.fillna('Empty', inplace=True)
    
    text_cols = ['Text_Ayah', 'Metaphor_Span', 'Qarina_Span', 'Type', 'Origin', 'Context']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            
    return df
