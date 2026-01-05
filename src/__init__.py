# src/__init__.py

# We export the main classes and functions to simplify imports
# Instead of writing: from src.data_engine import QuranMetaphorDataset
# You can write: from src import QuranMetaphorDataset

from .data_engine import QuranMetaphorDataset, load_and_preprocess_data
