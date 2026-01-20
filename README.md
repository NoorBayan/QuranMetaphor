# QuranMetaphor: Deep Rhetoric Learning for Qur'anic *Istiʿāra*

[![Part of Project Borhan](https://img.shields.io/badge/Project-Borhan-005b96?style=for-the-badge&logo=bookstack)](https://github.com/YourUsername/BorhanProject)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-blue?style=for-the-badge)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](./LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1n9dCxSFWBmg1L-EPZ2MRqRyGCXWpuev7/view?usp=sharing)

**QuranMetaphor** is the reference implementation for the **Metaphor Modeling Module** of the **Borhan Project (برهان)**. It introduces a "Deep Rhetoric" framework that operationalizes Classical Arabic *Balāgha* (Rhetoric) into a computational ontology, enabling machines to move beyond literal processing toward **Aesthetic Sensing**.

---

## 🏛️ Context: The Borhan Project (مشروع برهان)

> *"Rhetoric is the engineering of influence."*

This repository is part of **Borhan (Borhan Rhetoric Extraction)**, a paradigm-shifting initiative establishing the field of **Computational Rhetoric** in Qur'anic Studies. 

### 1. The Vision: From Literal to Aesthetic
Current NLP models are "rhetorically blind." When algorithms process a verse like *"Shall we believe as the fools believed?"*, they see syntax but miss the **sarcasm**, **tone**, and **social layering**. Borhan transforms fluid literary taste into a **solid cognitive ontology**, granting digital applications an "Emotional Intelligence" parallel to their linguistic capabilities.

### 2. The Innovation: A "Holistic Forest" of Meaning
Unlike traditional approaches that treat rhetorical devices as isolated trees, Borhan maps the "Forest of Meanings" through:
*   **Granular Modeling:** Deconstructing a single image into >20 attributes (Type, Origin, Context, Tone).
*   **Pragmatic & Affective Analysis:** Detecting the *speech act* (e.g., Deprecation vs. Mitigation) behind the metaphor.
*   **Conceptual Network Mapping:** Linking images (e.g., Trade, Scales) to reveal the *Transactional Logic* of the Qur'anic worldview.
*   **Accommodating Semantic Complexity:** A hybrid architecture that rejects binary classifications in favor of **Multi-Maqasid (Multi-Intent)** recognition.

### 3. Academic Reliability
This methodology is documented in **3 in-depth research papers** currently under review at Q1 journals (SAGE, IEEE, Elsevier), ensuring that our "Qarina-Aware" algorithms meet the highest global academic standards.

---

## 🔬 The Specific Solution: QuranMetaphor

This repository implements the **Qarina-Aware Modeling** (Contextual Clue Awareness) described in the Borhan methodology. It tackles the challenge of **Metaphor (*Istiʿāra*)** not as a detection task, but as a hierarchical inference problem.

### Methodology
We define the problem as a set of interdependent classification tasks, translating rhetorical constraints into a **Multi-Task Learning (MTL)** architecture:

1.  **Type ($T_{type}$):** Distinguishes *Taṣrīḥiyya* (Explicit) vs. *Makniyya* (Implicit).
2.  **Origin ($T_{origin}$):** Distinguishes Primary sensory metaphors vs. Derivative associations.
3.  **Functional Context ($T_{context}$):** Analyzes whether the metaphor is extended (*Murashaha*), absolute, or abstract.

### Model Architecture
The model uses a hard parameter-sharing approach with a specialized **Qarīna-Aware Interaction Layer**—an unsupervised mechanism that mimics the human cognitive process of scanning context for "blocking indicators" (Qarīna) to resolve ambiguity.

```mermaid
graph TD
    subgraph Input_Layer["Input Layer"]
        A["Tokenized Verse (X)"]
        B["Candidate Span (s)"]
    end

    A --> C["Arabic Encoder (MARBERT / CamelBERT)"]
    B --> C

    C --> D["Hidden States"]

    subgraph Borhan_Core["Borhan Deep Rhetoric Core"]
        D --> E["Qarīna-Aware Interaction Layer"]
        E -->|Unsupervised Context Fusion| F["Refined Rhetorical Embeddings"]
    end

    subgraph Multi_Task["Multi-Task Heads"]
        F --> G["Head 1: Type"]
        F --> H["Head 2: Origin"]
        F --> I["Head 3: Context"]
    end
```


```

## ⚖️ License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
