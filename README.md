# Named Entity Recognition using Transformers

This repository contains an end-to-end implementation of a **Named Entity Recognition (NER)** system using **Transformer-based models** such as **BERT / DistilBERT**.  
The project demonstrates **sequence labeling** by fine-tuning a pre-trained transformer for **token-level entity classification** on a labeled NER dataset.

---

## Project Objectives

### Objective 1 – Theory (Medium Article)

- Explain **sequence labeling tasks**
- Explain **Named Entity Recognition (NER)**
- Discuss **traditional NER approaches**
  - Rule-based systems
  - Statistical models (HMM, CRF)
- Explain why **Transformer-based models (BERT)** outperform traditional methods

A detailed Medium article covering these concepts is written separately (link provided below).


### Objective 2 – Implementation

- Use the **given labeled NER dataset**
- Convert raw text into **token–label pairs**
- Fine-tune a **pre-trained Transformer model**
- Perform **token-level classification**
- Evaluate using **Precision, Recall, and F1-score**
- Provide a clean, reproducible **Jupyter Notebook**

---

## What is Named Entity Recognition (NER)?

Named Entity Recognition is a sequence labeling task in Natural Language Processing where each token in a sentence is assigned an entity label.

### Example

```text
Sentence: John works in New York
Tokens:   ["John", "works", "in", "New", "York"]
Labels:   ["B-PER", "O", "O", "B-LOC", "I-LOC"]
```

### Entity Types

- **PER** – Person  
- **LOC** – Location  
- **ORG** – Organization  
- **DATE** – Date  
- **O** – Non-entity  

---

## Dataset

- File: `ner.csv`
- Format: CSV
- Contains:
  - Raw sentences
  - Word-level NER tags in **BIO format**

### Preprocessing Steps

1. Convert stringified lists to Python lists
2. Tokenize sentences into words
3. Remove misaligned rows
4. Ensure correct token–label alignment


This ensures safe and accurate training for token-level classification.

---

## Model Architecture

- **Base Model:** bert-base-cased / distilbert-base-cased
- **Framework:** Hugging Face Transformers + PyTorch

### Architecture Flow

```
Input Sentence
   ↓
Tokenizer (subword tokenization)
   ↓
Transformer Encoder
   ↓
Token Classification Head
   ↓
NER Labels
```

---

## Training Configuration

- **Loss Function:** CrossEntropyLoss (token-level)
- **Optimizer:** AdamW (handled internally by Trainer)
- **Training Method:** Fine-tuning a pre-trained model
- **Epochs:** 1 (optimized for CPU training)
- **Max Sequence Length:** 64
- **Batch Size:** 8
- **Evaluation Metric:** F1-score (using seqeval)

---

## Model Evaluation

The fine-tuned model is evaluated on a held-out validation set using:

- Precision  
- Recall  
- F1-score  

A detailed classification report is generated for each entity type.

---

## Optional Inference Demo

The notebook includes an optional inference cell to test the trained model on custom sentences.

```text
Input:  John works at Google in New York
Output: [(John, B-PER), (Google, B-ORG), (New, B-LOC), (York, I-LOC)]
```

---

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Accelerate
- Pandas
- Scikit-learn
- SeqEval
- Jupyter Notebook

---

## Repository Structure

```
NER-Transformer-BERT/
│
├── ner_transformer.ipynb
├── ner.csv
├── requirements.txt
├── README.md
└── .gitignore
```

Model checkpoints are intentionally excluded due to size constraints.

---

## How to Run Locally


1. Create virtual environment

  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```

2. Install dependencies

  ```bash
  pip install -r requirements.txt
  ```

3. Launch Jupyter Notebook

  ```bash
  jupyter notebook
  ```

  Run the notebook.

---

## Medium Article

The Medium article explains:

- Sequence labeling
- Traditional NER methods
- Transformer-based NER
- Fine-tuning BERT for NER

```bash
ADD_YOUR_MEDIUM_LINK_HERE
```

---

## Submission Guidelines

1. Host this repository on GitHub
2. Publish the Medium article
3. Share both links in the #ai Discord channel
4. Use the hashtag #cl-ai-nertransformer

---

## Notes

- Model checkpoints are excluded to keep the repository lightweight.
- Training was performed on CPU, so epochs and sequence length were optimized.
- The notebook is fully reproducible using the provided requirements.txt.
- DistilBERT was used for faster fine-tuning while maintaining performance.

---

## Conclusion

This project demonstrates how Transformer-based models can be effectively fine-tuned for Named Entity Recognition, a core sequence labeling task in NLP. By leveraging pre-trained language models, the system achieves strong performance without manual feature engineering.

---

## License

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files to deal in the Software without restriction.


---

By Jairaj R.
