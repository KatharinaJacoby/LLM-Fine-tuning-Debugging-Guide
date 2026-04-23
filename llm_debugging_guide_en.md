# LLM Fine-tuning Debugging Guide

## 🎯 Project

Development of a medical LLM for diagnostic support using T5 fine-tuning for educational purposes only.

This guide serves as a structured, end-to-end case study for when things get messy — demonstrating how known principles are applied to navigate a concrete, real-world fine-tuning problem. Documenting bugs and debugging methods is really important and sometimes can help the community even more than polished successful projects.

---

## 📋 Initial Situation

### Original Code (Working but Limited)

```python
import pandas as pd
import transformers
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

data = [
    {"input": "Symptoms: Fever, cough. CRP: 67. Imaging: Basal infiltrate right. What is the most likely diagnosis?", "output": "Pneumonia"},
    {"input": "Symptoms: Dyspnea, leg swelling left. D-dimer elevated. What is the most likely diagnosis?", "output": "Pulmonary embolism"},
    {"input": "Symptoms: Fatigue, pallor. Hb: low. What is the most likely diagnosis?", "output": "Anemia"},
    {"input": "Symptoms: Chest pain, troponin high, ECG ST elevation. What is the most likely diagnosis?", "output": "Myocardial infarction"},
    {"input": "Symptoms: Polyuria, polydipsia, BG 320 mg/dl. What is the most likely diagnosis?", "output": "Diabetes mellitus"}
]

data = pd.DataFrame(data)
tokenizer = T5Tokenizer.from_pretrained("t5-small")

def tokenize(example):
    input_enc = tokenizer(example["input"], truncation=True, padding="max_length", max_length=128)
    output_enc = tokenizer(example["output"], truncation=True, padding="max_length", max_length=32)
    input_enc["labels"] = output_enc["input_ids"]
    return input_enc

dataset = Dataset.from_pandas(data)
tokenized_dataset = dataset.map(tokenize)
model = T5ForConditionalGeneration.from_pretrained("t5-small")

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=20,
    logging_steps=1,
    save_strategy="no",
    report_to="none"
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()

def predict(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).input_ids
    outputs = model.generate(inputs, max_length=32)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

test_prompt = "Symptoms: Shortness of breath, fever, CRP 90, X-ray: Right infiltrate. What is the most likely diagnosis?"
print("Answer:", predict(test_prompt))
```

### First Results (Problematic but Functional)

| Metric | Value |
|--------|-------|
| Output | "Pneumonia. DD: Pneumonia, Pneumonia" (repetitive) |
| Loss | 8.78 → 0.43 (very good) |
| Problem | Repetitive/incorrect differential diagnoses |

---

## 🚨 Problem Phase 1: Structural Improvement Leads to "True" Bug

### Attempt: Implement Extended Features

**Goal:** 100 examples, validation split, better output structure

**Changes:**
- Dataset expanded to 100 examples
- Structured DD output: "Diagnosis: X | DD: Y, Z, W"
- Train/validation split (80/20)
- `as_target_tokenizer()` → `text_target` (deprecated fix)
- `tokenizer` → `processing_class` parameter

### Problem: "True" Bug

```
# Expected: "Pneumonia. DD: Bronchitis, Pleuritis"
# Actual: "True"
```

**Symptoms:**
- All outputs only "True"
- Model behaves like binary classifier
- Missing keys warning: `embed_tokens.weight`, `lm_head.weight`

---

## 🔍 Debugging Phase 1: Systematic Problem Identification

### Step 1: Parameter Instability Hypothesis

**Observation:** Multiple deprecated/new parameters changed simultaneously
- `evaluation_strategy` → TypeError
- `processing_class` vs `tokenizer`
- `text_target` vs `as_target_tokenizer()`

**Insight:** "New parameters are unstable, old parameters work better" turned out to be a detour. The real problems (missing task prefix, string columns in dataset) had nothing to do with API deprecation. You reach root causes faster by focusing on the model's actual behavior rather than library warning messages.

### Step 2: Stepwise Regression

**Strategy:** Change one variable at a time

**Test 1: `as_target_tokenizer()` Fix**
```python
# Back to deprecated but working method
with tokenizer.as_target_tokenizer():
    output_enc = tokenizer(example["output"], ...)
```
**Result:** "rmelkinese" (corrupted, but no longer "True")

**Test 2: Original vs Fix Comparison**
**Result:** Both times "rmelkinese" → Problem lies elsewhere

---

## 🧹 Debugging Phase 2: Fresh Environment Strategy

### Step 3: Clean Slate Approach

**Decision:** Fresh notebook, back to working baseline

**Baseline Test (5 examples, original code):**
```python
# Minimal test for root cause isolation
data = [original 5 examples without DD]
```
**Result:** "What is the most likely diagnosis?" (input echo)

---

## 🔬 Debugging Phase 3: Pipeline Diagnosis

### Step 4: Labels Debug

**Check:** Are labels correctly tokenized?
```python
print("Sample tokenized data:")
print(f"Labels: {tokenized_dataset[0]['labels'][:10]}")
print(f"Decoded Labels: {tokenizer.decode(tokenized_dataset[0]['labels'])}")
```
**Result:** ✅ Labels perfect: "Pneumonia</s><pad>..."

### Step 5: Attention Mask Debug

**Check:** Does attention mechanism work?
```python
inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128)
print(f"Attention mask: {inputs.attention_mask}")
print(f"Attention mask sum: {inputs.attention_mask[0].sum()}")
```
**Result:** ✅ Attention perfect: 36/36 tokens attended

### Step 6: EOS/PAD Token Debug

**Check:** Token handling correct?
```python
print(f"PAD token: '{tokenizer.pad_token}' -> ID: {tokenizer.pad_token_id}")
print(f"EOS token: '{tokenizer.eos_token}' -> ID: {tokenizer.eos_token_id}")
```
**Result:** ✅ Token setup correct, but generation produces input echo

---

## 🚨 Problem Phase 2: DataCollator Crash

### Step 7: Label-Training-Pipeline Debug

**Deeper Test:** What happens during training?

**CRASH:**
```
ValueError: Unable to create tensor... Perhaps your features (`input` in this case) have excessive nesting
```

**Root Cause: String Features in Dataset**

**Problem:** DataCollator cannot tensorize all features
```python
tokenized_dataset.features = {
    "input": "string",      # ❌ DataCollator crash
    "output": "string",     # ❌ DataCollator crash
    "input_ids": "tensor",  # ✅ OK
    "labels": "tensor"      # ✅ OK
}
```

**Fix: Remove String Features**
```python
tokenized_dataset = tokenized_dataset.remove_columns(["input", "output"])
```
**Result:** Training runs, but output still wrong

---

## 🔍 Debugging Phase 4: T5-Specific Problems

### Step 8: T5 Training Mode Check

**Check:** Does T5 understand our task?

**DISCOVERY:** T5 has task-specific parameters:
```python
model.config.task_specific_params = {
    'summarization': {'prefix': 'summarize: '},
    'translation_en_to_de': {'prefix': 'translate English to German: '},
    ...
}
```
**Problem:** T5 doesn't understand what to do without a task prefix!

### Step 9: Task Prefix Implementation

```python
def tokenize_with_task_prefix(example):
    task_prefixed_input = f"medical diagnosis: {example['input']}"
    input_enc = tokenizer(task_prefixed_input, truncation=True, padding="max_length", max_length=128)
    output_enc = tokenizer(example["output"], truncation=True, padding="max_length", max_length=32)
    input_enc["labels"] = output_enc["input_ids"]
    return input_enc
```
**Result:** Input echo stops, but only empty outputs remain

---

## 🚨 Problem Phase 3: PAD Token Loop

### Step 10: Generation Mechanism Debug

**Problem:** Model generates only PAD tokens `[0,0,0,...]`

**Deep Debug:**
```python
# Raw token analysis
outputs = model.generate(inputs, max_length=32, do_sample=False)
print(f"Raw tokens: {outputs[0]}")
# Result: [0, 0, 0, 0, 0, 0, ...]
```

**Hypothesis:** Training Volume vs Decoder Mechanism

**Discussion:**
- Are 10 epochs too few for task prefix learning?
- Or is decoder-start mechanism broken?

### Step 11: A/B Test Strategy

**Test 1:** Continue Training (+20 epochs)  
**Test 2:** Fresh Training (30 epochs from scratch)

| Test | Loss | Output |
|------|------|--------|
| Continue Training | 2.0 → 0.15-0.30 | "Morbus Morbus Morbus..." ✅ (medical terms, but repetitive) |
| Fresh Training | 10.1 → 0.30-0.85 | "" (empty, PAD tokens) |

**Insight:** It's likely an artifact of optimizer state, cached gradients, or learning rate scheduling. Sometimes continuing a corrupted training run is worse than starting fresh.

---

## 🎯 Breakthrough Phase: Generation Parameter Optimization

### Step 12: Improved Generation Parameters

**Problem:** Repetitive output ("Morbus Morbus Morbus...")

**Solution: Advanced generation parameters**
```python
def predict_improved(prompt):
    prefixed_prompt = f"medical diagnosis: {prompt}"
    inputs = tokenizer(prefixed_prompt, return_tensors="pt", padding=True, truncation=True)

    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=32,
        repetition_penalty=2.0,    # ← Anti-repetition
        num_beams=4,               # ← Better quality
        early_stopping=True,       # ← Stop at EOS
        eos_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

**Breakthrough Results:**
- **Input:** "Symptoms: Shortness of breath, fever, CRP 90..."
- **Output:** "Shortness of breath, fever, CRP 90, X-ray" ✅

**Analysis:** Model extracts relevant medical information, but still no diagnosis!

---

## 🚀 Final Phase: Debugging Complete, Ready for Real Training

### Step 13: Preparing the Baseline

At this stage, the debugging is complete. The pipeline is stable, the model architecture is understood, and the generation parameters are tuned. The code below represents the functional baseline required to begin serious training.

> **Note:** The "160 examples, 40 epochs, loss 0.009" scenario mentioned previously was a stress test to confirm the pipeline works. It is not the final model. Real training requires:
> - A larger, diverse dataset
> - Proper train/validation/test splits
> - Hyperparameter tuning for generalization, not just loss minimization
> - Rigorous validation against overfitting

The goal of this guide was to get the system to a state where training can happen reliably.

### Overfitting Risk

160 examples, 40 epochs, loss of 0.009, 100% accuracy on 5 hand-crafted test cases — this is almost certainly overfitting. The model likely memorized the training data.

**Next Steps for Production:**

Before deploying, you must implement a proper validation strategy (hold-out sets, cross-validation, and out-of-distribution testing) to ensure the model generalizes beyond the training data.

---

## 📋 Debugging Steps Summary

### 🔍 Systematic Problem Identification

1. **Parameter Instability Analysis**
   - Cross-pattern recognition between different deprecated warnings
   - Isolate individual parameter changes
   - *Note: This was a detour; focus on model behavior first*

2. **Pipeline Component Testing**
   - Labels tokenization ✅
   - Attention mask ✅
   - EOS/PAD token handling ✅
   - DataCollator ❌ → FIXED

3. **T5-Specific Requirements**
   - Task prefix requirement identified
   - Encoder-decoder pipeline understood

4. **Generation Mechanism Optimization**
   - Parameter tuning for anti-repetition
   - Beam search for better quality

5. **Pipeline Stability**
   - Confirmed the system can train without crashing
   - Confirmed the system can generate non-empty output
   - Ready for scaling data and hyperparameter tuning

---

## 🛠️ Final Debugged Code (Ready for Training)

```python
import pandas as pd
import transformers
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

# DATA PREPARATION: Replace with your full, diverse dataset
data = [
    # ... [Your full dataset here]
]

tokenizer = T5Tokenizer.from_pretrained("t5-small")

# T5 TASK PREFIX (Critical for T5 performance)
def tokenize_with_task_prefix(example):
    task_prefixed_input = f"medical diagnosis: {example['input']}"
    input_enc = tokenizer(task_prefixed_input, truncation=True, padding="max_length", max_length=128)
    output_enc = tokenizer(example["output"], truncation=True, padding="max_length", max_length=32)
    input_enc["labels"] = output_enc["input_ids"]
    return input_enc

dataset = Dataset.from_pandas(data)
tokenized_dataset = dataset.map(tokenize_with_task_prefix)

# DATACOLLATOR FIX: Remove string features to prevent crashes
tokenized_dataset = tokenized_dataset.remove_columns(["input", "output"])

model = T5ForConditionalGeneration.from_pretrained("t5-small")

# TRAINING ARGUMENTS: Tuned for stability
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    num_train_epochs=40,          # Adjust based on validation performance
    learning_rate=3e-4,
    warmup_steps=50,
    logging_steps=10,
    evaluation_strategy="epoch",  # Recommended: Monitor validation loss
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none"
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# START REAL TRAINING HERE
trainer.train()

# OPTIMIZED PREDICTION FUNCTION
def predict_medical_diagnosis(prompt):
    prefixed_prompt = f"medical diagnosis: {prompt}"
    inputs = tokenizer(prefixed_prompt, return_tensors="pt", padding=True, truncation=True)

    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=32,
        repetition_penalty=2.0,    # Anti-repetition
        num_beams=4,               # Better quality
        early_stopping=True,       # Stop at EOS
        eos_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# TEST
test_prompt = "Symptoms: Shortness of breath, fever, CRP 90, X-ray: Right infiltrate. What is the most likely diagnosis?"
result = predict_medical_diagnosis(test_prompt)
print(f"Diagnosis: {result}")
```

---

## 📊 Critical Success Factors

### ✅ Must-Have Components

| Component | Purpose |
|-----------|---------|
| T5 Task Prefix | `"medical diagnosis: "` — Essential for T5 understanding |
| DataCollator Fix | Remove string features |
| Sufficient Data | At least 100+ examples for complex mappings |
| Advanced Generation | Repetition penalty, beam search, early stopping |
| Training Volume | Adequate epochs for task learning |

### ❌ Common Pitfalls

| Pitfall | Risk |
|---------|------|
| Deprecated Parameters | New APIs not always more stable |
| Fresh vs Continue | Continue training can be better than fresh (but investigate why) |
| Cache/Memory Issues | Fresh environment solves many problems |
| Generation Parameters | Standard parameters often insufficient |
| Dataset Size | Too small datasets lead to overfitting/repetition |
| Overfitting Misread as Success | Near-zero loss on a small dataset is a warning sign, not a victory |

---

## 🧠 Debugging Strategies

### 1. Systematic Isolation
- Change one variable at a time
- Start from working baseline
- Forward debugging instead of backward guessing

### 2. Pipeline-Oriented Diagnosis
```
Input → Tokenization → Attention → Training → Generation → Output
  ✅        ✅           ✅         ❌         ❌        ❌
```
Test each step systematically

### 3. Fresh Environment as Debugging Tool
- Eliminate cache/memory issues
- Clean state for reproducible tests
- Enable controlled experiments

### 4. Parameter Instability Recognition
- Take deprecated warnings seriously — but don't chase them at the expense of investigating actual model behavior
- Cross-pattern recognition between different errors
- Conservative parameter choice when uncertain

### 5. Model-Specific Requirements Understanding
- T5 needs task prefix for new tasks
- Encoder-decoder models have special requirements
- Generation parameters are critical for output quality

### 6. Interrogate Apparent Findings
- "Continue training worked better" — why? Optimizer state? Learning rate schedule? Don't generalize without understanding the mechanism.
- "Loss dropped to 0.009" — on how many examples? With how many epochs? Near-zero loss on small data is overfitting until proven otherwise.

---

## 🎯 Final Insights

### What Worked

| Principle | Application |
|-----------|-------------|
| Medicine debugging principles | → ML engineering |
| Systematic differential diagnosis | → Bug isolation |
| "Better safe than sorry" | → Conservative development |
| Fresh environment strategy | → Clean testing |
| Cross-pattern recognition | → Root cause analysis |

### Performance Metrics (With Caveats)

| Metric | Value | Note |
|--------|-------|------|
| Training Loss | 9.9 → 0.009 | 99.9% improvement — likely overfitting |
| Test Accuracy | 100% on 5 cases | Training distribution only |
| Specialty Coverage | 19 medical specialties | — |
| Debugging Time | ~3 hours | Systematic analysis |

> ⚠️ **Note:** These metrics reflect training performance, not generalization capability. The debugging phase is now complete; the next phase is rigorous training and validation.

---

## 💡 Key Takeaways for ML Engineering

### 1. Debugging Is a Systematic Process
Don't guess — test methodically

### 2. Domain Knowledge + Technical Skills = Success
Medical expertise + ML engineering = powerful combination

### 3. Fresh Environment Is a Powerful Tool
"Turn it off and on again" works for ML too

### 4. Conservative Parameter Choice Pays Off
Old, stable parameters > new, unstable parameters

### 5. Model-Specific Requirements Are Critical
T5, BERT, GPT all have different best practices

### 6. Success Metrics Require Context
Low loss + high accuracy on small test set ≠ production-ready model. Always validate generalization before deployment.

### 7. Focus on Behavior, Not Warnings
Chasing deprecation messages feels productive but can waste time. Prioritize investigating what the model is actually doing over what the library is complaining about.

---

*This document shows how real ML problems are solved in practice: Not through luck or intuition, but through systematic analysis, methodical testing, and step-by-step problem solving — with honest acknowledgment of limitations and overfitting risks.*
```
