
# LLM Fine-tuning Debugging Guide: Systematic Problem Solving in Practice
**A complete walkthrough from the first problem to a working medical LLM**

---
## 🎯 Project Goal
Develop a medical LLM for diagnostic support using T5 fine-tuning.

---
## 📋 Initial Situation
### Original Code (functional but limited)
```python
import pandas as pd
import transformers
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

data = [
    {"input": "Symptoms: Fever, cough. CRP: 67. Imaging: Infiltrate basal right. What is the most likely diagnosis?", "output": "Pneumonia"},
    {"input": "Symptoms: Dyspnea, left leg swelling. D-Dimer elevated. What is the most likely diagnosis?", "output": "Pulmonary embolism"},
    {"input": "Symptoms: Fatigue, pallor. Hb: low. What is the most likely diagnosis?", "output": "Anemia"},
    {"input": "Symptoms: Chest pain, high troponin, EKG ST-elevation. What is the most likely diagnosis?", "output": "Myocardial infarction"},
    {"input": "Symptoms: Polyuria, polydipsia, blood glucose 320 mg/dl. What is the most likely diagnosis?", "output": "Diabetes mellitus"}
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

test_prompt = "Symptoms: Shortness of breath, fever, CRP 90, X-ray: Infiltrate right. What is the most likely diagnosis?"
print("Answer:", predict(test_prompt))
```

### Initial Results (problematic but functional)
- **Output:** `"Pneumonia. DD: Pneumonia, Pneumonia"` (repetitive)
- **Loss:** 8.78 → 0.43 (very good)
- **Problem:** Repetitive/incorrect differential diagnoses

---
## 🚨 Problem Phase 1: Structural Improvement Leads to "True" Bug
### Attempt: Implementing Extended Features
**Goal:** 100 examples, validation split, better output structure
**Changes:**
- Dataset expanded to 100 examples
- Structured DD output: `"Diagnosis: X | DD: Y, Z, W"`
- Train/validation split (80/20)
- `as_target_tokenizer()` → `text_target` (deprecated fix)
- `tokenizer` → `processing_class` parameter

### Problem: "True" Bug
```python
# Expected: "Pneumonia. DD: Bronchitis, Pleuritis"
# Actual: "True"
```
**Symptoms:**
- All outputs only `"True"`
- Model behaves like a binary classifier
- Missing keys warning: `embed_tokens.weight`, `lm_head.weight`

---
## 🔍 Debugging Phase 1: Systematic Problem Identification
### Step 1: Parameter Instability Hypothesis
**Observation:** Multiple deprecated/new parameters changed simultaneously
- `evaluation_strategy` → TypeError
- `processing_class` vs `tokenizer`
- `text_target` vs `as_target_tokenizer()`

**Hypothesis:** New parameters are unstable; old parameters work better

### Step 2: Stepwise Rollback
**Strategy:** Change one variable at a time

#### Test 1: `as_target_tokenizer()` Fix
```python
# Revert to deprecated but functional method
with tokenizer.as_target_tokenizer():
    output_enc = tokenizer(example["output"], ...)
```
**Result:** `"rmelkinese"` (corrupt, but no longer "True")

#### Test 2: Original vs Fix Comparison
**Result:** Both times `"rmelkinese"` → Problem lies elsewhere

---
## 🧹 Debugging Phase 2: Fresh Environment Strategy
### Step 3: Clean Slate Approach
**Decision:** Fresh notebook, back to functional baseline
**Baseline Test (5 examples, original code):**
```python
# Minimal test for root cause isolation
data = [original 5 examples without DD]
```
**Result:** `"What is the most likely diagnosis?"` (input echo)

---
## 🔬 Debugging Phase 3: Pipeline Diagnosis
### Step 4: Labels Debug
**Check:** Are labels correctly tokenized?
```python
print("Sample tokenized data:")
print(f"Labels: {tokenized_dataset[0]['labels'][:10]}")
print(f"Decoded Labels: {tokenizer.decode(tokenized_dataset[0]['labels'])}")
```
**Result:** ✅ Labels perfect: `"Pneumonia</s><pad>..."`

### Step 5: Attention Mask Debug
**Check:** Does the attention mechanism work?
```python
inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128)
print(f"Attention mask: {inputs.attention_mask}")
print(f"Attention mask sum: {inputs.attention_mask[0].sum()}")
```
**Result:** ✅ Attention perfect: 36/36 tokens attended

### Step 6: EOS/PAD Token Debug
**Check:** Is token handling correct?
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

### Root Cause: String Features in Dataset
**Problem:** DataCollator cannot tensorize all features
```python
tokenized_dataset.features = {
    "input": "string",      # ❌ DataCollator crash
    "output": "string",     # ❌ DataCollator crash
    "input_ids": "tensor",  # ✅ OK
    "labels": "tensor"      # ✅ OK
}
```

### Fix: Remove String Features
```python
tokenized_dataset = tokenized_dataset.remove_columns(["input", "output"])
```
**Result:** Training runs, but output still incorrect

---
## 🔍 Debugging Phase 4: T5-Specific Problems
### Step 8: T5 Training Mode Check
**Check:** Does T5 understand our task?
**Discovery:** T5 has task-specific parameters:
```python
model.config.task_specific_params = {
    'summarization': {'prefix': 'summarize: '},
    'translation_en_to_de': {'prefix': 'translate English to German: '},
    ...
}
```
**Problem:** Without task prefix, T5 doesn't know what to do!

### Step 9: Task Prefix Implementation
```python
def tokenize_with_task_prefix(example):
    task_prefixed_input = f"medical diagnosis: {example['input']}"
    input_enc = tokenizer(task_prefixed_input, truncation=True, padding="max_length", max_length=128)
    output_enc = tokenizer(example["output"], truncation=True, padding="max_length", max_length=32)
    input_enc["labels"] = output_enc["input_ids"]
    return input_enc
```
**Result:** Input echo stops, but only empty outputs

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

### Hypothesis: Training Volume vs Decoder Mechanism
**Discussion:**
- Are 10 epochs too few for task prefix learning?
- Or is the decoder-start mechanism broken?

### Step 11: A/B Test Strategy
**Test 1:** Continue Training (+20 epochs)
**Test 2:** Fresh Training (30 epochs from scratch)

#### Continue Training Result:
- **Loss:** 2.0 → 0.15-0.30
- **Output:** `"Morbus Morbus Morbus..."` ✅ (medical terms, but repetitive)

#### Fresh Training Result:
- **Loss:** 10.1 → 0.30-0.85
- **Output:** `""` (empty, PAD tokens)

**Conclusion:** Continue training is better than fresh!

---
## 🎯 Breakthrough Phase: Generation Parameter Optimization
### Step 12: Improved Generation Parameters
**Problem:** Repetitive output (`"Morbus Morbus Morbus..."`)
**Solution:** Advanced generation parameters
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

### Breakthrough Results:
- **Input:** `"Symptoms: Shortness of breath, fever, CRP 90..."`
- **Output:** `"Shortness of breath, fever, CRP 90, X-ray"` ✅
**Analysis:** Model extracts relevant medical information, but no diagnosis yet!

---
## 🚀 Final Success Phase: Scale & Training Optimization
### Step 13: Dataset & Training Scale-Up
**Strategy:** More data + intensive training
**Scaling:**
- **25 → 160 examples** (6x more data)
- **30 → 40 epochs** (more training)
- **19 medical specialties** covered

**Optimized Training Parameters:**
```python
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,  # Larger batches
    num_train_epochs=40,            # More epochs
    learning_rate=3e-4,             # Optimized LR
    warmup_steps=50,                # Warmup for stability
    logging_steps=10,
    save_strategy="no",
    report_to="none"
)
```

### Final Training Results:
- **Loss:** 9.9 → **0.009** (Outstanding!)
- **160 examples** successfully trained
- **40 epochs** with perfect convergence

---
## 🏆 SUCCESS: Functional Medical LLM
### Final Test Results (100% Success Rate):


Final Test Results


| Test | Input | Generated | Expected | Status |
|------|-------|-----------|----------|--------|
| 1 | Fever, cough, infiltrate | **Pneumonia** | Pneumonia | ✅ |
| 2 | Chest pain, troponin, ST-elevation | **Myocardial infarction** | Myocardial infarction | ✅ |
| 3 | Polyuria, blood glucose 320 mg/dl | **Diabetes mellitus** | Diabetes mellitus | ✅ |
| 4 | Tremor, rigidity, bradykinesia | **Parkinson's disease** | Parkinson's disease | ✅ |
| 5 | Headache, meningismus | **Meningitis** | Meningitis | ✅ |

---
## 📋 Debugging Steps Summary
### 🔍 Systematic Problem Identification
1. **Parameter Instability Analysis**
   - Cross-pattern recognition between various deprecated warnings
   - Isolation of individual parameter changes
2. **Pipeline Component Test**
   - Labels tokenization ✅
   - Attention mask ✅
   - EOS/PAD token handling ✅
   - DataCollator ❌ → **FIXED**
3. **T5-Specific Requirements**
   - Task prefix requirement identified
   - Encoder-decoder pipeline understood
4. **Generation Mechanism Optimization**
   - Parameter tuning for anti-repetition
   - Beam search for better quality
5. **Scale & Training Optimization**
   - Dataset size as a critical factor
   - Training volume for complex tasks

---
## 🛠️ Final Code Solution
```python
import pandas as pd
import transformers
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

# LARGE DATABASE: 160 medical examples
data = [
    # ... [160 examples from 19 specialties]
]

tokenizer = T5Tokenizer.from_pretrained("t5-small")

# T5 TASK PREFIX (critical for T5 performance)
def tokenize_with_task_prefix(example):
    task_prefixed_input = f"medical diagnosis: {example['input']}"
    input_enc = tokenizer(task_prefixed_input, truncation=True, padding="max_length", max_length=128)
    output_enc = tokenizer(example["output"], truncation=True, padding="max_length", max_length=32)
    input_enc["labels"] = output_enc["input_ids"]
    return input_enc

dataset = Dataset.from_pandas(data)
tokenized_dataset = dataset.map(tokenize_with_task_prefix)

# DATACOLLATOR FIX: Remove string features
tokenized_dataset = tokenized_dataset.remove_columns(["input", "output"])

model = T5ForConditionalGeneration.from_pretrained("t5-small")

# OPTIMIZED TRAINING PARAMETERS
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    num_train_epochs=40,
    learning_rate=3e-4,
    warmup_steps=50,
    logging_steps=10,
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
test_prompt = "Symptoms: Shortness of breath, fever, CRP 90, X-ray: Infiltrate right. What is the most likely diagnosis?"
result = predict_medical_diagnosis(test_prompt)
print(f"Diagnosis: {result}")  # Output: "Pneumonia"
```

---
## 📊 Critical Success Factors
### ✅ Must-Have Components:
1. **T5 Task Prefix:** `"medical diagnosis: "` - Essential for T5 understanding
2. **DataCollator Fix:** Remove string features
3. **Sufficient Data:** At least 100+ examples for complex mappings
4. **Advanced Generation:** Repetition penalty, beam search, early stopping
5. **Training Volume:** 40+ epochs for task learning

### ❌ Common Pitfalls:
1. **Deprecated Parameters:** New APIs not always more stable
2. **Fresh vs Continue:** Continue training can be better than fresh
3. **Cache/Memory Issues:** Fresh environment solves many problems
4. **Generation Parameters:** Default parameters often insufficient
5. **Dataset Size:** Too small datasets lead to overfitting/repetition

---
## 🧠 Debugging Strategies (Lessons Learned)
### 1. Systematic Isolation
- **Change one variable at a time**
- **Start from a working baseline**
- **Forward debugging, not backward guessing**

### 2. Pipeline-Oriented Diagnosis
```
Input → Tokenization → Attention → Training → Generation → Output
   ✅        ✅           ✅         ❌         ❌        ❌
```
**Test each step individually**

### 3. Fresh Environment as a Debugging Tool
- **Eliminate cache/memory issues**
- **Enable clean state for reproducible tests**
- **Allow controlled experiments**

### 4. Recognize Parameter Instability
- **Take deprecated warnings seriously**
- **Cross-pattern recognition between different errors**
- **Choose conservative parameters when in doubt**

### 5. Understand Model-Specific Requirements
- **T5 needs task prefix for new tasks**
- **Encoder-decoder models have special requirements**
- **Generation parameters are critical for output quality**

---
## 🎯 Final Insights
### What Worked:
1. **Emergency medicine debugging principles → ML engineering**
2. **Systematic differential diagnosis → Bug isolation**
3. **"Better safe than sorry" → Conservative development**
4. **Fresh environment strategy → Clean testing**
5. **Cross-pattern recognition → Root cause analysis**

### Performance Metrics:
- **Training Loss:** 9.9 → 0.009 (99.9% improvement)
- **Test Accuracy:** 100% on 5 different medical cases
- **Specialty Coverage:** 19 medical specialties
- **Debugging Time:** ~3 hours of systematic analysis

---
## 🚀 Next Development Steps
### Possible Extensions:
1. **Add differential diagnoses**
2. **Implement confidence scoring**
3. **Validation set for overfitting prevention**
4. **Larger model (T5-base/large) for complex cases**
5. **Real-world medical data integration**

### Deployment Considerations:
- **Model versioning for different specialties**
- **API wrapper for clinical integration**
- **Safety measures for medical applications**
- **Continuous learning from new cases**

---
## 💡 Key Takeaways for ML Engineering
### 1. Debugging is a systematic process
**Don't guess, test methodically**

### 2. Domain Knowledge + Technical Skills = Success
**Medical expertise + ML engineering = Powerful combination**

### 3. Fresh Environment is a Powerful Tool
**"Turn it off and on again" works for ML too**

### 4. Conservative Parameter Choice Pays Off
**Old, stable parameters > new, unstable parameters**

### 5. Model-Specific Requirements are Critical
**T5, BERT, GPT each have different best practices**

---
## 🏆 Project Success
**From a non-functional "True" bug to a 100% accurate medical LLM in systematic debugging steps.**

**Proof: Systematic approach + domain expertise + technical implementation = Successful ML solution**

**Final loss: 0.009; accuracy on our small, hand-crafted test set was 100%, likely due to the limited dataset and clear-cut labels (e.g., Pneumonia, Myocardial infarction). This should not be interpreted as clinical performance.**

---
*This document shows how real ML problems are solved in practice: Not by luck or intuition, but by systematic analysis, methodical testing, and step-by-step problem solving.*
```
