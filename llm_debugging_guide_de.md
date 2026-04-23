# LLM Fine-tuning Debugging Guide: Systematische Problemlösung in der Praxis

**Ein kompletter Walkthrough vom ersten Problem bis zum funktionierenden medizinischen LLM**

## 🎯 Projektziel
Entwicklung eines medizinischen LLM zur Diagnose-Unterstützung mittels T5-Fine-tuning
**Disclaimer**: Nur für Bildungszwecke- ist nicht für den medizinischen Einsatz geeignet.

---

## 📋 Ausgangssituation

### Ursprünglicher Code (funktionierend, aber begrenzt)
```python
import pandas as pd
import transformers
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

data = [
    {"input": "Symptome: Fieber, Husten. CRP: 67. Bildgebung: Infiltrat basal rechts. Was ist die wahrscheinlichste Diagnose?", "output": "Pneumonie"},
    {"input": "Symptome: Dyspnoe, Beinschwellung links. D-Dimer erhöht. Was ist die wahrscheinlichste Diagnose?", "output": "Lungenembolie"},
    {"input": "Symptome: Müdigkeit, Blässe. Hb: niedrig. Was ist die wahrscheinlichste Diagnose?", "output": "Anämie"},
    {"input": "Symptome: Brustschmerz, Troponin hoch, EKG ST-Hebung. Was ist die wahrscheinlichste Diagnose?", "output": "Herzinfarkt"},
    {"input": "Symptome: Polyurie, Polydipsie, BZ 320 mg/dl. Was ist die wahrscheinlichste Diagnose?", "output": "Diabetes mellitus"}
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

test_prompt = "Symptome: Atemnot, Fieber, CRP 90, Röntgen: Infiltrat rechts. Was ist die wahrscheinlichste Diagnose?"
print("Antwort:", predict(test_prompt))
```

### Erste Ergebnisse (problematisch aber funktional)
- **Output:** `"Pneumonie. DD: Pneumonie, Pneumonie"` (repetitiv)
- **Loss:** 8.78 → 0.43 (sehr gut)
- **Problem:** Repetitive/falsche Differentialdiagnosen

---

## 🚨 Problem Phase 1: Strukturelle Verbesserung führt zu "True"-Bug

### Versuch: Erweiterte Features implementieren
**Ziel:** 100 Beispiele, Validation Split, bessere Output-Struktur

**Änderungen:**
- Dataset auf 100 Beispiele erweitert
- Strukturierte DD-Ausgabe: `"Diagnose: X | DD: Y, Z, W"`
- Train/Validation Split (80/20)
- `as_target_tokenizer()` → `text_target` (deprecated fix)
- `tokenizer` → `processing_class` parameter

### Problem: "True"-Bug
```python
# Expected: "Pneumonie. DD: Bronchitis, Pleuritis"  
# Actual: "True"
```

**Symptome:**
- Alle Outputs nur noch `"True"`
- Model verhält sich wie Binary Classifier
- Missing keys warning: `embed_tokens.weight`, `lm_head.weight`

---

## 🔍 Debugging Phase 1: Systematische Problemidentifikation

### Step 1: Parameter-Instabilität-Hypothese
**Beobachtung:** Mehrere deprecated/neue Parameter gleichzeitig geändert
- `evaluation_strategy` → TypeError  
- `processing_class` vs `tokenizer`
- `text_target` vs `as_target_tokenizer()`

**Hypothese:** Neue Parameter sind instabil, alte Parameter funktionieren besser

### Step 2: Schrittweise Rückführung
**Strategie:** Eine Variable zur Zeit ändern

#### Test 1: `as_target_tokenizer()` Fix
```python
# Zurück zu deprecated aber funktionierender Methode
with tokenizer.as_target_tokenizer():
    output_enc = tokenizer(example["output"], ...)
```
**Ergebnis:** `"rmelkinese"` (korrupt, aber nicht mehr "True")

#### Test 2: Original vs Fix Vergleich
**Ergebnis:** Beide Male `"rmelkinese"` → Problem liegt woanders

---

## 🧹 Debugging Phase 2: Fresh Environment Strategy

### Step 3: Clean Slate Approach
**Entscheidung:** Fresh Notebook, zurück zur funktionierenden Basis

**Baseline Test (5 Beispiele, Original-Code):**
```python
# Minimaler Test für Root Cause Isolation
data = [ursprüngliche 5 Beispiele ohne DD]
```
**Ergebnis:** `"Was ist die wahrscheinlichste Diagnose?"` (Input-Echo)

---

## 🔬 Debugging Phase 3: Pipeline-Diagnose

### Step 4: Labels-Debug
**Check:** Sind Labels korrekt tokenisiert?
```python
print("Sample tokenized data:")
print(f"Labels: {tokenized_dataset[0]['labels'][:10]}")
print(f"Decoded Labels: {tokenizer.decode(tokenized_dataset[0]['labels'])}")
```
**Ergebnis:** ✅ Labels perfekt: `"Pneumonie</s><pad>..."`

### Step 5: Attention Mask Debug  
**Check:** Funktioniert die Attention-Mechanik?
```python
inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128)
print(f"Attention mask: {inputs.attention_mask}")
print(f"Attention mask sum: {inputs.attention_mask[0].sum()}")
```
**Ergebnis:** ✅ Attention perfekt: 36/36 tokens attended

### Step 6: EOS/PAD Token Debug
**Check:** Token-Handling korrekt?
```python
print(f"PAD token: '{tokenizer.pad_token}' -> ID: {tokenizer.pad_token_id}")
print(f"EOS token: '{tokenizer.eos_token}' -> ID: {tokenizer.eos_token_id}")
```
**Ergebnis:** ✅ Token-Setup korrekt, aber Generation produziert Input-Echo

---

## 🚨 Problem Phase 2: DataCollator Crash

### Step 7: Label-Training-Pipeline Debug
**Tieferer Test:** Was passiert im Training?

**CRASH:** 
```
ValueError: Unable to create tensor... Perhaps your features (`input` in this case) have excessive nesting
```

### Root Cause: String Features im Dataset
**Problem:** DataCollator kann nicht alle Features tensorizieren
```python
tokenized_dataset.features = {
    "input": "string",      # ❌ DataCollator crash  
    "output": "string",     # ❌ DataCollator crash
    "input_ids": "tensor",  # ✅ OK
    "labels": "tensor"      # ✅ OK
}
```

### Fix: String Features entfernen
```python
tokenized_dataset = tokenized_dataset.remove_columns(["input", "output"])
```

**Ergebnis:** Training läuft, aber Output immer noch falsch

---

## 🔍 Debugging Phase 4: T5-spezifische Probleme

### Step 8: T5 Training Mode Check
**Check:** Versteht T5 unseren Task?

**DISCOVERY:** T5 hat task-specific parameters:
```python
model.config.task_specific_params = {
    'summarization': {'prefix': 'summarize: '},
    'translation_en_to_de': {'prefix': 'translate English to German: '},
    ...
}
```

**Problem:** T5 versteht ohne Task-Prefix nicht was zu tun ist!

### Step 9: Task Prefix Implementation
```python
def tokenize_with_task_prefix(example):
    task_prefixed_input = f"medical diagnosis: {example['input']}"
    input_enc = tokenizer(task_prefixed_input, truncation=True, padding="max_length", max_length=128)
    output_enc = tokenizer(example["output"], truncation=True, padding="max_length", max_length=32)
    input_enc["labels"] = output_enc["input_ids"]
    return input_enc
```

**Ergebnis:** Input-Echo stoppt, aber nur leere Outputs

---

## 🚨 Problem Phase 3: PAD Token Loop

### Step 10: Generation Mechanism Debug
**Problem:** Model generiert nur PAD tokens `[0,0,0,...]`

**Deep Debug:**
```python
# Raw token analysis
outputs = model.generate(inputs, max_length=32, do_sample=False)
print(f"Raw tokens: {outputs[0]}")
# Result: [0, 0, 0, 0, 0, 0, ...]
```

### Hypothese: Training Volume vs Decoder Mechanism
**Diskussion:** 
- Sind 10 Epochen zu wenig für Task Prefix Learning?
- Oder ist Decoder-Start-Mechanism kaputt?

### Step 11: A/B Test Strategy
**Test 1:** Continue Training (+20 Epochen)
**Test 2:** Fresh Training (30 Epochen from scratch)

#### Continue Training Ergebnis:
- **Loss:** 2.0 → 0.15-0.30
- **Output:** `"Morbus Morbus Morbus..."` ✅ (medizinische Begriffe, aber repetitiv)

#### Fresh Training Ergebnis:  
- **Loss:** 10.1 → 0.30-0.85
- **Output:** `""` (leer, PAD tokens)

**Conclusion:** Continue Training ist besser als Fresh!

---

## 🎯 Breakthrough Phase: Generation Parameter Optimization

### Step 12: Improved Generation Parameters
**Problem:** Repetitive Output (`"Morbus Morbus Morbus..."`)

**Solution:** Advanced Generation Parameters
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
- **Input:** `"Symptome: Atemnot, Fieber, CRP 90..."`
- **Output:** `"Atemnot, Fieber, CRP 90, Röntgen"` ✅

**Analysis:** Model extrahiert relevante medizinische Information, aber noch keine Diagnose!

---

## 🚀 Final Success Phase: Scale & Training Optimization

### Step 13: Dataset & Training Scale-Up
**Strategy:** Mehr Daten + Intensiveres Training

**Scaling:**
- **25 → 160 Beispiele** (6x mehr Daten)
- **30 → 40 Epochen** (mehr Training)
- **19 medizinische Fachbereiche** abgedeckt

**Optimierte Training-Parameter:**
```python
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,  # Größere Batches
    num_train_epochs=40,            # Mehr Epochen
    learning_rate=3e-4,             # Optimierte LR
    warmup_steps=50,                # Warmup für Stabilität
    logging_steps=10,
    save_strategy="no",
    report_to="none"
)
```

### Final Training Results:
- **Loss:** 9.9 → **0.009** (Outstanding!)
- **160 Beispiele** erfolgreich trainiert
- **40 Epochen** mit perfekter Konvergenz

---

## 🏆 ERFOLG: Funktionierendes Medizinisches LLM

### Final Test Results (100% Success Rate):

| Test | Input | Generated | Expected | Status |
|------|-------|-----------|----------|--------|
| 1 | Fieber, Husten, Infiltrat | **Pneumonie** | Pneumonie | ✅ |
| 2 | Brustschmerz, Troponin, ST-Hebung | **Herzinfarkt** | Herzinfarkt | ✅ |
| 3 | Polyurie, BZ 320 mg/dl | **Diabetes mellitus** | Diabetes mellitus | ✅ |
| 4 | Tremor, Rigor, Bradykinesie | **Morbus Parkinson** | Morbus Parkinson | ✅ |
| 5 | Kopfschmerz, Meningismus | **Meningitis** | Meningitis | ✅ |

---

## 📋 Debugging-Schritte Zusammenfassung

### 🔍 Systematische Problemidentifikation

1. **Parameter-Instabilität-Analyse**
   - Cross-Pattern Recognition zwischen verschiedenen deprecated warnings
   - Isolierung einzelner Parameter-Änderungen

2. **Pipeline-Komponenten-Test**  
   - Labels-Tokenization ✅
   - Attention Mask ✅
   - EOS/PAD Token Handling ✅
   - DataCollator ❌ → **FIXED**

3. **T5-spezifische Anforderungen**
   - Task Prefix Requirement identifiziert
   - Encoder-Decoder Pipeline verstanden

4. **Generation-Mechanismus-Optimierung**
   - Parameter-Tuning für Anti-Repetition
   - Beam Search für bessere Qualität

5. **Scale & Training-Optimierung**
   - Dataset-Größe als kritischer Faktor
   - Training-Volumen für komplexe Tasks

---

## 🛠️ Finale Code-Lösung

```python
import pandas as pd
import transformers
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

# GROSSE DATENBASIS: 160 medizinische Beispiele
data = [
    # ... [160 Beispiele aus 19 Fachbereichen]
]

tokenizer = T5Tokenizer.from_pretrained("t5-small")

# T5 TASK PREFIX (kritisch für T5-Performance)
def tokenize_with_task_prefix(example):
    task_prefixed_input = f"medical diagnosis: {example['input']}"
    input_enc = tokenizer(task_prefixed_input, truncation=True, padding="max_length", max_length=128)
    output_enc = tokenizer(example["output"], truncation=True, padding="max_length", max_length=32)
    input_enc["labels"] = output_enc["input_ids"]
    return input_enc

dataset = Dataset.from_pandas(data)
tokenized_dataset = dataset.map(tokenize_with_task_prefix)

# DATACOLLATOR FIX: String features entfernen
tokenized_dataset = tokenized_dataset.remove_columns(["input", "output"])

model = T5ForConditionalGeneration.from_pretrained("t5-small")

# OPTIMIERTE TRAINING-PARAMETER
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

# OPTIMIERTE PREDICTION-FUNKTION
def predict_medical_diagnosis(prompt):
    prefixed_prompt = f"medical diagnosis: {prompt}"
    inputs = tokenizer(prefixed_prompt, return_tensors="pt", padding=True, truncation=True)
    
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=32,
        repetition_penalty=2.0,    # Anti-repetition
        num_beams=4,               # Bessere Qualität
        early_stopping=True,       # Stop bei EOS
        eos_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# TEST
test_prompt = "Symptome: Atemnot, Fieber, CRP 90, Röntgen: Infiltrat rechts. Was ist die wahrscheinlichste Diagnose?"
result = predict_medical_diagnosis(test_prompt)
print(f"Diagnose: {result}")  # Output: "Pneumonie"
```

---

## 📊 Kritische Erfolgsfaktoren

### ✅ Must-Have Komponenten:
1. **T5 Task Prefix:** `"medical diagnosis: "` - Essentiell für T5-Verständnis
2. **DataCollator Fix:** String features entfernen
3. **Sufficient Data:** Mindestens 100+ Beispiele für komplexe Mappings  
4. **Advanced Generation:** Repetition penalty, beam search, early stopping
5. **Training Volume:** 40+ Epochen für Task Learning

### ❌ Häufige Fallstricke:
1. **Deprecated Parameter:** Neue APIs nicht immer stabiler
2. **Fresh vs Continue:** Continue Training kann besser sein als Fresh
3. **Cache/Memory Issues:** Fresh Environment löst viele Probleme
4. **Generation Parameters:** Standard-Parameter oft unzureichend
5. **Dataset Size:** Zu kleine Datasets führen zu Overfitting/Repetition

---

## 🧠 Debugging-Strategien (Lessons Learned)

### 1. Systematische Isolation
- **Eine Variable zur Zeit ändern**
- **Von funktionierender Basis ausgehen**
- **Vorwärts-Debugging statt Rückwärts-Raten**

### 2. Pipeline-orientierte Diagnose
```
Input → Tokenization → Attention → Training → Generation → Output
   ✅        ✅           ✅         ❌         ❌        ❌
```
**Systematisch jeden Schritt einzeln testen**

### 3. Fresh Environment als Debugging-Tool
- **Cache/Memory-Issues** eliminieren
- **Clean State** für reproduzierbare Tests
- **Controlled Experiments** ermöglichen

### 4. Parameter-Instabilität erkennen
- **Deprecated Warnings** ernst nehmen
- **Cross-Pattern Recognition** zwischen verschiedenen Fehlern
- **Conservative Parameter Choice** bei Unsicherheit

### 5. Model-spezifische Anforderungen verstehen
- **T5 braucht Task Prefix** für neue Tasks
- **Encoder-Decoder Models** haben spezielle Anforderungen
- **Generation Parameters** sind kritisch für Output-Qualität

---

## 🎯 Finale Erkenntnisse

### Was funktioniert hat:
1. **Notfallmedizin-Debugging-Prinzipien** → ML Engineering
2. **Systematische Differential-Diagnose** → Bug Isolation
3. **"Better safe than sorry"** → Conservative Development
4. **Fresh Environment Strategy** → Clean Testing
5. **Cross-Pattern Recognition** → Root Cause Analysis

### Performance Metriken:
- **Training Loss:** 9.9 → 0.009 (99.9% Improvement)
- **Test Accuracy:** 100% auf 5 verschiedenen medizinischen Cases
- **Fachbereich-Abdeckung:** 19 medizinische Specialitäten
- **Debugging-Zeit:** ~3 Stunden systematischer Analyse

---

## 🚀 Nächste Entwicklungsschritte

### Mögliche Erweiterungen:
1. **Differentialdiagnosen** hinzufügen
2. **Confidence Scoring** implementieren  
3. **Validation Set** für Overfitting-Prävention
4. **Größeres Model** (T5-base/large) für komplexere Cases
5. **Real-world Medical Data** Integration

### Deployment-Überlegungen:
- **Model Versioning** für verschiedene Specialitäten
- **API Wrapper** für klinische Integration
- **Safety Measures** für medizinische Anwendungen
- **Continuous Learning** aus neuen Cases

---

## 💡 Key Takeaways für ML Engineering

### 1. Debugging ist ein systematischer Prozess
**Nicht raten, sondern methodisch testen**

### 2. Domain Knowledge + Technical Skills = Erfolg
**Medizinische Expertise + ML Engineering = Powerful Combination**

### 3. Fresh Environment ist ein mächtiges Tool
**"Turn it off and on again" funktioniert auch bei ML**

### 4. Conservative Parameter Choice zahlt sich aus
**Alte, stabile Parameter > neue, instabile Parameter**

### 5. Model-spezifische Anforderungen sind kritisch
**T5, BERT, GPT haben alle verschiedene Best Practices**

---

## 🏆 Projekt-Erfolg

**Von einem nicht-funktionierenden "True"-Bug zu einem 100% akkuraten medizinischen LLM in systematischen Debugging-Schritten.**

**Beweis: Systematische Herangehensweise + Domain-Expertise + Technische Umsetzung = Erfolgreiche ML-Lösung**

**Final loss: 0.009; accuracy on our small, hand-crafted test set was 100%, likely due to the limited dataset and clear-cut labels (e.g., Pneumonia, Myocardial infarction). This should not be interpreted as clinical performance.

---

*Dieses Dokument zeigt, wie echte ML-Probleme in der Praxis gelöst werden: Nicht durch Glück oder Intuition, sondern durch systematische Analyse, methodisches Testen und schrittweise Problemlösung.*
