# Fixing T5 Fine-Tuning Bugs for Medical Diagnosis

This notebook documents the debugging process of fine-tuning T5-small for medical diagnosis tasks. What started as a simple text-to-text classification project turned into a deep dive into tokenizer handling, generation bugs, and training instabilities.
**Disclaimer**: this is for educational purposes only and not for actual medical diagnosis.

## Problem
After training, the model only output `"True"` or repeated inputs. Despite a great loss curve, the outputs were useless.

## Method
I approached this by:
- Isolating tokenizer and data collator issues
- Testing task prefix impact (critical for T5!)
- Comparing fresh vs continued training
- Tuning generation parameters (e.g. beam search, repetition penalty)

## Results
Final loss: 0.009; accuracy on our small, hand-crafted test set was 100%, likely due to the limited dataset and clear-cut labels (e.g., Pneumonia, Myocardial infarction). This should not be interpreted as clinical performance.
www.kaggle.com/code/kjacoby/debugging-guide-t5-fine-tuning-true-bug
