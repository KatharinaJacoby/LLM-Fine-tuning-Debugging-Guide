## AI Debugging Methodology Inspired by Medical Diagnostics

### 1. Introduction
- Purpose of this guide: share a structured debugging framework drawn from emergency medicine practices
- Target audience: AI/ML practitioners of all levels, especially those without clinical background
- Why medical diagnostics? High-stakes, time-sensitive, iterative decision-making under uncertainty

### 2. Core Principles from Emergency Medicine
1. **Rapid Assessment & Triage**
   - Briefly survey symptoms (error messages, logs)
   - Prioritize critical failures (data pipeline breaks vs. minor warnings)
2. **Hypothesis Generation**
   - Formulate multiple potential causes quickly
   - Use differential diagnoses (e.g., "Did the tokenizer crash or the attention mask fill?")
3. **Focused Testing**
   - Run focused checks (unit tests, small data subsets) to confirm or rule out causes
4. **Iterative Plan A → B → C...**
   - Develop successive remediation steps when initial fixes fail
5. **Continuous Monitoring**
   - Track metrics (loss curves, latency) to detect emerging issues early
6. **Team Communication & Handoff**
   - Clear notes/logs for collaborators or future you

### 3. Mapping to AI/ML Debugging
- Tokenization errors as "vital sign" alerts
- Data schema mismatches as "lab anomalies"
- Model convergence failures as "hemodynamic instability"

### 4. Step-by-Step Workflow
1. **Triage Phase**
   - Collect logs, set an incident title
   - Classify severity
2. **Differential Phase**
   - List 3–5 potential root causes
   - Rank by likelihood and impact
3. **Testing Phase**
   - Write minimal test scripts
   - Verify assumptions (data shape, model input/output)
4. **Intervention Phase**
   - Apply one fix at a time
   - Document outcome and revert if needed
5. **Re-evaluation Phase**
   - Assess if issue resolved or pursue next hypothesis

### 5. Case Studies
- Example 1: LLM fine-tuning loop bug (string columns & DataCollator)
- Example 2: Safety Mamba GRU-heads shape mismatch

### 6. Best Practices & Tips
- Use automated tests and CI
- Maintain a debugging checklist
- Leverage peer review and pair-programming

### 7. Conclusion
- Recap the medical analogy
- Encourage adaptation to individual workflows
- Invite feedback and contributions (link to GitHub)

---
*This guide leverages clinical diagnostic strategies to structure AI/ML debugging in high-stakes environments.*

