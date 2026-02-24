# 📄 Remedial Report
**H.E.A.D.S — Hybrid Entropy Anomaly Detection System (v1.0)**

---

## Identity
- **Name:** Adi Suryadi
- **Semester:** 4
- **Year:** 2026

---

## 1. Project Summary
In this remedial/evaluation project, I developed an anomaly detection system based on *Machine Learning* to analyze static *files* and identify potential *malware* threats based on low-level statistical characteristics *(byte-level)* such as *entropy* and non-printable *byte* distributions.

This system evolved by utilizing the **Isolation Forest** algorithm coupled with a **Hybrid Anomaly Detection** approach. Its final architecture was built to ensure that this detection system remains stable, consistent, and ready to be deployed directly in real-world monitoring scenarios.

## 2. Concepts and Architecture Used
This system comprises several core sub-components:
- **Feature Extraction:** Transformation based on the degree of randomness (entropy) and byte statistics, encompassing 11 primary features optimized for density.
- **Isolation Forest Model:** Utilizing the algorithm with 300 *estimators* parameter setting.
- **Locked Persisted Threshold:** An absolute locking mechanism on the detection boundary threshold (the threshold is not recalculated dynamically during the *inference* phase).
- **Realistic Evaluation Metrics:** In-depth evaluation tracking using *Precision*, *Recall*, and *F1 Score*.
- **Production Audit Monitoring:** Rigorous false alarm (*False Positive*) suppression testing and daily evaluation of model stability.

This approach focuses purely on behavioral *Anomaly Detection*, rather than relying on legacy signature-based *malware* recognition.

### System Workflow
To provide an operational overview, here is the specific breakdown of how data moves within the **H.E.A.D.S** system:
1. **Data Ingestion Phase**
   The system reads the extracted data rows of the raw *file* being scanned.
2. **Feature Engineering Phase**
   Calculates metrics like *Global Entropy*, *Block Entropy*, and the ratio of unreadable characters (*Non-printable Ratio*). Then builds special combinations to hunt for encrypted payloads (e.g., `entropy_x_nonprint`).
3. **Isolation Phase (Isolation Forest Engine)**
   The 11 extracted features are fed into the `IsolationForest` decision trees. The model then outputs a measure of isolation (*Raw Anomaly Score*).
4. **Locked Thresholding Phase**
   The raw score is compared to the absolute threshold frozen from the *Training* phase metadata (*Persisted Threshold*). If the score shows extreme negativity below the threshold line, the file is assumed to initially be a *Malware* threat.
5. **Benign Confirmation Layer (BCL) Phase**
   The file suspected as *Malware* is intercepted by the **BCL** filter. If the system detects a standard text composition (ASCII > 85%) and reasonable randomness (Entropy < 4.8), the *"accusation"* is revoked and the status returns to safe (*Benign/0*).
6. **Final Verdict Phase**
   The final status is issued (Benign / Malware) and recorded in the system logs.

#### 🔀 H.E.A.D.S Architecture Flowchart
This flowchart represents how the file traverses through the system, from the data extraction up to the final decision.

+--------------------------------------------------+
|              STATIC INPUT FILE & LOG            |
+--------------------------------------------------+
                        |
                        v
+--------------------------------------------------+
|     FEATURE EXTRACTION & ENGINEERING            |
|               (11 METRICS)                      |
+--------------------------------------------------+
                        |
                        v
+--------------------------------------------------+
|           ISOLATION FOREST MODEL                |
|         (Anomaly Score Calculation)             |
+--------------------------------------------------+
                        |
                        v
+--------------------------------------------------+
|              THRESHOLD EVALUATION               |
+--------------------------------------------------+
            |                          |
            |                          |
            v                          v
   Score > Threshold            Score <= Threshold
   (Normal Traits)              (Anomaly Symptoms)
            |                          |
            v                          v
+--------------------+        +-----------------------------+
| STATUS: BENIGN     |        | BENIGN CONFIRMATION LAYER   |
| / SAFE             |        +-----------------------------+
+--------------------+                   |
            |                              |
            |                              v
            |                   +-----------------------------+
            |                   | ASCII > 85% & Entropy < 4.8 |
            |                   | (Standard Text Structure)    |
            |                   +-----------------------------+
            |                              |
            |                              v
            |                   +-----------------------------+
            |                   | STATUS: REVOKED / SAFE      |
            |                   +-----------------------------+
            |                              |
            |                              |
            |                   +-----------------------------+
            |                   | Non-Printable > 0.015       |
            |                   | & Entropy > 4.75            |
            |                   | (Asymmetric File)           |
            |                   +-----------------------------+
            |                              |
            |                              v
            |                   +-----------------------------+
            |                   | STATUS: MALWARE / THREAT    |
            |                   +-----------------------------+
            |                              |
            +--------------+---------------+
                           |
                           v
                +-------------------------+
                | SYSTEM LOG & REPORTING  |
                +-------------------------+

## 3. Development Methodology Process
The development was systematically carried out through several *MLOps* stages:
1. **Extraction Analysis:** Exploring data distribution and crafting features.
2. **Model Training:** Constructing the base algorithm and testing in the lab with the *5-Fold Cross-Validation* method.
3. **Real Data Testing:** Evaluating using real-world data to calculate *False Positive* occurrences and measure resulting generalizations/gaps.
4. **Model Stabilization (Production Phase):** The primary focus was drawn into maintaining the *Precision* vs *Recall* balance, suppressing unnecessary alarms for a long-term reliable lifecycle.

## 4. Final Model Evaluation Results
The final outcome of the architecture (*Version 1.0 Stable*) demonstrates highly reliable operational metrics:
- **High and Stable Precision:** Proving minimum amounts of *False Positives* (highly immune to false alarms).
- **Stable Recall:** Successfully holding back malicious anomalies from escaping.
- **Small Evaluation vs Real Gap:** The algorithm grasps the generalization of actual data patterns flawlessly without any symptom of *Overfitting*.
- **Zero Data Drift:** There is no prominent deviation or distortion found in data.
- **Cross-Environment Consistency:** The model strictly implements the exact same threshold during executive training and operational inference.

The system is categorized as a robust monitoring system, highly durable, and readily available for production monitoring.

## 5. The Use of AI Over the Development
In iterating the software arrangement process, I utilized Artificial Intelligence/AI (such as ChatGPT) as a *Tooling Support* instrument relating strictly to technical operations strings:
- Brainstorming structural ideas / logic architecture strategies.
- Code Refactoring to deliver a *Clean Code Architecture*.
- Drafting document structures aesthetically.
- Validating the introspective logic behind metric scaling formulas.

However, **all core designs, testing methodologies, result interpretation and evaluation (*Trade-offs*), to any final technical decisions were independently evaluated and verified**. The AI stands merely as a medium that boosts operational productivity (*Typing Assistant*), whilst clinical architectural decisions were entirely comprehended and commanded consciously by myself as the lead developer.

## 6. Conclusion
This system has been successfully actualized as a fundamental layer for a Hybrid Anomaly inspection driven by static analysis. Highly stable, with zero runtime threshold miscalculations, and prepared as a blueprint for endpoint development. This project demonstrates an elaborate understanding of MLOps logic; being capable of standing as an early threat layer independently without functioning like a heavy signature-based Antivirus.

---

## 7. H.E.A.D.S Usage Manual
Below is a practical instructional guide to executing the V1.0 model in your terminal:

### A. Run Full Pipeline (Train & Test)
Use this command to retrain the model with updated data, monitor Cross-Validation accuracy, and run immediate inferences:
```bash
python tools/run_pipeline.py
```

### B. Audit the Model's Health
To dive specifically into the False Positives statistics, current threshold active evaluations, and caught malware samples details:
```bash
python tools/audit_model.py
```

### C. Open Live Monitoring Dashboard
As a System Administrator, you can deploy a live-terminal view to review the overall system detection logic at runtime:
```bash
python tools/dashboard_monitor.py
```
*(Press `CTRL+C` on your keyboard to log off and exit the dashboard).*
