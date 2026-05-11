# PPT Content — Network Intrusion Detection System

Copy and paste the content below into Gemini with the prompt: *"Create a professional PowerPoint presentation with these slides. Use modern styling, clean layouts, data visualizations (charts/tables)."*

---

## Slide 1: Title Slide
**Title:** AI-Powered Network Intrusion Detection System
**Subtitle:** Machine Learning-Based Threat Classification Using NSL-KDD
**Footer:** Intrusion Detection Project | 2026

---

## Slide 2: Problem Statement
**Title:** The Challenge

- Network attacks are evolving rapidly — traditional signature-based IDS cannot detect novel/zero-day attacks
- Organizations need ML-based solutions that generalize to unseen attack patterns
- Key requirements:
  - Real-time packet classification (normal vs attack)
  - Attack type identification (DoS, Probe, R2L, U2R)
  - High accuracy with low false alarm rate
  - GPU-accelerated inference for production use

---

## Slide 3: Dataset — NSL-KDD
**Title:** Dataset Overview

- **Benchmark:** NSL-KDD (improved version of KDD Cup 1999)
- **Training set:** 125,973 records
- **Test set:** 22,544 records
- **Features:** 41 network traffic features
  - 38 numeric (duration, byte counts, error rates, etc.)
  - 3 categorical (protocol_type, service, flag)
- **Label distribution:**
  - Training: 67,343 normal + 58,630 attack
  - Testing: 9,711 normal + 12,833 attack
- **Attack categories (4 types):**
  - DoS: back, land, neptune, pod, smurf, teardrop, apache2, udpstorm, etc.
  - Probe: satan, ipsweep, nmap, portsweep, mscan, saint
  - R2L: guess_passwd, ftp_write, imap, phf, snmpguess, httptunnel, etc.
  - U2R: buffer_overflow, loadmodule, rootkit, perl, sqlattack, etc.
- **Challenge:** Test set contains novel attack types NOT seen in training (e.g., mscan, snmpguess, httptunnel, saint, apache2)

---

## Slide 4: Feature Engineering Pipeline
**Title:** Feature Engineering (17 New Features)

**Objective:** Enhance model discriminability by creating domain-relevant features

**Engineered Features:**

| Feature | Formula | Purpose |
|---------|---------|---------|
| src_dst_byte_ratio | src_bytes / dst_bytes | Traffic asymmetry |
| total_bytes | src_bytes + dst_bytes | Total data volume |
| count_srv_ratio | count / srv_count | Connection density |
| same_srv_diff_host | same_srv_rate × srv_diff_host_rate | Service scanning |
| dst_host_conn_ratio | dst_host_count / dst_host_srv_count | Host connection pattern |
| auth_fail_ratio | num_failed_logins / logged_in | Auth attack pattern |
| root_access_score | root_shell × 2 + su_attempted | Privilege escalation |
| file_op_score | num_file_creations + num_access_files | File access pattern |
| compromised_score | num_compromised + num_root + root_shell | Host compromise |
| hot_login_ratio | hot / logged_in | Hot indicator density |
| urgent_wrong | urgent + wrong_fragment | Anomalous flags |
| num_shells_access | num_shells + num_access_files | Shell access pattern |
| duration_byte_ratio | total_bytes / duration | Throughput rate |
| protocol_service | protocol_type + "_" + service | Protocol-service combo |
| protocol_flag | protocol_type + "_" + flag | Protocol-flag combo |

**Total after one-hot encoding:** 220 features (51 numeric + 169 dummies)

---

## Slide 5: Model Architecture — Two-Stage Pipeline
**Title:** Model Architecture

```
                  ┌──────────────────────┐
                  │  Raw Packet Features │
                  │    (41 features)     │
                  └──────────┬───────────┘
                             │
                  ┌──────────▼───────────┐
                  │  Feature Engineering │
                  │   (+17 features)     │
                  └──────────┬───────────┘
                             │
                  ┌──────────▼───────────┐
                  │  One-Hot Encoding    │
                  │  (5 cols → 220 feat) │
                  └──────────┬───────────┘
                             │
                  ┌──────────▼───────────┐
                  │  StandardScaler      │
                  └──────────┬───────────┘
                             │
                ┌────────────▼────────────┐
                │  Stage 1: Binary        │
                │  XGBoost Classifier     │
                │  (normal vs attack)     │
                └────────────┬────────────┘
                             │
                    ┌────────▼────────┐
                    │  Attack?        │
                    └───┬────────┬────┘
                   No   │        │  Yes
                        │        │
                 ┌──────▼──┐  ┌──▼──────────────┐
                 │ normal  │  │ Stage 2: Multi-  │
                 └─────────┘  │ class XGBoost    │
                              │ (DoS/Probe/R2L/  │
                              │  U2R/normal)     │
                              └──────────────────┘
```

**Key design decision:** Two separate models — a binary classifier optimized for detection, and a multi-class classifier for attack typing. This keeps the binary decision clean while preserving attack classification.

---

## Slide 6: Model 1 — Binary XGBoost (GPU)
**Title:** Binary Classifier — XGBoost GPU

**Configuration:**
- Algorithm: XGBoost (Gradient Boosted Decision Trees)
- Hardware: GPU-accelerated (CUDA)
- Trees: 5,000 estimators
- Max depth: 10
- Learning rate: 0.01
- Early stopping: 100 rounds
- Sample weighting: Balanced class weights

**Decision Threshold:** 0.001 (ultra-low, prioritized attack capture)

**Results on NSL-KDD Test Set:**

| Metric | Value |
|--------|-------|
| Accuracy | **90.57%** |
| Attack Detection (TPR) | 89.6% |
| False Alarm Rate (FPR) | 8.2% |
| Precision | 92.1% |
| F1-Score | 0.908 |

**Why threshold 0.001?** In intrusion detection, missing an attack is far worse than a false alarm. A very low threshold ensures maximum attack capture (89.6%) while the model's high confidence on normal traffic keeps false alarms manageable (8.2%).

---

## Slide 7: Model 2 — Multi-Class XGBoost (GPU)
**Title:** Attack Type Classifier — Multi-Class XGBoost (GPU)

**Configuration:**
- Algorithm: XGBoost with softprob objective
- Hardware: GPU-accelerated (CUDA)
- Trees: 2,000 estimators
- SMOTE oversampling to handle class imbalance
- 5 classes: normal, DoS, Probe, R2L, U2R

**Results on NSL-KDD Test Set:**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| DoS | 0.94 | 0.95 | 0.95 |
| Probe | 0.86 | 0.81 | 0.83 |
| R2L | 0.56 | 0.58 | 0.57 |
| U2R | 0.64 | 0.24 | 0.35 |
| Normal | 0.95 | 0.95 | 0.95 |

**Overall Accuracy: ~79%**

**Challenge:** R2L and U2R are severely underrepresented in training (995 and 52 samples vs 67,343 normal). SMOTE helps but novel attack variants in the test set (e.g., httptunnel, snmpguess) make detection difficult.

---

## Slide 8: Model 3 — Random Forest
**Title:** Random Forest Baseline

**Configuration:**
- Algorithm: Random Forest (Bagged Decision Trees)
- Trees: 200 estimators
- Max depth: 20
- Min samples leaf: 5
- Parallel processing: 8 cores

**Results on NSL-KDD Test Set (Default Threshold 0.50):**

| Metric | Value |
|--------|-------|
| Accuracy | 76.73% |
| Attack Detection (TPR) | 61.1% |
| False Alarm Rate (FPR) | 2.6% |

At default 0.5 threshold, RF is very conservative — it rarely flags attacks (low FPR), but misses 39% of them.

**Results with Optimized Threshold (0.02):**

| Metric | Value |
|--------|-------|
| Accuracy | **92.37%** |
| Attack Detection (TPR) | **98.2%** |
| False Alarm Rate (FPR) | 15.3% |

Lowering the threshold to 0.02 dramatically improves attack capture (98.2%) at the cost of more false alarms (15.3%). RF with tuned threshold achieves the highest raw accuracy among all models.

---

## Slide 9: Comparison — All Models
**Title:** Model Comparison

| Aspect | XGBoost GPU | Random Forest | Linear Regression |
|--------|-------------|---------------|-------------------|
| **Binary Accuracy** | 90.57% | **92.37%** | 82.27% |
| **Attack Detection** | 89.6% | **98.2%** | 82.5% |
| **False Alarms** | **8.2%** | 15.3% | 18.0% |
| **Multi-class Accuracy** | **~79%** | — | 70.80% |
| **Training Time** | ~15 min | ~7 sec | ~12 sec |
| **Inference Speed** | ~1ms | ~0.3ms | ~0.1ms |
| **Threshold** | 0.001 | 0.02 | 0.07 |

**Trade-off Analysis:**

| Priority | Best Model |
|----------|-----------|
| Highest accuracy | **Random Forest** (92.37%) |
| Lowest false alarms | **Random Forest at 0.50** (2.6%) but misses 39% of attacks |
| Best attack capture + low false alarms | **XGBoost GPU** (89.6% capture, 8.2% FPR) |
| Fastest training | **Random Forest** (7 seconds) |
| Interpretability | **Linear Regression** (coefficient weights) |

**Takeaway:** XGBoost GPU is the primary production model — best balance of attack capture (89.6%) and low false alarms (8.2%). Random Forest with tuned threshold catches the most attacks (98.2%) but generates nearly double the false alarms. Linear Regression serves as a fast, interpretable baseline.

---

## Slide 9: Web Dashboard — Flask Application
**Title:** Interactive Web Dashboard

**Technology Stack:**
- Backend: Flask (Python) REST API
- Frontend: HTML5, CSS3, Vanilla JavaScript
- Database: SQLite
- Model serving: XGBoost GPU inference

**API Endpoints:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/send_traffic` | POST | Simulate & classify network packet |
| `/api/stats` | GET | Retrieve logs, accuracy, attack distribution |

**Features:**
- Live traffic simulation (normal/attack/category-specific)
- Real-time dashboard updates every 1 second
- Attack type breakdown visualization
- Model accuracy tracking
- SQLite logging of all predictions

---

## Slide 10: Live Dashboard Demo
**Title:** Dashboard Walkthrough

**Metrics Cards (top row):**
1. Total Packets — number of classified requests
2. Attacks Detected — malicious packets flagged
3. Normal Traffic — legitimate packets
4. Live Accuracy — real-time prediction accuracy

**Log Table:**
- Columns: Log ID, Timestamp, True Traffic Type, AI Prediction, Attack Type, Verdict
- Color-coded badges (red = attack, green = normal, blue = correct/wrong)
- Auto-refresh every 1 second

**Traffic Emulator Panel:**
- "Send Normal Packet" button
- "Launch Attack" button
- Hidable panel with toggle button
- Standalone CLI scripts for automated testing:
  - `python scripts/sim_http.py -n 50` (DoS simulation)
  - `python scripts/sim_probe.py` (Probe attack)
  - `python scripts/sim_r2l.py` (R2L attack)
  - `python scripts/sim_u2r.py` (U2R attack)

---

## Slide 11: Project Architecture
**Title:** System Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                     User / Client                         │
│  Browser (Dashboard)     CLI Scripts (sim_*.py)          │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP REST API
                         ▼
┌─────────────────────────────────────────────────────────┐
│                   Flask Web Server                        │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  /api/send_  │  │  /api/stats  │  │  /dashboard   │  │
│  │  traffic     │  │              │  │  /             │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬────────┘  │
│         │                 │                 │            │
└─────────┼─────────────────┼─────────────────┼────────────┘
          │                 │                 │
    ┌─────▼─────┐     ┌────▼────┐      ┌─────▼─────┐
    │ Preprocess│     │ SQLite  │      │  HTML     │
    │ Pipeline  │     │ DB      │      │ Templates │
    └─────┬─────┘     └─────────┘      └───────────┘
          │
    ┌─────▼──────────────────┐
    │  Model Inference       │
    │  ┌──────────────────┐  │
    │  │ Binary XGBoost   │  │
    │  │ (GPU) threshold  │  │
    │  │ = 0.001          │  │
    │  └────────┬─────────┘  │
    │           │ attack      │
    │    ┌──────▼──────────┐ │
    │    │ Multi-class     │ │
    │    │ XGBoost (GPU)   │ │
    │    │ → DoS/Probe/    │ │
    │    │   R2L/U2R       │ │
    │    └─────────────────┘ │
    └────────────────────────┘
```

---

## Slide 13: Results Summary
**Title:** Key Results

| Model | Binary Accuracy | Attack Recall | False Alarm Rate |
|-------|----------------|---------------|------------------|
| XGBoost GPU (Primary) | 90.57% | 89.6% | **8.2%** |
| Random Forest (tuned) | **92.37%** | **98.2%** | 15.3% |
| Linear Regression | 82.27% | 82.5% | 18.0% |

**Attack Type Classification (XGBoost Multi-Class):**

```
DoS:    94% precision, 95% recall  ───── Excellent
Probe:  86% precision, 81% recall  ───── Good
R2L:    56% precision, 58% recall  ───── Fair (limited training data)
U2R:    64% precision, 24% recall  ───── Poor (extreme class imbalance)
Normal: 95% precision, 95% recall  ───── Excellent
```

---

## Slide 13: Challenges & Limitations
**Title:** Challenges

1. **Class Imbalance:** R2L (995 train samples) and U2R (52 train samples) are severely underrepresented. Novel variants in test set make this worse.

2. **Novel Attack Types:** Test set contains attacks not seen in training (mscan, snmpguess, httptunnel, saint, apache2). Model must generalize to unseen patterns.

3. **R2L Detection Gap:** Even XGBoost misses ~42% of R2L attacks because these attacks mimic normal user behavior (e.g., password guessing via SSH).

4. **False Alarm Tradeoff:** Ultra-low threshold (0.001) catches 89.6% of attacks but produces 8.2% false alarms. Further optimization could reduce this.

---

## Slide 14: Future Work
**Title:** Future Improvements

1. **Deep Learning:** Replace XGBoost with a neural network (e.g., Autoencoder + classifier) for better feature learning

2. **Real-time Streaming:** Integrate with Apache Kafka/Spark for live packet capture from network interfaces (pcap)

3. **Active Learning:** Continuously retrain on novel attack patterns discovered in production

4. **Explainability:** Add SHAP/LIME explanations for each prediction

5. **Ensemble:** Combine XGBoost + Neural Network predictions for robust detection

6. **Adversarial Robustness:** Test against adversarial packet manipulations

---

## Slide 15: Thank You
**Title:** Thank You

**Links:**
- GitHub Repository: [link]
- Demo: http://localhost:5000

**Contact:** [Your Name / Team]
