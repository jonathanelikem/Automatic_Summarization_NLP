[README.md](https://github.com/user-attachments/files/26447175/README.md)
# 🎓 Automatic Text Summarization — UG Business School Research Analysis

> **NLP-driven profiling of research output from the University of Ghana Business School (UGBS).**  
> This project automatically summarizes 140+ academic papers from 7 department heads (HODs), identifies research themes, compares three NLP models, and produces a cross-departmental analysis answering six research questions about the school's collective academic identity.

---

## 📌 Project Overview

| Item | Detail |
|---|---|
| **Institution** | University of Ghana Business School (UGBS) |
| **Course** | OMIS 405 — Business Analytics / NLP |
| **Scope** | 7 HODs · 20 papers each · 140 papers total |
| **NLP Models** | LDA · BERTopic · BART (T5 Abstractive) |
| **Accuracy Metrics** | Keyword Overlap % · ROUGE-1 F1 |
| **Output** | Excel workbook · 15+ visualisations · PowerPoint presentation |

---

## 🗂️ Repository Structure

```
Automatic_Summarization_NLP/
│
├── HOD_NLP_Summariser_v12.ipynb   ← Main Jupyter notebook (run this)
│
├── HOD_TXT_CORPUS/                ← Auto-created on first run
│   ├── Accounting - Prof. Godfred Matthew Yaw Owusu/
│   │   ├── 1 - Determinants of environmental disclosures.txt
│   │   └── ... (20 papers)
│   ├── Finance - Dr. Vera Fiador/
│   ├── HR - Obi Berko Obeng Damoah/
│   ├── HSM - Prof. Lily Yarney/
│   ├── Marketing - Prof. Mahmoud Abdulai/
│   ├── OMIS - Prof. Eric Afful Dadzie/
│   └── Public Admin - Prof. Albert Ahenkan/
│
├── UGBS_HOD_NLP_Analysis.xlsx     ← Auto-generated Excel output (6 sheets)
├── UGBS_Research_Analysis.pptx    ← Auto-generated PowerPoint (15 slides)
│
└── outputs/                       ← All saved visualisation charts (.png)
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9 or newer
- Jupyter Notebook or JupyterLab
- Git (to clone the repo)

### 1. Clone the repository
```bash
git clone https://github.com/jonathanelikem/Automatic_Summarization_NLP.git
cd Automatic_Summarization_NLP
```

### 2. Install dependencies
Run the first cell in the notebook, or manually:
```bash
pip install hdbscan --prefer-binary
pip install umap-learn --prefer-binary
pip install bertopic sentence-transformers --prefer-binary
pip install scikit-learn gensim nltk requests openpyxl pandas \
            matplotlib seaborn wordcloud transformers torch
```

### 3. Open and run the notebook
```bash
jupyter notebook HOD_NLP_Summariser_v12.ipynb
```
Then run cells **top to bottom** using `Shift + Enter`.

---

## 🚀 How the Corpus Loader Works

The notebook uses a smart two-path loader — **no manual setup needed**:

```
Run notebook
       ↓
Does HOD_TXT_CORPUS/ exist next to this notebook?
       ├── YES → Load from disk  (fast, no internet)
       └── NO  → Download from GitHub + save to disk
                         ↓
                 HOD_TXT_CORPUS/ created locally
                         ↓
                 Every run after → takes the fast path
```

To re-fetch fresh files from GitHub, simply **delete the `HOD_TXT_CORPUS/` folder** and re-run Step 3.

---

## 🤖 NLP Pipeline

```
Raw .txt papers
       ↓
  Text cleaning (noise removal, front-matter stripping, reference removal)
       ↓
  ┌──────────────────────────────────────────────────┐
  │  EXTRACTIVE                                      │
  │  ├── TF-IDF → top-5 sentence summary per paper  │
  │  ├── LDA    → 5 topic clusters per HOD           │
  │  └── BERTopic → semantic clustering (BERT embed) │
  │                                                  │
  │  ABSTRACTIVE                                     │
  │  └── BART (facebook/bart-large-cnn)              │
  │       → generates new summary sentences          │
  └──────────────────────────────────────────────────┘
       ↓
  Accuracy Validation (3 papers/HOD, seed=42)
  ├── Keyword Overlap %
  ├── ROUGE-1 F1 %
  ├── Hallucination detection (abstractive)
  └── Missing key point detection
       ↓
  Cross-HOD Analysis (6 research questions)
       ↓
  Excel export + visualisations
```

---

## 📊 Research Questions Answered

| # | Question | Method |
|---|---|---|
| **(a)** | What are the dominant research themes? | TF-IDF frequency · LDA topics · bigrams/trigrams |
| **(b)** | Is there a unifying school of thought? | Keyword intersection across HODs |
| **(c)** | Does research align with UG's mission? | Theoretical vs Applied signal scoring |
| **(d)** | What are the preferred research methods? | Quantitative / Qualitative / Design Science signal scoring |
| **(e)** | What are the emerging hot topics? | Signal detection in most-recent papers |
| **(f)** | Where do HOD research interests overlap? | Keyword overlap matrix · bubble chart |

---

## 📁 Excel Output — Sheet Guide

| Sheet | Contents |
|---|---|
| **Paper Summaries** | One row per paper — abstract, TF-IDF summary, LDA keywords, BERTopic keywords, word count |
| **Accuracy Validation** | 3 sampled papers per HOD — Keyword Overlap %, ROUGE-1 F1, hallucination flag, issues |
| **HOD Comparison Table** | One row per HOD — top themes, method, orientation, cross-HOD connections |
| **LDA Topics** | All 5 LDA topics per HOD with top keywords |
| **Model Comparison** | Per-paper overlap scores — LDA vs BERTopic vs T5, winner column |
| **T5 Abstractive Summaries** | All 140 BART-generated summaries |

---

## 📈 Key Findings

- **All 7 HODs** use predominantly **quantitative methods** (PLS-SEM, regression, panel data)
- Research is **applied** across the board — oriented toward real-world Ghanaian business and policy problems
- **Five research pillars** identified: Financial Behaviour · Corporate Governance · Ghana Development · Healthcare Delivery · Digital Innovation
- **BART (T5)** outperforms LDA and BERTopic on abstract accuracy (avg ~24% overlap vs ~4.5% for LDA)
- Strongest collaboration potential: **Accounting ↔ Finance**, **Finance ↔ HR**, **HSM ↔ OMIS**

---

## 🛠️ Known Limitations

| Issue | Status |
|---|---|
| BERTopic requires ≥ 5 papers per cluster | Fixed in v12 — `min_topic_size` dynamically scaled |
| `et` / `al` appearing in LDA topics | Fixed in v11 — custom stopword list passed to `CountVectorizer` |
| Windows MAX_PATH (260 char) filename limit | Fixed in v12 — filenames auto-shortened on save |
| BART cannot process > ~1024 tokens | By design — papers truncated to 3000 chars (~1024 tokens) |

---

## 🔧 Tech Stack

| Tool | Purpose |
|---|---|
| `bertopic` + `sentence-transformers` | Semantic topic modelling |
| `scikit-learn` (LDA) | Classical topic modelling |
| `transformers` (BART) | Abstractive summarisation |
| `gensim` | NLP utilities |
| `nltk` | Tokenisation, stopwords, lemmatisation |
| `openpyxl` | Excel export |
| `matplotlib` + `seaborn` + `wordcloud` | Visualisations |
| `pandas` + `numpy` | Data manipulation |
| `requests` | GitHub API corpus fetch |

---

## 📄 License

This project was developed as part of an academic assignment at the University of Ghana Business School. All paper texts remain the intellectual property of their respective authors and publishers.

---

## 👤 Author

**Jonathan Elikem**  
BSc Business Administration (Business Analytics)  
University of Ghana Business School  
