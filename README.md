# 🌍 Macro-Economic SOM Analysis & Interactive Dashboard

**COMP5048 Assignment 2** — Ting-Chen (Selina) Chen

This repository implements a full visual-analytics pipeline using Self-Organizing Maps (SOM), data-driven period detection, macro-economic group classification, and a custom interactive dashboard to analyze **217 countries** across **2000–2024**.

---

## 📦 Project Structure

```
project/
├── 5048_data.xlsx                              # Raw economic indicator data
├── SOM_analysis_with_attribute_importance.ipynb # Jupyter notebook for analysis
├── 5048_groupings_with_importance.py           # Main Python implementation
├── interactive_dashboard.html                  # Interactive visualization dashboard
├── 5048_Ting-Chen.docx                         # Full project report
└── README.md                                   # This file
```

---

## 🚀 Overview

This project answers two core research questions:

**1️⃣ How did global economic structures evolve from 2000–2024?**  
**2️⃣ What combination of indicators best explains country grouping within each period?**

To answer these questions, the project integrates:

- ✅ **SOM topology learning** for pattern discovery
- ✅ **Automatic structural-break detection** for period identification
- ✅ **Rule-based macroeconomic classification** (Groups A/B/C/D)
- ✅ **ANOVA-based attribute importance ranking**
- ✅ **Temporal group evolution visualization**
- ✅ **Interactive exploration dashboard** (HTML + CSS + Plotly)

---

## 🧠 Method Summary

### 1. Data Preprocessing

**Techniques Applied:**
- ✔ IQR-based scaling for robustness
- ✔ Log transforms for skewed distributions
- ✔ Missing-value retention (no imputation)
- ✔ Yearly feature matrices for 217 countries

**Implemented in:** `5048_groupings_with_importance.py`

---

### 2. Period Detection (Automatic)

A **20×20 SOM** is trained on all observations. For each pair of consecutive years, a composite structural-change score is calculated using:

- **Euclidean distance**
- **Cosine distance**  
- **Wasserstein distance**

Top breakpoints identify **7 distinct periods** in global economic evolution.

---

### 3. Country Grouping (Macro Groups A/B/C/D)

Countries are classified into four macro-economic groups based on economic indicators:

| Group | Criteria | Definition |
|-------|----------|------------|
| **A** | GDP/capita > $25,000 | High-Income Advanced |
| **B** | $8,000–$25,000 & stable inflation | Emerging & Upper Middle |
| **C** | Below threshold | Developing & Lower-Middle |
| **D** | Very high inflation or unemployment | High-Inflation Vulnerable |

These rules override pure SOM clusters to ensure **realistic economic classifications**.

---

### 4. Attribute Importance (ANOVA)

Each of the **7 economic indicators** is ranked by:

- **F-statistic** (between-group variance)
- **Eta-squared** (effect size)
- **Standardized range** (discriminative power)

This reveals which economic attributes best discriminate macro-groups in each period.

---

## 🎨 Interactive Dashboard

**File:** `interactive_dashboard.html`

### Features:

- 📊 **Period selector** (P1–P7)
- 📈 **Macro-group distribution** bar chart
- 🔍 **Attribute importance** visualization
- 🌐 **Country list** with group highlights
- 📉 **Group evolution** over time
- 💹 **Economic indicator** comparisons

Built fully in **HTML + CSS + Plotly** (no external frameworks required).

---

## 🛠 How to Run

### 1. Install Dependencies

```bash
pip install pandas numpy scipy minisom plotly
```

### 2. Run the SOM Analysis

```bash
python 5048_groupings_with_importance.py
```

This will:
- Load and preprocess the data
- Train the SOM model
- Detect periods
- Classify countries into groups
- Calculate attribute importance

### 3. Launch the Dashboard

Simply open the file in any modern web browser:

```bash
# On macOS
open interactive_dashboard.html

# On Linux
xdg-open interactive_dashboard.html

# On Windows
start interactive_dashboard.html
```

Or just **double-click** `interactive_dashboard.html`

---

## 📊 Key Outputs

The analysis produces:

1. **Period boundaries** (2000–2024 divided into 7 periods)
2. **Country classifications** (A/B/C/D groups per period)
3. **Attribute importance rankings** (which indicators matter most)
4. **Interactive visualizations** (exploration dashboard)
5. **Temporal evolution patterns** (how countries transition between groups)

---

## 📝 Contribution Summary

*From: `5048_Ting-Chen.docx`*

### Task 1: Non-Visualization
- ✅ Data cleaning & preprocessing
- ✅ SOM pipeline design & training
- ✅ Period detection algorithm
- ✅ Attribute importance analysis
- ✅ Macro-group rule design
- ✅ All programming & debugging
- ✅ Dashboard construction in HTML/CSS

### Task 2: Visualization
- ✅ Period change-score charts
- ✅ Silhouette-score visual justification
- ✅ Attribute discriminative heatmap
- ✅ Temporal stacked bar charts
- ✅ Dashboard interaction design
- ✅ HCI-based refinement (Fitts' Law, Info-Seeking Mantra)

---

## 📚 References

Full references are included in the project report: `5048_Ting-Chen.docx`

Key methodologies:
- Self-Organizing Maps (Kohonen, 1982)
- Structural break detection
- ANOVA-based feature importance
- Interactive dashboard design principles

---

## 🎓 Academic Context

**Course:** COMP5048 – Visual Analytics  
**Institution:** University of Sydney  
**Year:** 2024  
**Author:** Ting-Chen (Selina) Chen

---

## 🎉 Acknowledgements

This project was completed as part of COMP5048 – Visual Analytics at the University of Sydney. Special thanks to the course instructors and teaching team for their guidance on visual analytics methodologies and best practices.

---

## 📄 License

This project is submitted as academic coursework for COMP5048. Please respect academic integrity policies when referencing or using this work.

---

## 📧 Contact

For questions or collaboration opportunities, please refer to the course submission portal or contact through official university channels.

---

**⭐ If you find this project useful, please consider starring the repository!**
