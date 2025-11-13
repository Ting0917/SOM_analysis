🌍 Macro-Economic SOM Analysis & Interactive Dashboard

COMP5048 Assignment 2 — Ting-Chen (Selina) Chen



This repository implements a full visual-analytics pipeline using Self-Organizing Maps (SOM), data-driven period detection, macro-economic group classification, and a custom interactive dashboard to analyze 217 countries across 2000–2024.

📦 Project Structure
📁 project/
 ├── 5048_data.xlsx
 ├── SOM_analysis_with_attribute_importance.ipynb
 ├── 5048_groupings_with_importance.py
 ├── interactive_dashboard.html
 ├── 5048_Ting-Chen.docx
 └── README.md

🚀 Overview

This project answers two core questions:

1️⃣ How did global economic structures evolve from 2000–2024?
2️⃣ What combination of indicators best explains country grouping within each period?

To answer these, the project integrates:

SOM topology learning

Automatic structural-break detection

Rule-based macroeconomic classification (A/B/C/D)

ANOVA-based attribute importance ranking

Temporal group evolution visualisation

Interactive exploration dashboard (HTML + CSS + Plotly)

🧠 Method Summary
1. Data Preprocessing

✔ IQR-based scaling
✔ Log transforms
✔ Missing-value retention (no imputation)
✔ Yearly feature matrices for 217 countries

Implemented in: 5048_groupings_with_importance.py


5048_groupings_with_importance

2. Period Detection (Automatic)

A 20×20 SOM is trained on all observations.
For each pair of consecutive years, a composite structural-change score is calculated using:

Euclidean distance

Cosine distance

Wasserstein distance

Top breakpoints → 7 periods.

3. Country Grouping (Macro Groups A/B/C/D)
Group	Criteria	Definition
A	GDP/cap > 25,000	High-Income Advanced
B	8,000–25,000 & stable inflation	Emerging & Upper Middle
C	Else	Developing & Lower-Middle
D	Very high inflation or unemployment	High-Inflation Vulnerable

These rules override pure SOM clusters → realistic classifications.

4. Attribute Importance (ANOVA)

Each of the 7 indicators is ranked by:

F-statistic

Eta-squared

Standardized range

This reveals which economic attributes discriminate macro-groups in each period.

🎨 Interactive Dashboard

File: interactive_dashboard.html


interactive_dashboard

Features:

Period selector (P1–P7)

Macro-group distribution bar chart

Attribute importance visualization

Country list with group highlights

Group evolution over time

Economic indicator comparisons

Built fully in HTML + CSS + Plotly (no frameworks).

🛠 How to Run
Install dependencies:
pip install pandas numpy scipy minisom plotly

Run the SOM analysis:
python 5048_groupings_with_importance.py

Launch dashboard:

Just open:

interactive_dashboard.html

📝 Contribution Summary

From: 5048_Ting-Chen.docx


5048_Ting-Chen

Task 1: Non-Visualisation

Data cleaning & preprocessing

SOM pipeline design & training

Period detection algorithm

Attribute importance analysis

Macro-group rule design

All programming & debugging

Dashboard construction in HTML/CSS

Task 2: Visualisation

Period change-score charts

Silhouette-score visual justification

Attribute discriminative heatmap

Temporal stacked bar charts

Dashboard interaction design

HCI-based refinement (Fitts’ Law, Info-Seeking Mantra)

📚 References

Full references included in report: 5048_Ting-Chen.docx


5048_Ting-Chen

🎉 Acknowledgements

Project for COMP5048 – Visual Analytics, University of Sydney.
