# EcoTariff

**Repository for cleaning, structuring, and analyzing historical electricity tariffs. Focused on tariffs (residential, commercial, industrial) with monthly resolution and multi-utility coverage.**

---

## Overview

EcoTariff is a repository for collecting, cleaning, and analyzing historical electricity tariff data from different Costa Rican distribution utilities. The project covers tariffs from **2013 to present**, with monthly resolution and detailed breakdowns by tariff type, consumption blocks, and components (energy, power, others).

The goal is to provide:

* Transparent access to tariff data.
* Tools for exploratory data analysis (EDA).
* Insights on tariff evolution, trends, and comparative analysis between utilities and customer classes.

---

## Repository Structure

```
EcoTariff/
│
├── data/              # Raw and processed datasets
│   ├── raw/           # Original structured dictionaries
│   └── processed/     # Flattened & cleaned CSV/Parquet files
│
├── notebooks/         # Jupyter notebooks for EDA & analysis
│   ├── 01_cleaning.ipynb
│   ├── 02_exploration.ipynb
│   └── 03_visualization.ipynb
│
├── src/               # Python scripts for data wrangling and utils
│   ├── flatten.py
│   ├── cleaning.py
│   └── viz.py
│
├── results/           # Figures, charts, summary tables
│
├── docs/              # Documentation and reports
│
└── README.md          # Project description
```

---

## Key Features

* Historical monthly tariffs since **2013**.
* Multiple **tariff types** (residential, general, industrial, medium voltage, EV charging, etc.).
* **Block tariffs** with detailed structure (0–140 kWh, >3000 kWh, etc.).
* Comparison across **distribution utilities**.
* Data pipeline to transform nested dictionaries into tidy DataFrames.
* Reproducible analysis via Jupyter notebooks.

---

## Example Analyses

* Year-over-year tariff changes per utility and tariff type.
* Price evolution adjusted for inflation.
* Comparison of residential vs industrial trends.
* Impact assessment for different consumption profiles (150 kWh, 500 kWh, 3000 kWh, etc.).
* Heatmaps of monthly variation.
* Forecasting and clustering of tariff behavior.

---

## Getting Started

### Requirements

* Python 3.9+
* pandas, numpy, matplotlib, seaborn, jupyter
* (optional) statsmodels, prophet, scikit-learn for advanced analysis

### Installation

Clone the repository:

```bash
git clone https://github.com/MarioPeraltaA/EcoTariff.git
cd EcoTariff
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Jupyter notebooks:

```bash
jupyter notebook notebooks/
```

---

## Data Sources

The dataset is built from structured tariff dictionaries provided for each distribution utility. Tariff codes include residential (T-RE, T-REH), commercial (T-CO), industrial (T-IN), medium voltage (T-MT, T-MTB, T-MT69), EV charging (T-VE, T-BE), and others.

---

## License

MIT License. Feel free to use, modify, and distribute with attribution.

---

## Contributing

Contributions are welcome! Please open an issue or pull request for improvements, bug fixes, or new analyses.

---

## Contact

Maintainer: *Mario R.*
Email: *[mario.peralta@ieee.org](mailto:mario.peralta@ieee.org)*

---


> *EcoTariff aims to support transparent, data-driven discussions around electricity pricing and its impact on consumers, regulators, and the energy transition.*
