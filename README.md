# Urban Intelligence Platform
**Big Data & Network Analytics for Sustainable City Systems**

A systems-oriented urban analytics platform that processes large-scale city data, applies network science and probabilistic simulation, and visualizes systemic urban behavior to support data-driven and sustainable urban planning.

Demo Video: [Click here!](https://drive.google.com/file/d/125NDxvzLwz9Ek9Esq88Iy7qxummDFP_4/view?usp=sharing)

---

## Overview
This project implements an **end-to-end Big Data analytics pipeline** for urban traffic systems, combining distributed data processing, network analysis, and simulation-based risk assessment.  
It emphasizes **systems thinking, interpretability, and decision-oriented analytics**, aligning with MSc-level Computer Science expectations.

---

## Key Features
- **Big Data Pipeline:** HDFS-based medallion architecture (Bronze / Silver / Gold)
- **Network Analysis:** Similarity-based graph modeling, centrality metrics, community detection
- **Monte Carlo Simulation:** 10,000 iterations for probabilistic congestion risk estimation
- **Factor Analysis:** Latent factor extraction for weather–traffic relationships
- **Interactive Dashboard:** Streamlit-based interface with 7 analytical views

---

## Dataset (Summary)
- **Region:** London Metropolitan Area (8 boroughs)
- **Scale:** 5,000 raw records → 4,794 cleaned → 586 merged observations
- **Analytics:**  
  - 10,000 Monte Carlo simulations  
  - 9-edge similarity network  
  - 3 detected communities

---

## Key Results
- **Weather Impact:**  
  - Strong winds → 72.9% congestion risk  
  - Heavy rain → 55.4% of simulated scenarios  
  - Accident risk increases by **5.75×** under adverse weather
- **Network Structure:**  
  - Most influential areas (PageRank): Westminster, Hammersmith, Camden  
  - Modularity: **0.215** (stable community structure)
- **Traffic Patterns:**  
  - Peak vulnerability during morning (7–9 AM) and evening (5–7 PM) rush hours  
  - 3 latent factors explain **42.5%** of traffic–weather variance

---

## Network Analysis (Technical Summary)
Urban areas are modeled as nodes in a **top-K similarity graph** based on combined traffic and weather correlations.  
Centrality metrics (PageRank, Betweenness, Closeness) identify **systemic intervention points**, while community detection reveals coordinated policy zones.

**Full methodology & justification:** `docs/network-analysis.md`

---

## Technology Stack
- **Big Data:** Apache Spark, Hadoop HDFS, Parquet, MinIO  
- **Analytics:** Python, NumPy, Pandas, SciPy, Factor Analysis  
- **Network Science:** NetworkX, PageRank, Community Detection  
- **Visualization:** Streamlit, Plotly  
- **Infrastructure:** Docker, Docker Compose

---

## System Architecture
- **Bronze:** Raw weather & traffic data (HDFS)
- **Silver:** Cleaned and validated Parquet datasets
- **Gold:** Analytics-ready features, simulations, and network models

The architecture is designed for scalability and can be extended to real-time smart-city data streams.

**Detailed architecture:** `docs/system-architecture.md`

---

## How to Run
```bash
pip install -r requirements.txt
docker-compose up -d
streamlit run dashboard.py
