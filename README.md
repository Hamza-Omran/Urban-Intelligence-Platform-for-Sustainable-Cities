# Urban Intelligence Platform

Big Data and Network Analytics for Sustainable City Systems

A data-driven urban analytics platform that processes large-scale city data, applies network analysis and simulations, and visualizes systemic urban behavior to support sustainable urban planning decisions.

Version 2.0 | Python 3.8+ | MIT License

## Overview

The Urban Intelligence Platform demonstrates a complete data analytics pipeline for sustainable cities, processing large-scale urban data through Big Data infrastructure, applying network analysis to reveal systemic patterns, and visualizing insights to support policy decisions.

This project is framed at MSc level, emphasizing systems thinking, interpretability, and decision-oriented analytics rather than pure prediction accuracy.

### Key Features

- Big Data Pipeline: HDFS medallion architecture (Bronze/Silver/Gold) with Spark-compatible processing
- Network Analysis: Graph modeling with centrality metrics and community detection
- Monte Carlo Simulation: 10,000 iterations for probabilistic risk assessment
- Factor Analysis: Dimensionality reduction revealing weather-traffic relationships
- Interactive Dashboard: Professional Streamlit interface with 7 analytical views

### Dataset

- Location: London Metropolitan Area (8 boroughs)
- Scale: 5,000 raw records, 4,794 cleaned, 586 merged observations
- Analytics: 10,000 Monte Carlo simulations, 9-edge network graph, 3 communities

## Sustainability Relevance

This project addresses critical urban sustainability challenges:

Climate Adaptation: Quantifies weather impacts on mobility and enables climate-adaptive transportation policies

Emission Reduction: Identifies congestion hotspots for targeted interventions to reduce idle time and emissions

Resource Optimization: Network analysis prioritizes infrastructure investments for maximum systemic impact

Decision Support: Transforms raw urban data into actionable policy recommendations

This work aligns directly with the AI for Sustainable Societies programme by demonstrating how AI-enabled analytics can support climate-adaptive and socially responsible urban decision-making.

## Technology Stack

Big Data: Apache Spark, Hadoop HDFS, MinIO, Parquet
Network Science: NetworkX, PageRank, Community Detection
Analytics: NumPy, Pandas, SciPy, Factor Analysis
Visualization: Streamlit, Plotly, Seaborn

## System Architecture

The platform implements a medallion data architecture:

Bronze Layer: Raw data in HDFS (5,000 weather records, 5,000 traffic records)
Silver Layer: Cleaned Parquet files (4,794 records after validation)
Gold Layer: Analytics-ready aggregations (simulations, networks, factors)

Data Flow:
Raw CSV Data → Bronze Layer HDFS → Data Cleaning and Validation → Silver Layer Parquet → Feature Engineering → Gold Layer Analytics → Monte Carlo Simulation + Factor Analysis + Network Graph → Interactive Dashboard → Policy Recommendations

Scalability: While this implementation uses preprocessed datasets for reproducibility, the documented architecture scales naturally to real-time urban data streams for smart-city deployment.

## Network Analysis Methodology

### Graph Model

Nodes: 8 London boroughs
Edges: Top-2 similarity strategy (each area connects to its 2 most correlated areas)
Weight: 60% traffic correlation + 40% weather correlation (Pearson coefficients)

Edges represent statistically significant similarity in traffic-weather response patterns rather than direct physical connectivity, modeling systemic urban behavior.

Earlier sparse-network experiments produced fragmented communities; the final top-k similarity strategy yields three stable, interpretable clusters (modularity = 0.215).

### Centrality Metrics

PageRank: Systemic influence (high-impact intervention points)
Betweenness: Traffic flow control (bottleneck identification)
Closeness: Network accessibility (connectivity hubs)
Degree: Connection count (widespread influence areas)
Clustering: Local density (coordinated policy zones)

High PageRank indicates systemic influence across the urban network, not direct congestion severity. This measures where interventions have ripple effects.

### Community Detection

Greedy modularity optimization identifies 3 communities with:
- Similar traffic congestion patterns
- Correlated weather responses
- Shared temporal dynamics

Modularity: 0.215 (significant community structure)

Community Composition:
- Community 0: Southwark, Hammersmith, Westminster, Islington (4 areas)
- Community 1: Chelsea, Hackney, Camden (3 areas)
- Community 2: Kensington (1 area, isolated pattern)

## Key Results

### Weather Impact
- Strong winds: 72.89% congestion risk
- Heavy rain: 55.36% of scenarios
- Accident risk: 5.75× increase under adverse weather
- Temperature has independent effect (Factor 1 loading: -0.547)

### Network Structure
- Most influential areas (PageRank): Westminster (0.175), Hammersmith (0.174), Camden (0.171)
- Network density: 0.321 (well-connected systemic behavior)
- Communities: 3 stable clusters (modularity = 0.215)
- Edges: 9 connections across 8 boroughs
- Avg degree: 2.25

### Traffic Patterns
- Peak vulnerability: Morning (7-9 AM) and evening (5-7 PM) rush hours
- High-risk areas: Chelsea and Hackney show consistent congestion
- Factor variance: 3 factors explain 42.48% of weather-traffic relationships

## Installation and Setup

Prerequisites:
- Python 3.8+
- Docker and Docker Compose (for full infrastructure)
- 4GB RAM minimum

Installation Steps:

```bash
git clone <repository-url>
cd Final_Complete_Project

pip install -r requirements.txt

docker-compose up -d
```

## Running the Analysis Pipeline

Execute the complete data pipeline:

```bash
python3 Scripts/generate_weather_data.py
python3 Scripts/generate_traffic_data.py
python3 Scripts/data_merging.py
python3 Scripts/monte_carlo_simulation.py
python3 Scripts/factor_analysis.py
python3 Scripts/network_analysis.py
```

Launch the dashboard:

```bash
streamlit run dashboard.py
```

Access at http://localhost:8501

## Dashboard Features

The interactive dashboard includes 7 analytical views:

1. Overview: Project summary, pipeline status, key metrics
2. Data Cleaning: Quality statistics, distributions, outlier handling
3. Data Merging: Merge results, engineered features, temporal patterns
4. Monte Carlo Simulation: Probability distributions, scenario analysis, risk heatmaps
5. Factor Analysis: Factor loadings, variance explained, interpretations
6. Network Analysis: Interactive graph, centrality metrics, community detection
7. Recommendations: Policy suggestions for urban planners

## Policy Recommendations

1. Dynamic Speed Limits: Weather-responsive limits for wind speeds exceeding 50 km/h
2. Targeted Interventions: Prioritize high-PageRank areas (Westminster, Hammersmith, Camden)
3. Community Coordination: Implement policies within detected communities
4. Predictive Routing: Leverage simulation results for real-time traffic management
5. Infrastructure Investment: Focus on high-betweenness bottleneck areas

## Project Structure

```
Final_Complete_Project/
├── Data/
│   ├── bronze/          Raw CSV files in HDFS
│   ├── silver/          Cleaned Parquet files
│   └── gold/            Analytics results
├── Scripts/
│   ├── generate_weather_data.py
│   ├── generate_traffic_data.py
│   ├── data_merging.py
│   ├── monte_carlo_simulation.py
│   ├── factor_analysis.py
│   └── network_analysis.py
├── dashboard.py
├── requirements.txt
└── README.md
```

## Network Design Details

### Edge Creation Strategy

Top-K Similarity Approach: Each area connects to its top 2 most correlated areas

Rationale:
- Guarantees network connectivity
- Academically rigorous (avoids arbitrary threshold selection)
- Computationally efficient

Correlation Calculation:
For each pair of areas, calculate traffic correlation and weather correlation using Pearson coefficient, then combine with weights: edge_weight = 0.6 * traffic_corr + 0.4 * weather_corr

Absolute values are used to capture anti-correlations as meaningful patterns.

### Centrality Interpretations

PageRank Formula: Iterative algorithm computing node influence based on neighbor importance

Urban Interpretation:
- NOT a measure of direct congestion severity
- IS a measure of systemic influence across the urban network
- High PageRank means interventions here have ripple effects to connected areas

Betweenness Centrality: Measures how often a node lies on shortest paths between other nodes. Identifies bottleneck areas where traffic disruptions cascade.

Closeness Centrality: Average shortest path distance to all other nodes. Highlights accessibility and areas well-connected to entire network.

Degree Centrality: Number of direct connections. Identifies areas with widespread similar behavior patterns.

Clustering Coefficient: Density of connections among a node's neighbors. Identifies tight-knit traffic zones for coordinated policies.

### Network Statistics

- Nodes: 8 London boroughs
- Edges: 9 connections
- Density: 0.321 (32.1% of possible edges)
- Diameter: 3 (maximum shortest path length)
- Average Clustering: 0.147 (moderate local density)
- Modularity: 0.215 (significant community structure)
- Components: 2 (near-connected, Kensington isolated)

## Academic Justification

Why Top-K Instead of Threshold:

Threshold Approach Problems:
- Arbitrary cutoff value
- Data-dependent (fails with sparse temporal overlap)
- Risk of disconnected graph or degenerate structure

Top-K Benefits:
- Adaptive to data distribution
- Guaranteed minimum connectivity
- Established in network science literature
- Interpretable: each area linked to its most similar neighbors

Similar strategies used in protein interaction networks, social network analysis, and recommendation systems.


## Contact

For questions or collaboration opportunities contact via GitHub or email.

Built for AI for Sustainable Societies - Transforming Urban Data into Climate-Adaptive Decisions
