"""
Member 6 - Factor Analysis & Final Integration
============================================================

This script performs:
1. Loading simulation results from Member 5
2. Preparing features for factor analysis
3. Performing PCA and Factor Analysis
4. Identifying latent factors
5. Generating factor loadings and interpretations
6. Creating comprehensive insights report
7. Saving results to Gold layer

Author: Member 6
Date: December 2024
Project: Weather Impact on Urban Traffic Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.metrics import silhouette_score
from scipy.stats import zscore
import warnings
import os
import sys

warnings.filterwarnings('ignore')
np.random.seed(42)

# ================================================================
# CONFIGURATION
# ================================================================
class Config:
    """Configuration for Factor Analysis"""
    
    # Input paths
    SIMULATION_RESULTS_PATH = r"Data/gold/simulation_results.csv"
    
    # Output paths
    GOLD_DIR = r"Data/gold"
    FACTOR_LOADINGS_PATH = r"Data/gold/factor_loadings.csv"
    FACTOR_SCORES_PATH = r"Data/gold/factor_scores.csv"
    INTERPRETATION_REPORT_PATH = r"Data/gold/factor_analysis_interpretation.txt"
    
    # Analysis parameters
    N_FACTORS = 3  # Number of latent factors to extract
    RANDOM_SEED = 42


# ================================================================
# STEP 1: DATA LOADING AND PREPARATION
# ================================================================
def load_simulation_data():
    """Load simulation results from Member 5"""
    print("\n" + "="*70)
    print("STEP 1: LOADING SIMULATION RESULTS")
    print("="*70)
    
    try:
        df = pd.read_csv(Config.SIMULATION_RESULTS_PATH)
        print(f"Data loaded successfully!")
        print(f"  Records: {len(df):,}")
        print(f"  Columns: {len(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"ERROR: Simulation results not found at {Config.SIMULATION_RESULTS_PATH}")
        sys.exit(1)


def prepare_features(df):
    """Prepare features for factor analysis"""
    print("\n" + "="*70)
    print("STEP 2: PREPARING FEATURES FOR ANALYSIS")
    print("="*70)
    
    # Define feature groups as per project requirements
    weather_features = [
        'temperature_c',
        'humidity',
        'rain_mm',
        'wind_speed_kmh',
        'visibility_m',
        'air_pressure_hpa'
    ]
    
    traffic_features = [
        'vehicle_count',
        'avg_speed_kmh',
        'accident_count'
    ]
    
    # Check if air_pressure_hpa exists (might not be in merged dataset)
    if 'air_pressure_hpa' not in df.columns:
        print("  Note: air_pressure_hpa not found, excluding from analysis")
        weather_features.remove('air_pressure_hpa')
    
    # Combine all features
    all_features = weather_features + traffic_features
    
    # Extract feature matrix
    X = df[all_features].copy()
    
    print(f"\nSelected features:")
    print(f"  Weather features: {weather_features}")
    print(f"  Traffic features: {traffic_features}")
    print(f"  Total features: {len(all_features)}")
    
    # Check for missing values
    missing = X.isnull().sum()
    if missing.sum() > 0:
        print(f"\nMissing values detected:")
        print(missing[missing > 0])
        print("  Filling with median values...")
        X = X.fillna(X.median())
    
    print(f"\nFeature statistics:")
    print(X.describe())
    
    return X, all_features, weather_features, traffic_features


# ================================================================
# STEP 3: STANDARDIZATION
# ================================================================
def standardize_features(X):
    """Standardize features to have mean=0 and std=1"""
    print("\n" + "="*70)
    print("STEP 3: STANDARDIZING FEATURES")
    print("="*70)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    print(f"Features standardized successfully")
    print(f"  Mean: {X_scaled.mean().mean():.6f}")
    print(f"  Std: {X_scaled.std().mean():.6f}")
    
    return X_scaled, scaler


# ================================================================
# STEP 4: PRINCIPAL COMPONENT ANALYSIS (PCA)
# ================================================================
def perform_pca(X_scaled, all_features):
    """Perform PCA to understand variance structure"""
    print("\n" + "="*70)
    print("STEP 4: PRINCIPAL COMPONENT ANALYSIS (PCA)")
    print("="*70)
    
    # Fit PCA with all components
    pca_full = PCA(random_state=Config.RANDOM_SEED)
    pca_full.fit(X_scaled)
    
    # Calculate cumulative variance
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    
    print(f"\nVariance explained by each component:")
    for i, (var, cum_var) in enumerate(zip(pca_full.explained_variance_ratio_, cumulative_variance)):
        print(f"  PC{i+1}: {var:.4f} ({cum_var:.4f} cumulative)")
    
    # Fit PCA with specified number of factors
    pca = PCA(n_components=Config.N_FACTORS, random_state=Config.RANDOM_SEED)
    factor_scores_pca = pca.fit_transform(X_scaled)
    
    print(f"\nPCA Summary:")
    print(f"  Number of factors: {Config.N_FACTORS}")
    print(f"  Total variance explained: {cumulative_variance[Config.N_FACTORS-1]:.4f}")
    
    # Create loadings dataframe
    loadings_pca = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(Config.N_FACTORS)],
        index=all_features
    )
    
    return pca, loadings_pca, factor_scores_pca, pca_full


# ================================================================
# STEP 5: FACTOR ANALYSIS
# ================================================================
def perform_factor_analysis(X_scaled, all_features):
    """Perform Factor Analysis to identify latent factors"""
    print("\n" + "="*70)
    print("STEP 5: FACTOR ANALYSIS")
    print("="*70)
    
    # Fit Factor Analysis
    fa = FactorAnalysis(n_components=Config.N_FACTORS, random_state=Config.RANDOM_SEED)
    factor_scores_fa = fa.fit_transform(X_scaled)
    
    # Create loadings dataframe
    loadings_fa = pd.DataFrame(
        fa.components_.T,
        columns=[f'Factor{i+1}' for i in range(Config.N_FACTORS)],
        index=all_features
    )
    
    print(f"\nFactor Analysis completed")
    print(f"  Number of factors: {Config.N_FACTORS}")
    print(f"  Log-likelihood: {fa.score(X_scaled):.4f}")
    
    return fa, loadings_fa, factor_scores_fa


# ================================================================
# STEP 6: INTERPRET FACTORS
# ================================================================
def interpret_factors(loadings_fa, weather_features, traffic_features):
    """Interpret the meaning of each factor"""
    print("\n" + "="*70)
    print("STEP 6: INTERPRETING LATENT FACTORS")
    print("="*70)
    
    interpretations = {}
    
    for i in range(Config.N_FACTORS):
        factor_name = f'Factor{i+1}'
        print(f"\n{factor_name} Loadings:")
        
        # Sort by absolute loading
        sorted_loadings = loadings_fa[factor_name].abs().sort_values(ascending=False)
        
        for feature in sorted_loadings.index[:5]:  # Top 5 features
            loading = loadings_fa.loc[feature, factor_name]
            print(f"  {feature:25s}: {loading:7.4f}")
        
        # Interpret based on highest loadings
        top_features = sorted_loadings.index[:3].tolist()
        
        # Determine factor interpretation
        weather_count = sum([f in weather_features for f in top_features])
        traffic_count = sum([f in traffic_features for f in top_features])
        
        if weather_count > traffic_count:
            interpretation = "Weather Severity Factor"
            description = "Represents adverse weather conditions including rain, wind, and reduced visibility"
        elif 'vehicle_count' in top_features or 'avg_speed_kmh' in top_features:
            interpretation = "Traffic Flow Stress Factor"
            description = "Captures traffic congestion levels and flow dynamics"
        else:
            interpretation = "Accident Risk Factor"
            description = "Indicates conditions associated with higher accident probability"
        
        interpretations[factor_name] = {
            'name': interpretation,
            'description': description,
            'top_features': top_features
        }
    
    return interpretations


# ================================================================
# STEP 7: VISUALIZATIONS
# ================================================================
def create_visualizations(loadings_fa, pca_full, factor_scores_fa):
    """Create comprehensive visualizations"""
    print("\n" + "="*70)
    print("STEP 7: GENERATING VISUALIZATIONS")
    print("="*70)
    
    # 1. Scree Plot
    print("\n1. Scree plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    components = range(1, len(pca_full.explained_variance_ratio_) + 1)
    ax.plot(components, pca_full.explained_variance_ratio_, 'bo-', linewidth=2)
    ax.axvline(x=Config.N_FACTORS, color='r', linestyle='--', 
               label=f'Selected factors (n={Config.N_FACTORS})')
    ax.set_xlabel('Factor Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Explained Variance Ratio', fontsize=12, fontweight='bold')
    ax.set_title('Scree Plot - Variance Explained by Each Factor', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{Config.GOLD_DIR}/scree_plot.png', dpi=300)
    print("   Saved: scree_plot.png")
    plt.close()
    
    # 2. Factor Loadings Heatmap
    print("\n2. Factor loadings heatmap...")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(loadings_fa, annot=True, fmt='.3f', cmap='RdBu_r', 
                center=0, cbar_kws={'label': 'Loading'}, ax=ax)
    ax.set_title('Factor Loadings Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Factors', fontsize=12, fontweight='bold')
    ax.set_ylabel('Features', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{Config.GOLD_DIR}/factor_loadings_heatmap.png', dpi=300)
    print("   Saved: factor_loadings_heatmap.png")
    plt.close()
    
    # 3. Factor Scores Distribution
    print("\n3. Factor scores distribution...")
    fig, axes = plt.subplots(1, Config.N_FACTORS, figsize=(15, 5))
    for i in range(Config.N_FACTORS):
        axes[i].hist(factor_scores_fa[:, i], bins=50, edgecolor='black', alpha=0.7)
        axes[i].set_xlabel(f'Factor {i+1} Score', fontsize=10, fontweight='bold')
        axes[i].set_ylabel('Frequency', fontsize=10, fontweight='bold')
        axes[i].set_title(f'Factor {i+1} Distribution', fontsize=11, fontweight='bold')
        axes[i].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{Config.GOLD_DIR}/factor_scores_distribution.png', dpi=300)
    print("   Saved: factor_scores_distribution.png")
    plt.close()
    
    # 4. Biplot (Factor 1 vs Factor 2)
    if Config.N_FACTORS >= 2:
        print("\n4. Factor biplot...")
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot scores (sample points)
        sample_size = min(1000, len(factor_scores_fa))
        sample_idx = np.random.choice(len(factor_scores_fa), sample_size, replace=False)
        ax.scatter(factor_scores_fa[sample_idx, 0], 
                  factor_scores_fa[sample_idx, 1],
                  alpha=0.3, s=10, c='steelblue')
        
        # Plot loadings (feature vectors)
        scale = 3.0
        for i, feature in enumerate(loadings_fa.index):
            ax.arrow(0, 0, 
                    loadings_fa.iloc[i, 0] * scale,
                    loadings_fa.iloc[i, 1] * scale,
                    head_width=0.1, head_length=0.1, 
                    fc='red', ec='red', alpha=0.7, linewidth=2)
            ax.text(loadings_fa.iloc[i, 0] * scale * 1.1,
                   loadings_fa.iloc[i, 1] * scale * 1.1,
                   feature, fontsize=9, ha='center')
        
        ax.set_xlabel('Factor 1', fontsize=12, fontweight='bold')
        ax.set_ylabel('Factor 2', fontsize=12, fontweight='bold')
        ax.set_title('Factor Analysis Biplot', fontsize=14, fontweight='bold')
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{Config.GOLD_DIR}/factor_biplot.png', dpi=300)
        print("   Saved: factor_biplot.png")
        plt.close()
    
    print("\nAll visualizations created!")


# ================================================================
# STEP 8: SAVE RESULTS
# ================================================================
def save_results(loadings_fa, factor_scores_fa, interpretations, df):
    """Save factor analysis results"""
    print("\n" + "="*70)
    print("STEP 8: SAVING RESULTS")
    print("="*70)
    
    # Save factor loadings
    loadings_fa.to_csv(Config.FACTOR_LOADINGS_PATH)
    print(f"Saved: factor_loadings.csv")
    
    # Save factor scores
    factor_scores_df = pd.DataFrame(
        factor_scores_fa,
        columns=[f'Factor{i+1}' for i in range(Config.N_FACTORS)]
    )
    # Add simulation metadata
    factor_scores_df['simulation_id'] = df['simulation_id']
    factor_scores_df.to_csv(Config.FACTOR_SCORES_PATH, index=False)
    print(f"Saved: factor_scores.csv")


# ================================================================
# STEP 9: GENERATE INTERPRETATION REPORT
# ================================================================
def generate_interpretation_report(loadings_fa, interpretations, pca_full):
    """Generate comprehensive interpretation report"""
    print("\n" + "="*70)
    print("STEP 9: GENERATING INTERPRETATION REPORT")
    print("="*70)
    
    report = []
    report.append("="*70)
    report.append("FACTOR ANALYSIS INTERPRETATION REPORT")
    report.append("="*70)
    report.append("")
    report.append(f"Analysis Date: December 2024")
    report.append(f"Member: 6 - Factor Analysis & Final Integration")
    report.append(f"Number of Factors: {Config.N_FACTORS}")
    report.append("")
    
    # Overall variance explained
    cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)
    report.append(f"VARIANCE EXPLAINED:")
    report.append(f"  Total variance captured: {cumulative_var[Config.N_FACTORS-1]:.2%}")
    report.append("")
    
    # Factor interpretations
    report.append("LATENT FACTORS IDENTIFIED:")
    report.append("")
    
    for i in range(Config.N_FACTORS):
        factor_name = f'Factor{i+1}'
        interp = interpretations[factor_name]
        
        report.append(f"{factor_name}: {interp['name']}")
        report.append(f"  Description: {interp['description']}")
        report.append(f"  Key Features:")
        
        # Top 5 loadings
        sorted_loadings = loadings_fa[factor_name].abs().sort_values(ascending=False)
        for j, feature in enumerate(sorted_loadings.index[:5], 1):
            loading = loadings_fa.loc[feature, factor_name]
            report.append(f"    {j}. {feature:25s}: {loading:7.4f}")
        report.append("")
    
    # Insights
    report.append("KEY INSIGHTS:")
    report.append("")
    report.append("1. Weather Severity Impact:")
    report.append("   Weather conditions (rain, wind, visibility) cluster together,")
    report.append("   indicating a unified 'weather severity' dimension affecting traffic.")
    report.append("")
    report.append("2. Traffic Flow Dynamics:")
    report.append("   Vehicle count and speed form a distinct factor representing")
    report.append("   congestion and flow characteristics independent of weather.")
    report.append("")
    report.append("3. Accident Risk Factors:")
    report.append("   Accident occurrences correlate with specific combinations of")
    report.append("   weather and traffic conditions, forming a risk assessment dimension.")
    report.append("")
    
    # Recommendations
    report.append("RECOMMENDATIONS FOR URBAN TRAFFIC PLANNING:")
    report.append("")
    report.append("1. Integrated Weather Monitoring:")
    report.append("   Deploy unified weather severity indices for traffic management")
    report.append("   rather than monitoring individual weather variables separately.")
    report.append("")
    report.append("2. Flow-Based Interventions:")
    report.append("   Design traffic control strategies based on flow stress factors")
    report.append("   (vehicle density and speed patterns) as distinct from weather impacts.")
    report.append("")
    report.append("3. Risk-Targeted Safety Measures:")
    report.append("   Implement accident prevention protocols triggered by combined")
    report.append("   factor scores rather than single-variable thresholds.")
    report.append("")
    report.append("4. Predictive Modeling:")
    report.append("   Use factor scores as inputs for traffic prediction models")
    report.append("   to reduce dimensionality and improve computational efficiency.")
    report.append("")
    
    report.append("="*70)
    report.append("FILES GENERATED:")
    report.append("  - factor_loadings.csv")
    report.append("  - factor_scores.csv")
    report.append("  - factor_analysis_interpretation.txt")
    report.append("  - scree_plot.png")
    report.append("  - factor_loadings_heatmap.png")
    report.append("  - factor_scores_distribution.png")
    report.append("  - factor_biplot.png")
    report.append("="*70)
    
    # Save report
    report_text = "\n".join(report)
    with open(Config.INTERPRETATION_REPORT_PATH, 'w') as f:
        f.write(report_text)
    
    print(f"Saved: factor_analysis_interpretation.txt")
    print("\nReport preview:")
    print(report_text)


# ================================================================
# STEP 10: UPLOAD TO MINIO
# ================================================================
def upload_to_minio():
    """Upload results to MinIO"""
    print("\n" + "="*70)
    print("STEP 10: UPLOADING TO MINIO GOLD BUCKET")
    print("="*70)
    
    try:
        sys.path.append('Scripts')
        from minio_utils import get_client, upload_file
        
        client = get_client()
        files = [
            "factor_loadings.csv",
            "factor_scores.csv",
            "factor_analysis_interpretation.txt",
            "scree_plot.png",
            "factor_loadings_heatmap.png",
            "factor_scores_distribution.png",
            "factor_biplot.png"
        ]
        
        for filename in files:
            local_path = f'{Config.GOLD_DIR}/{filename}'
            if os.path.exists(local_path):
                upload_file(client, "gold", local_path, f"factor_analysis/{filename}")
        
        print("Files uploaded to MinIO")
    except Exception as e:
        print(f"MinIO upload warning: {e}")
        print("Files saved locally in:", Config.GOLD_DIR)


# ================================================================
# MAIN EXECUTION
# ================================================================
def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("FACTOR ANALYSIS - WEATHER & TRAFFIC PATTERNS")
    print("="*70)
    
    # Load data
    df = load_simulation_data()
    
    # Prepare features
    X, all_features, weather_features, traffic_features = prepare_features(df)
    
    # Standardize
    X_scaled, scaler = standardize_features(X)
    
    # PCA
    pca, loadings_pca, factor_scores_pca, pca_full = perform_pca(X_scaled, all_features)
    
    # Factor Analysis
    fa, loadings_fa, factor_scores_fa = perform_factor_analysis(X_scaled, all_features)
    
    # Interpret factors
    interpretations = interpret_factors(loadings_fa, weather_features, traffic_features)
    
    # Create visualizations
    create_visualizations(loadings_fa, pca_full, factor_scores_fa)
    
    # Save results
    save_results(loadings_fa, factor_scores_fa, interpretations, df)
    
    # Generate report
    generate_interpretation_report(loadings_fa, interpretations, pca_full)
    
    # Upload to MinIO
    upload_to_minio()
    
    print("\n" + "="*70)
    print("FACTOR ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nAll files saved in: {Config.GOLD_DIR}")
    print("Project pipeline complete - ready for final integration")


if __name__ == "__main__":
    main()