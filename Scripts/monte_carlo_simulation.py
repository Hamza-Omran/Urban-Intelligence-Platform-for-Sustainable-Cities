import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
import sys
warnings.filterwarnings('ignore')
np.random.seed(42)
class Config:
    MERGED_DATA_PATH = r"output/merged_with_features.parquet"
    SIMULATION_RESULTS_PATH = r"Data/gold/simulation_results.csv"
    SCENARIO_ANALYSIS_PATH = r"Data/gold/scenario_analysis.csv"
    GOLD_DIR = r"Data/gold"
    PLOTS_DIR = r"output/plots"
    N_SIMULATIONS = 10000
    RANDOM_SEED = 42
    BASE_CONGESTION_PROB = 0.3
    BASE_ACCIDENT_PROB = 0.05
    HEAVY_RAIN_THRESHOLD = 20
    TEMP_LOW_THRESHOLD = 0
    TEMP_HIGH_THRESHOLD = 30
    HIGH_HUMIDITY_THRESHOLD = 80
    LOW_VISIBILITY_THRESHOLD = 100
    STRONG_WIND_THRESHOLD = 50
def setup_directories():
    os.makedirs(Config.GOLD_DIR, exist_ok=True)
    os.makedirs(Config.PLOTS_DIR, exist_ok=True)
    print(" Output directories created")
def load_merged_data():
    print("\n" + "="*70)
    print("STEP 1: LOADING MERGED DATASET FROM MEMBER 4")
    print("="*70)
    try:
        df = pd.read_parquet(Config.MERGED_DATA_PATH)
        print(f" Dataset loaded successfully!")
        print(f"  Total records: {len(df):,}")
        print(f"  Columns: {len(df.columns)}")
        print(f"\n  Dataset overview:")
        print(f"    Date range: {df['date_time'].min()} to {df['date_time'].max()}")
        print(f"    Cities: {df['city'].unique().tolist()}")
        print(f"    Areas: {df['area'].nunique()} unique areas")
        return df
    except FileNotFoundError:
        print(f" ERROR: Merged dataset not found at {Config.MERGED_DATA_PATH}")
        sys.exit(1)
    except Exception as e:
        print(f" ERROR loading dataset: {e}")
        sys.exit(1)
def define_scenarios():
    print("\n" + "="*70)
    print("STEP 2: DEFINING SIMULATION SCENARIOS")
    print("="*70)
    scenarios = {
        "Heavy Rain": {
            "condition": lambda row: row["rain_mm"] > Config.HEAVY_RAIN_THRESHOLD,
            "description": f"Rainfall exceeding {Config.HEAVY_RAIN_THRESHOLD}mm",
            "risk_multiplier": 1.8
        },
        "Temperature Extremes": {
            "condition": lambda row: (row["temperature_c"] < Config.TEMP_LOW_THRESHOLD) | 
                                    (row["temperature_c"] > Config.TEMP_HIGH_THRESHOLD),
            "description": f"Temperature below {Config.TEMP_LOW_THRESHOLD}°C or above {Config.TEMP_HIGH_THRESHOLD}°C",
            "risk_multiplier": 1.6
        },
        "High Humidity": {
            "condition": lambda row: row["humidity"] > Config.HIGH_HUMIDITY_THRESHOLD,
            "description": f"Humidity exceeding {Config.HIGH_HUMIDITY_THRESHOLD}%",
            "risk_multiplier": 1.4
        },
        "Low Visibility": {
            "condition": lambda row: row["visibility_m"] < Config.LOW_VISIBILITY_THRESHOLD,
            "description": f"Visibility below {Config.LOW_VISIBILITY_THRESHOLD} meters",
            "risk_multiplier": 2.0
        },
        "Strong Winds": {
            "condition": lambda row: row["wind_speed_kmh"] > Config.STRONG_WIND_THRESHOLD,
            "description": f"Wind speed exceeding {Config.STRONG_WIND_THRESHOLD} km/h",
            "risk_multiplier": 1.5
        }
    }
    print("\nDefined scenarios:")
    for i, (name, info) in enumerate(scenarios.items(), 1):
        print(f"  {i}. {name}")
        print(f"     Description: {info['description']}")
        print(f"     Risk multiplier: {info['risk_multiplier']}x")
    return scenarios
def analyze_scenarios_in_data(df, scenarios):
    print("\n" + "="*70)
    print("ANALYZING SCENARIOS IN CURRENT DATASET")
    print("="*70)
    for scenario_name, scenario_info in scenarios.items():
        matching_rows = df[df.apply(scenario_info["condition"], axis=1)]
        percentage = (len(matching_rows) / len(df)) * 100
        print(f"\n{scenario_name}:")
        print(f"  Occurrences: {len(matching_rows):,} ({percentage:.2f}%)")
        if len(matching_rows) > 0:
            print(f"  Congestion levels:")
            for level, count in matching_rows["congestion_level"].value_counts().items():
                print(f"    - {level}: {count} ({count/len(matching_rows)*100:.1f}%)")
def calculate_congestion_probability(weather_severity, traffic_intensity, 
                                    base_prob=Config.BASE_CONGESTION_PROB):
    weather_norm = min(weather_severity / 50.0, 1.0)
    traffic_norm = min(traffic_intensity / 100.0, 1.0)
    combined_factor = (0.6 * weather_norm) + (0.4 * traffic_norm)
    probability = base_prob + (1 - base_prob) * (combined_factor ** 1.5)
    return min(probability, 0.95)
def calculate_accident_probability(weather_severity, avg_speed, congestion_level, 
                                   road_condition, base_prob=Config.BASE_ACCIDENT_PROB):
    weather_factor = 1.0 + min(weather_severity / 30.0, 1.5)
    speed_factor = 1.0 + (avg_speed / 100.0) ** 2
    congestion_factors = {"Low": 1.0, "Medium": 1.3, "High": 1.6}
    congestion_factor = congestion_factors.get(congestion_level, 1.0)
    road_factors = {"Dry": 1.0, "Wet": 1.5, "Snowy": 2.0, "Damaged": 2.5}
    road_factor = road_factors.get(road_condition, 1.0)
    probability = base_prob * weather_factor * speed_factor * congestion_factor * road_factor
    return min(probability, 0.80)
def test_models():
    print("\n" + "="*70)
    print("STEP 3: TESTING TRAFFIC BEHAVIOR MODELS")
    print("="*70)
    test_cases = [
        {"name": "Normal Conditions", "weather_sev": 5, "traffic_int": 20, 
         "speed": 50, "congestion": "Low", "road": "Dry"},
        {"name": "Heavy Rain", "weather_sev": 35, "traffic_int": 60, 
         "speed": 30, "congestion": "High", "road": "Wet"},
        {"name": "Extreme Weather", "weather_sev": 50, "traffic_int": 80, 
         "speed": 20, "congestion": "High", "road": "Snowy"}
    ]
    for test in test_cases:
        jam_prob = calculate_congestion_probability(test["weather_sev"], test["traffic_int"])
        acc_prob = calculate_accident_probability(test["weather_sev"], test["speed"],
                                                  test["congestion"], test["road"])
        print(f"\n{test['name']}:")
        print(f"  Congestion Probability: {jam_prob:.2%}")
        print(f"  Accident Probability: {acc_prob:.2%}")
def run_monte_carlo_simulation(df, scenarios, n_simulations=Config.N_SIMULATIONS):
    print("\n" + "="*70)
    print(f"STEP 4: RUNNING MONTE CARLO SIMULATION ({n_simulations:,} iterations)")
    print("="*70)
    results = []
    start_time = datetime.now()
    for i in range(n_simulations):
        if (i + 1) % 2000 == 0:
            print(f"  Progress: {i+1:,}/{n_simulations:,} ({(i+1)/n_simulations*100:.1f}%)")
        sample = df.sample(n=1, replace=True).iloc[0]
        weather_severity = sample["weather_severity_index"] * np.random.uniform(0.8, 1.2)
        traffic_intensity = sample["traffic_intensity_score"] * np.random.uniform(0.8, 1.2)
        avg_speed = max(5, sample["avg_speed_kmh"] * np.random.uniform(0.9, 1.1))
        congestion_prob = calculate_congestion_probability(weather_severity, traffic_intensity)
        accident_prob = calculate_accident_probability(weather_severity, avg_speed,
                                                       sample["congestion_level"],
                                                       sample["road_condition"])
        congestion_occurred = int(np.random.random() < congestion_prob)
        accident_occurred = int(np.random.random() < accident_prob)
        active_scenarios = []
        for scenario_name, scenario_info in scenarios.items():
            if scenario_info["condition"](sample):
                active_scenarios.append(scenario_name)
        results.append({
            "simulation_id": i + 1,
            "timestamp": sample["date_time"],
            "city": sample["city"],
            "area": sample["area"],
            "hour": int(sample["hour"]),
            "day_of_week": int(sample["day_of_week"]),
            "is_weekend": int(sample["is_weekend"]),
            "temperature_c": round(sample["temperature_c"], 2),
            "humidity": int(sample["humidity"]),
            "rain_mm": round(sample["rain_mm"], 2),
            "wind_speed_kmh": round(sample["wind_speed_kmh"], 2),
            "visibility_m": int(sample["visibility_m"]),
            "weather_severity_index": round(weather_severity, 2),
            "vehicle_count": int(sample["vehicle_count"]),
            "avg_speed_kmh": round(avg_speed, 2),
            "accident_count": int(sample["accident_count"]),
            "congestion_level": sample["congestion_level"],
            "road_condition": sample["road_condition"],
            "traffic_intensity_score": round(traffic_intensity, 2),
            "congestion_probability": round(congestion_prob, 4),
            "accident_probability": round(accident_prob, 4),
            "congestion_occurred": congestion_occurred,
            "accident_occurred": accident_occurred,
            "active_scenarios": "|".join(active_scenarios) if active_scenarios else "Normal",
            "scenario_count": len(active_scenarios)
        })
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n Simulation completed in {elapsed:.1f} seconds")
    return pd.DataFrame(results)
def summarize_simulation_results(simulation_results):
    print("\n" + "="*70)
    print("SIMULATION RESULTS SUMMARY")
    print("="*70)
    print(f"\nOverall Statistics:")
    print(f"  Total simulations: {len(simulation_results):,}")
    print(f"  Avg congestion probability: {simulation_results['congestion_probability'].mean():.2%}")
    print(f"  Avg accident probability: {simulation_results['accident_probability'].mean():.2%}")
    print(f"  Congestion occurred: {simulation_results['congestion_occurred'].sum():,} times")
    print(f"  Accidents occurred: {simulation_results['accident_occurred'].sum():,} times")
def analyze_by_scenario(simulation_results, scenarios):
    print("\n" + "="*70)
    print("STEP 5: ANALYZING RESULTS BY WEATHER SCENARIO")
    print("="*70)
    scenario_analysis = []
    for scenario_name in scenarios.keys():
        scenario_results = simulation_results[
            simulation_results["active_scenarios"].str.contains(scenario_name, na=False)
        ]
        if len(scenario_results) > 0:
            analysis = {
                "Scenario": scenario_name,
                "Occurrences": len(scenario_results),
                "Percentage": (len(scenario_results) / len(simulation_results)) * 100,
                "Avg_Congestion_Prob": scenario_results["congestion_probability"].mean(),
                "Avg_Accident_Prob": scenario_results["accident_probability"].mean(),
                "Congestion_Rate": scenario_results["congestion_occurred"].mean(),
                "Accident_Rate": scenario_results["accident_occurred"].mean()
            }
            scenario_analysis.append(analysis)
    scenario_df = pd.DataFrame(scenario_analysis)
    scenario_df = scenario_df.sort_values("Avg_Congestion_Prob", ascending=False)
    print("\nScenario Analysis:")
    print(scenario_df.to_string(index=False))
    return scenario_df
def create_visualizations(simulation_results, scenario_df):
    print("\n" + "="*70)
    print("STEP 6: GENERATING VISUALIZATIONS")
    print("="*70)
    sns.set_style("whitegrid")
    print("\n1. Congestion probability distribution...")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(simulation_results['congestion_probability'], bins=50, 
            edgecolor='black', alpha=0.7, color='steelblue')
    mean_val = simulation_results['congestion_probability'].mean()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_val:.2%}')
    ax.set_xlabel('Congestion Probability', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Congestion Probabilities', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{Config.GOLD_DIR}/congestion_probability_distribution.png', dpi=300)
    print("    Saved")
    plt.close()
    print("\n2. Accident probability distribution...")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(simulation_results['accident_probability'], bins=50,
            edgecolor='black', alpha=0.7, color='coral')
    mean_val = simulation_results['accident_probability'].mean()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_val:.2%}')
    ax.set_xlabel('Accident Probability', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Accident Probabilities', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{Config.GOLD_DIR}/accident_probability_distribution.png', dpi=300)
    print("    Saved")
    plt.close()
    print("\n3. Scenario comparison chart...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    scenario_sorted = scenario_df.sort_values('Congestion_Rate', ascending=True)
    axes[0].barh(scenario_sorted['Scenario'], scenario_sorted['Congestion_Rate'],
                 color='steelblue', edgecolor='black')
    axes[0].set_xlabel('Congestion Rate', fontsize=12, fontweight='bold')
    axes[0].set_title('Congestion Rate by Scenario', fontsize=13, fontweight='bold')
    axes[1].barh(scenario_sorted['Scenario'], scenario_sorted['Accident_Rate'],
                 color='coral', edgecolor='black')
    axes[1].set_xlabel('Accident Rate', fontsize=12, fontweight='bold')
    axes[1].set_title('Accident Rate by Scenario', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{Config.GOLD_DIR}/scenario_comparison.png', dpi=300)
    print("    Saved")
    plt.close()
    print("\n4. Risk heatmap by area and hour...")
    heatmap_data = simulation_results.groupby(['area', 'hour'])['congestion_probability'].mean().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='area', columns='hour', values='congestion_probability')
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(heatmap_pivot, annot=False, cmap='YlOrRd',
                cbar_kws={'label': 'Avg Congestion Probability'}, ax=ax)
    ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
    ax.set_ylabel('Area', fontsize=12, fontweight='bold')
    ax.set_title('Congestion Risk Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{Config.GOLD_DIR}/risk_heatmap_area_hour.png', dpi=300)
    print("    Saved")
    plt.close()
    print("\n All visualizations created!")
def save_results(simulation_results, scenario_df):
    print("\n" + "="*70)
    print("STEP 7: SAVING RESULTS")
    print("="*70)
    simulation_results.to_csv(Config.SIMULATION_RESULTS_PATH, index=False)
    print(f" Saved: simulation_results.csv ({len(simulation_results):,} rows)")
    scenario_df.to_csv(Config.SCENARIO_ANALYSIS_PATH, index=False)
    print(f" Saved: scenario_analysis.csv")
def upload_to_minio():
    print("\n" + "="*70)
    print("STEP 8: UPLOADING TO MINIO GOLD BUCKET")
    print("="*70)
    try:
        sys.path.append('Scripts')
        from minio_utils import get_client, upload_file
        client = get_client()
        files = [
            "simulation_results.csv",
            "scenario_analysis.csv",
            "congestion_probability_distribution.png",
            "accident_probability_distribution.png",
            "scenario_comparison.png",
            "risk_heatmap_area_hour.png"
        ]
        for filename in files:
            local_path = f'{Config.GOLD_DIR}/{filename}'
            if os.path.exists(local_path):
                upload_file(client, "gold", local_path, f"simulation/{filename}")
        print(" Files uploaded to MinIO")
    except Exception as e:
        print(f" MinIO upload warning: {e}")
        print("  Files saved locally in:", Config.GOLD_DIR)
def main():
    print("\n" + "="*70)
    print("MONTE CARLO SIMULATION - TRAFFIC RISK PREDICTION")
    print("="*70)
    setup_directories()
    df = load_merged_data()
    scenarios = define_scenarios()
    analyze_scenarios_in_data(df, scenarios)
    test_models()
    simulation_results = run_monte_carlo_simulation(df, scenarios)
    summarize_simulation_results(simulation_results)
    scenario_df = analyze_by_scenario(simulation_results, scenarios)
    create_visualizations(simulation_results, scenario_df)
    save_results(simulation_results, scenario_df)
    upload_to_minio()
    print("\n" + "="*70)
    print(" SIMULATION COMPLETE!")
    print("="*70)
    print(f"\nFiles saved in: {Config.GOLD_DIR}")
    print("Ready for Member 6 - Factor Analysis")
if __name__ == "__main__":
    main()
