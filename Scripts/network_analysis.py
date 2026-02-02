import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms import community as nx_community
from scipy.stats import pearsonr
import warnings
import os
import sys
warnings.filterwarnings('ignore')
class Config:
    MERGED_DATA_PATH = r"output/merged_with_features.parquet"
    SIMULATION_RESULTS_PATH = r"Data/gold/simulation_results.csv"
    NETWORK_METRICS_PATH = r"Data/gold/network_metrics.csv"
    COMMUNITY_LABELS_PATH = r"Data/gold/community_labels.csv"
    EDGE_WEIGHTS_PATH = r"Data/gold/edge_weights.csv"
    NETWORK_STATS_PATH = r"Data/gold/network_statistics.txt"
    CORRELATION_THRESHOLD = 0.15
    MIN_SHARED_OBSERVATIONS = 3
def load_data():
    print("\n" + "="*70)
    print("NETWORK ANALYSIS - DATA LOADING")
    print("="*70)
    try:
        merged = pd.read_parquet(Config.MERGED_DATA_PATH)
        print(f" Loaded merged dataset: {len(merged):,} records")
        print(f"  Unique areas: {merged['area'].nunique()}")
        sim_results = pd.read_csv(Config.SIMULATION_RESULTS_PATH)
        print(f" Loaded simulation results: {len(sim_results):,} simulations")
        return merged, sim_results
    except Exception as e:
        print(f" Error loading data: {e}")
        sys.exit(1)
def aggregate_by_area(merged_df):
    print("\n" + "="*70)
    print("AGGREGATING DATA BY AREA")
    print("="*70)
    area_agg = merged_df.groupby('area').agg({
        'weather_severity_index': 'mean',
        'traffic_intensity_score': 'mean',
        'temperature_c': 'mean',
        'humidity': 'mean',
        'rain_mm': 'mean',
        'wind_speed_kmh': 'mean',
        'visibility_m': 'mean',
        'vehicle_count': 'mean',
        'avg_speed_kmh': 'mean',
        'accident_count': 'sum'
    }).reset_index()
    area_agg.columns = ['area', 'avg_weather_severity', 'avg_traffic_intensity',
                        'avg_temperature', 'avg_humidity', 'avg_rain', 'avg_wind',
                        'avg_visibility', 'avg_vehicles', 'avg_speed', 'total_accidents']
    print(f" Aggregated {len(area_agg)} areas")
    print(f"\nAreas: {area_agg['area'].tolist()}")
    return area_agg
def build_urban_network(merged_df, area_agg):
    print("\n" + "="*70)
    print("BUILDING URBAN NETWORK GRAPH")
    print("="*70)
    G = nx.Graph()
    for _, row in area_agg.iterrows():
        G.add_node(row['area'], 
                   weather_severity=row['avg_weather_severity'],
                   traffic_intensity=row['avg_traffic_intensity'],
                   avg_speed=row['avg_speed'],
                   total_accidents=row['total_accidents'])
    print(f" Added {G.number_of_nodes()} nodes (areas)")
    areas = area_agg['area'].tolist()
    all_correlations = []
    for i, area1 in enumerate(areas):
        area_correlations = []
        for area2 in areas:
            if area1 == area2:
                continue
            area1_data = merged_df[merged_df['area'] == area1]
            area2_data = merged_df[merged_df['area'] == area2]
            common_times = set(area1_data['date_time']).intersection(
                set(area2_data['date_time'])
            )
            if len(common_times) >= Config.MIN_SHARED_OBSERVATIONS:
                area1_aligned = area1_data[area1_data['date_time'].isin(common_times)].sort_values('date_time').reset_index(drop=True)
                area2_aligned = area2_data[area2_data['date_time'].isin(common_times)].sort_values('date_time').reset_index(drop=True)
                if len(area1_aligned) == len(area2_aligned) and len(area1_aligned) >= Config.MIN_SHARED_OBSERVATIONS:
                    try:
                        traffic_corr, _ = pearsonr(
                            area1_aligned['traffic_intensity_score'].values,
                            area2_aligned['traffic_intensity_score'].values
                        )
                        weather_corr, _ = pearsonr(
                            area1_aligned['weather_severity_index'].values,
                            area2_aligned['weather_severity_index'].values
                        )
                        combined_corr = (0.6 * traffic_corr + 0.4 * weather_corr)
                        area_correlations.append({
                            'area1': area1,
                            'area2': area2,
                            'weight': combined_corr,
                            'traffic_correlation': traffic_corr,
                            'weather_correlation': weather_corr,
                            'observations': len(area1_aligned)
                        })
                    except:
                        pass
        area_correlations.sort(key=lambda x: x['weight'], reverse=True)
        top_k = min(2, len(area_correlations))
        all_correlations.extend(area_correlations[:top_k])
    edge_set = set()
    edge_data = []
    for corr in all_correlations:
        edge_key = tuple(sorted([corr['area1'], corr['area2']]))
        if edge_key not in edge_set:
            edge_set.add(edge_key)
            G.add_edge(corr['area1'], corr['area2'], 
                       weight=abs(corr['weight']),
                       traffic_correlation=corr['traffic_correlation'],
                       weather_correlation=corr['weather_correlation'])
            edge_data.append({
                'area1': edge_key[0],
                'area2': edge_key[1],
                'weight': round(abs(corr['weight']), 3),
                'traffic_correlation': round(corr['traffic_correlation'], 3),
                'weather_correlation': round(corr['weather_correlation'], 3),
                'observations': corr['observations']
            })
    print(f" Added {G.number_of_edges()} edges (top-2 most correlated per node)")
    print(f"  Edge creation strategy: Each area connected to its 2 most similar areas")
    edge_df = pd.DataFrame(edge_data)
    return G, edge_df
def calculate_centrality_metrics(G):
    print("\n" + "="*70)
    print("CALCULATING CENTRALITY METRICS")
    print("="*70)
    metrics = {}
    print("  → Degree centrality...")
    metrics['degree_centrality'] = nx.degree_centrality(G)
    print("  → Betweenness centrality...")
    metrics['betweenness_centrality'] = nx.betweenness_centrality(G, weight='weight')
    print("  → Closeness centrality...")
    metrics['closeness_centrality'] = nx.closeness_centrality(G, distance='weight')
    print("  → PageRank...")
    metrics['pagerank'] = nx.pagerank(G, weight='weight')
    print("  → Clustering coefficient...")
    metrics['clustering_coefficient'] = nx.clustering(G, weight='weight')
    metrics['degree'] = dict(G.degree())
    print(" All centrality metrics calculated")
    return metrics
def detect_communities(G):
    print("\n" + "="*70)
    print("DETECTING COMMUNITIES")
    print("="*70)
    if G.number_of_edges() == 0:
        print(" Warning: Graph has no edges. Assigning each node to its own community.")
        partition = {node: i for i, node in enumerate(G.nodes())}
        modularity = 0.0
        n_communities = G.number_of_nodes()
        print(f" Created {n_communities} singleton communities")
        return partition, modularity
    communities = nx_community.greedy_modularity_communities(G, weight='weight')
    partition = {}
    for i, community_set in enumerate(communities):
        for node in community_set:
            partition[node] = i
    n_communities = len(communities)
    print(f" Detected {n_communities} communities")
    modularity = nx_community.modularity(G, communities, weight='weight')
    print(f"  Modularity score: {modularity:.4f}")
    for community_id, community_set in enumerate(communities):
        members = list(community_set)
        print(f"\n  Community {community_id}: {len(members)} areas")
        print(f"    {', '.join(members)}")
    return partition, modularity
def calculate_network_statistics(G, modularity):
    print("\n" + "="*70)
    print("CALCULATING NETWORK STATISTICS")
    print("="*70)
    stats = {}
    stats['num_nodes'] = G.number_of_nodes()
    stats['num_edges'] = G.number_of_edges()
    stats['density'] = nx.density(G)
    stats['num_components'] = nx.number_connected_components(G)
    stats['is_connected'] = nx.is_connected(G)
    if stats['is_connected']:
        stats['diameter'] = nx.diameter(G)
        stats['radius'] = nx.radius(G)
        stats['average_shortest_path'] = nx.average_shortest_path_length(G, weight='weight')
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_cc)
        stats['diameter'] = nx.diameter(subgraph)
        stats['radius'] = nx.radius(subgraph)
        stats['average_shortest_path'] = nx.average_shortest_path_length(subgraph, weight='weight')
    stats['average_clustering'] = nx.average_clustering(G, weight='weight')
    stats['transitivity'] = nx.transitivity(G)
    stats['modularity'] = modularity
    degrees = [deg for node, deg in G.degree()]
    stats['avg_degree'] = np.mean(degrees)
    stats['max_degree'] = max(degrees)
    stats['min_degree'] = min(degrees)
    print("\nNetwork Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    return stats
def create_metrics_dataframe(G, centrality_metrics, partition, area_agg):
    print("\n" + "="*70)
    print("COMPILING RESULTS")
    print("="*70)
    nodes = list(G.nodes())
    metrics_df = pd.DataFrame({
        'area': nodes,
        'degree': [centrality_metrics['degree'][node] for node in nodes],
        'degree_centrality': [centrality_metrics['degree_centrality'][node] for node in nodes],
        'betweenness_centrality': [centrality_metrics['betweenness_centrality'][node] for node in nodes],
        'closeness_centrality': [centrality_metrics['closeness_centrality'][node] for node in nodes],
        'pagerank': [centrality_metrics['pagerank'][node] for node in nodes],
        'clustering_coefficient': [centrality_metrics['clustering_coefficient'][node] for node in nodes],
        'community': [partition[node] for node in nodes]
    })
    metrics_df = metrics_df.merge(area_agg, on='area', how='left')
    metrics_df = metrics_df.sort_values('pagerank', ascending=False)
    print(f" Compiled metrics for {len(metrics_df)} areas")
    print(f"\nTop 5 areas by PageRank:")
    print(metrics_df[['area', 'pagerank', 'degree', 'community']].head(5).to_string(index=False))
    return metrics_df
def save_results(metrics_df, edge_df, stats):
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    metrics_df.to_csv(Config.NETWORK_METRICS_PATH, index=False)
    print(f" Saved: network_metrics.csv")
    edge_df.to_csv(Config.EDGE_WEIGHTS_PATH, index=False)
    print(f" Saved: edge_weights.csv")
    community_df = metrics_df[['area', 'community']].copy()
    community_df.to_csv(Config.COMMUNITY_LABELS_PATH, index=False)
    print(f" Saved: community_labels.csv")
    with open(Config.NETWORK_STATS_PATH, 'w') as f:
        f.write("="*70 + "\n")
        f.write("URBAN NETWORK STATISTICS\n")
        f.write("="*70 + "\n\n")
        for key, value in stats.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")
    print(f" Saved: network_statistics.txt")
def main():
    print("\n" + "="*70)
    print("URBAN INTELLIGENCE PLATFORM - NETWORK ANALYSIS")
    print("="*70)
    os.makedirs("Data/gold", exist_ok=True)
    merged_df, sim_results_df = load_data()
    area_agg = aggregate_by_area(merged_df)
    G, edge_df = build_urban_network(merged_df, area_agg)
    centrality_metrics = calculate_centrality_metrics(G)
    partition, modularity = detect_communities(G)
    stats = calculate_network_statistics(G, modularity)
    metrics_df = create_metrics_dataframe(G, centrality_metrics, partition, area_agg)
    save_results(metrics_df, edge_df, stats)
    print("\n" + "="*70)
    print(" NETWORK ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults saved in: Data/gold/")
    print("  - network_metrics.csv")
    print("  - edge_weights.csv")
    print("  - community_labels.csv")
    print("  - network_statistics.txt")
    print("\nReady for dashboard integration!")
if __name__ == "__main__":
    main()
