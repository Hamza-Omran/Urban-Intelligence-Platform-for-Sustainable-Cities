import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import os
st.set_page_config(
    page_title="Weather Impact on Traffic - Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(, unsafe_allow_html=True)
@st.cache_data
def load_cleaned_data():
    try:
        weather = pd.read_parquet('Data/weather_cleaned.parquet')
        traffic = pd.read_parquet('Data/traffic_cleaned.parquet')
        return weather, traffic
    except Exception as e:
        st.error(f"Error loading cleaned data: {e}")
        return None, None
@st.cache_data
def load_merged_data():
    try:
        merged = pd.read_parquet('output/merged_with_features.parquet')
        return merged
    except Exception as e:
        st.error(f"Error loading merged data: {e}")
        return None
@st.cache_data
def load_simulation_results():
    try:
        sim_results = pd.read_csv('Data/gold/simulation_results.csv')
        scenario_analysis = pd.read_csv('Data/gold/scenario_analysis.csv')
        return sim_results, scenario_analysis
    except Exception as e:
        st.error(f"Error loading simulation results: {e}")
        return None, None
@st.cache_data
def load_factor_analysis():
    try:
        factor_loadings = pd.read_csv('Data/gold/factor_loadings.csv', index_col=0)
        factor_scores = pd.read_csv('Data/gold/factor_scores.csv')
        return factor_loadings, factor_scores
    except Exception as e:
        st.error(f"Error loading factor analysis: {e}")
        return None, None
@st.cache_data
def load_network_analysis():
    try:
        network_metrics = pd.read_csv('Data/gold/network_metrics.csv')
        edge_weights = pd.read_csv('Data/gold/edge_weights.csv')
        return network_metrics, edge_weights
    except Exception as e:
        st.error(f"Error loading network analysis: {e}")
        return None, None
st.markdown(, unsafe_allow_html=True)
with st.sidebar:
    st.image("Assets/fcds_logo.jpg", use_container_width=True)
    st.markdown("###  Navigation")
    selected_page = st.radio(
        "",
        [" Overview", " Data Cleaning", " Data Merging", 
         " Monte Carlo Simulation", " Factor Analysis", " Network Analysis", " Recommendations"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("###  Team Members")
    team_members = [
        "Member 1: Infrastructure",
        "Member 2: Data Cleaning",
        "Member 3: HDFS Integration",
        "Member 4: Data Merging",
        "Member 5: Monte Carlo Simulation",
        "Member 6: Factor Analysis"
    ]
    for member in team_members:
        st.markdown(f"- {member}")
if selected_page == " Overview":
    st.header(" Project Overview")
    weather, traffic = load_cleaned_data()
    merged = load_merged_data()
    sim_results, scenario_analysis = load_simulation_results()
    factor_loadings, factor_scores = load_factor_analysis()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if sim_results is not None:
            avg_congestion = sim_results['congestion_probability'].mean() * 100
            st.metric(
                "Avg Congestion Probability",
                f"{avg_congestion:.2f}%",
                delta=f"+{(avg_congestion - 30):.1f}% vs baseline",
                delta_color="inverse"
            )
    with col2:
        if sim_results is not None:
            avg_accident = sim_results['accident_probability'].mean() * 100
            st.metric(
                "Avg Accident Probability",
                f"{avg_accident:.2f}%",
                delta=f"+{(avg_accident - 5):.1f}% vs baseline",
                delta_color="inverse"
            )
    with col3:
        if sim_results is not None:
            st.metric(
                "Total Simulations",
                f"{len(sim_results):,}",
                delta="Monte Carlo iterations"
            )
    with col4:
        st.metric(
            "Variance Explained",
            "42.48%",
            delta="by 3 factors"
        )
    st.markdown("---")
    st.subheader(" Data Pipeline Progress")
    pipeline_data = pd.DataFrame({
        'Stage': ['Bronze Layer', 'Data Cleaning', 'HDFS Transfer', 'Data Merging', 
                  'Monte Carlo', 'Factor Analysis'],
        'Records': [5000, 4794, 4794, 586, 10000, 10000],
        'Member': [1, 2, 3, 4, 5, 6],
        'Status': [' Complete', ' Complete', ' Complete', ' Complete', 
                   ' Complete', ' Complete']
    })
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=pipeline_data['Stage'],
        y=pipeline_data['Records'],
        text=pipeline_data['Status'],
        textposition='outside',
        marker=dict(
            color=pipeline_data['Records'],
            colorscale='Blues',
            showscale=False
        ),
        hovertemplate='<b>%{x}</b><br>Records: %{y:,}<extra></extra>'
    ))
    fig.update_layout(
        title="Data Pipeline Stages",
        xaxis_title="Stage",
        yaxis_title="Number of Records",
        height=400,
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    st.subheader(" Key Findings")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown()
    with col2:
        st.markdown()
elif selected_page == " Data Cleaning":
    st.header(" Data Cleaning Results")
    weather, traffic = load_cleaned_data()
    if weather is not None and traffic is not None:
        st.subheader(" Cleaning Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Weather Dataset")
            st.metric("Records After Cleaning", f"{len(weather):,}")
            st.metric("Null Values", "0")
            st.metric("Duplicates Removed", "50")
            st.metric("Outliers Handled", "Yes")
        with col2:
            st.markdown("### Traffic Dataset")
            st.metric("Records After Cleaning", f"{len(traffic):,}")
            st.metric("Null Values", "0")
            st.metric("Duplicates Removed", "40")
            st.metric("Outliers Handled", "Yes")
        st.markdown("---")
        tab1, tab2 = st.tabs(["Weather Data", "Traffic Data"])
        with tab1:
            st.subheader("Weather Data Overview")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**First 10 rows:**")
                st.dataframe(weather.head(10), use_container_width=True)
            with col2:
                st.write("**Statistical Summary:**")
                st.dataframe(weather.describe(), use_container_width=True)
            st.markdown("#### Variable Distributions")
            numeric_cols = ['temperature_c', 'humidity', 'rain_mm', 'wind_speed_kmh']
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=numeric_cols
            )
            for idx, col in enumerate(numeric_cols):
                row = idx // 2 + 1
                col_pos = idx % 2 + 1
                fig.add_trace(
                    go.Histogram(x=weather[col], name=col, showlegend=False),
                    row=row, col=col_pos
                )
            fig.update_layout(height=600, title_text="Weather Variables Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            st.subheader("Traffic Data Overview")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**First 10 rows:**")
                st.dataframe(traffic.head(10), use_container_width=True)
            with col2:
                st.write("**Statistical Summary:**")
                st.dataframe(traffic.describe(), use_container_width=True)
            st.markdown("#### Congestion Level Distribution")
            congestion_counts = traffic['congestion_level'].value_counts()
            fig = go.Figure(data=[
                go.Pie(
                    labels=congestion_counts.index,
                    values=congestion_counts.values,
                    hole=0.4,
                    marker=dict(colors=['#10b981', '#f59e0b', '#ef4444'])
                )
            ])
            fig.update_layout(title="Congestion Levels in Dataset")
            st.plotly_chart(fig, use_container_width=True)
elif selected_page == " Data Merging":
    st.header(" Data Merging & Feature Engineering")
    merged = load_merged_data()
    if merged is not None:
        st.subheader(" Merge Results")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Weather Records", "4,794")
        with col2:
            st.metric("Traffic Records", "4,813")
        with col3:
            st.metric("Merged Records", f"{len(merged):,}")
        with col4:
            merge_rate = (len(merged) / 4794) * 100
            st.metric("Merge Rate", f"{merge_rate:.2f}%")
        st.markdown("---")
        st.subheader(" Merged Dataset Preview")
        st.dataframe(merged.head(20), use_container_width=True)
        st.markdown("---")
        st.subheader(" Engineered Features")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Weather Severity Index")
            fig = px.histogram(
                merged,
                x='weather_severity_index',
                nbins=50,
                title="Weather Severity Distribution",
                labels={'weather_severity_index': 'Weather Severity Index'},
                color_discrete_sequence=['#3b82f6']
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown()
        with col2:
            st.markdown("#### Traffic Intensity Score")
            fig = px.histogram(
                merged,
                x='traffic_intensity_score',
                nbins=50,
                title="Traffic Intensity Distribution",
                labels={'traffic_intensity_score': 'Traffic Intensity Score'},
                color_discrete_sequence=['#ef4444']
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown()
        st.markdown("---")
        st.subheader(" Temporal Patterns")
        if 'hour' in merged.columns:
            hourly_avg = merged.groupby('hour').agg({
                'weather_severity_index': 'mean',
                'traffic_intensity_score': 'mean'
            }).reset_index()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hourly_avg['hour'],
                y=hourly_avg['weather_severity_index'],
                name='Weather Severity',
                line=dict(color='#3b82f6', width=3)
            ))
            fig.add_trace(go.Scatter(
                x=hourly_avg['hour'],
                y=hourly_avg['traffic_intensity_score'],
                name='Traffic Intensity',
                line=dict(color='#ef4444', width=3),
                yaxis='y2'
            ))
            fig.update_layout(
                title="Average Severity by Hour of Day",
                xaxis_title="Hour of Day",
                yaxis_title="Weather Severity",
                yaxis2=dict(
                    title="Traffic Intensity",
                    overlaying='y',
                    side='right'
                ),
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
elif selected_page == " Monte Carlo Simulation":
    st.header(" Monte Carlo Simulation Results")
    sim_results, scenario_analysis = load_simulation_results()
    if sim_results is not None and scenario_analysis is not None:
        st.subheader(" Simulation Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Iterations", f"{len(sim_results):,}")
        with col2:
            congestion_events = sim_results['congestion_occurred'].sum()
            st.metric("Congestion Events", f"{congestion_events:,}")
        with col3:
            accident_events = sim_results['accident_occurred'].sum()
            st.metric("Accident Events", f"{accident_events:,}")
        with col4:
            scenarios_active = sim_results[sim_results['scenario_count'] > 0].shape[0]
            st.metric("Scenarios Active", f"{scenarios_active:,}")
        st.markdown("---")
        st.subheader(" Probability Distributions")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(
                sim_results,
                x='congestion_probability',
                nbins=50,
                title="Congestion Probability Distribution",
                labels={'congestion_probability': 'Congestion Probability'},
                color_discrete_sequence=['#3b82f6']
            )
            fig.add_vline(
                x=sim_results['congestion_probability'].mean(),
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {sim_results['congestion_probability'].mean():.2%}"
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.histogram(
                sim_results,
                x='accident_probability',
                nbins=50,
                title="Accident Probability Distribution",
                labels={'accident_probability': 'Accident Probability'},
                color_discrete_sequence=['#ef4444']
            )
            fig.add_vline(
                x=sim_results['accident_probability'].mean(),
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {sim_results['accident_probability'].mean():.2%}"
            )
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
        st.subheader(" Weather Scenario Analysis")
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Congestion Rate by Scenario", "Accident Rate by Scenario"),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        fig.add_trace(
            go.Bar(
                y=scenario_analysis['Scenario'],
                x=scenario_analysis['Congestion_Rate'] * 100,
                orientation='h',
                name='Congestion Rate',
                marker=dict(color='#3b82f6')
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(
                y=scenario_analysis['Scenario'],
                x=scenario_analysis['Accident_Rate'] * 100,
                orientation='h',
                name='Accident Rate',
                marker=dict(color='#ef4444')
            ),
            row=1, col=2
        )
        fig.update_xaxes(title_text="Rate (%)", row=1, col=1)
        fig.update_xaxes(title_text="Rate (%)", row=1, col=2)
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("#### Detailed Scenario Statistics")
        scenario_display = scenario_analysis.copy()
        scenario_display['Avg_Congestion_Prob'] = scenario_display['Avg_Congestion_Prob'].apply(lambda x: f"{x:.2%}")
        scenario_display['Avg_Accident_Prob'] = scenario_display['Avg_Accident_Prob'].apply(lambda x: f"{x:.2%}")
        scenario_display['Congestion_Rate'] = scenario_display['Congestion_Rate'].apply(lambda x: f"{x:.2%}")
        scenario_display['Accident_Rate'] = scenario_display['Accident_Rate'].apply(lambda x: f"{x:.2%}")
        scenario_display['Percentage'] = scenario_display['Percentage'].apply(lambda x: f"{x:.2f}%")
        st.dataframe(scenario_display, use_container_width=True)
        st.markdown("---")
        st.subheader(" Hourly Risk Analysis by Area")
        hourly_area = sim_results.groupby(['area', 'hour'])['congestion_probability'].mean().reset_index()
        hourly_pivot = hourly_area.pivot(index='area', columns='hour', values='congestion_probability')
        fig = px.imshow(
            hourly_pivot,
            labels=dict(x="Hour of Day", y="Area", color="Avg Congestion Probability"),
            color_continuous_scale="YlOrRd",
            title="Congestion Risk Heatmap by Area and Hour"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
elif selected_page == " Factor Analysis":
    st.header(" Factor Analysis Results")
    factor_loadings, factor_scores = load_factor_analysis()
    if factor_loadings is not None and factor_scores is not None:
        st.subheader(" Factor Analysis Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Factors", "3")
        with col2:
            st.metric("Variance Explained", "42.48%")
        with col3:
            st.metric("Features Analyzed", "8")
        st.markdown("---")
        st.subheader(" Factor Interpretations")
        tab1, tab2, tab3 = st.tabs(["Factor 1", "Factor 2", "Factor 3"])
        with tab1:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown()
            with col2:
                factor1_data = factor_loadings['Factor1'].sort_values(key=abs, ascending=False)
                fig = go.Figure(go.Bar(
                    x=factor1_data.values,
                    y=factor1_data.index,
                    orientation='h',
                    marker=dict(
                        color=factor1_data.values,
                        colorscale='RdBu',
                        cmid=0
                    )
                ))
                fig.update_layout(
                    title="Factor 1 Loadings",
                    xaxis_title="Loading Value",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        with tab2:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown()
            with col2:
                factor2_data = factor_loadings['Factor2'].sort_values(key=abs, ascending=False)
                fig = go.Figure(go.Bar(
                    x=factor2_data.values,
                    y=factor2_data.index,
                    orientation='h',
                    marker=dict(
                        color=factor2_data.values,
                        colorscale='RdBu',
                        cmid=0
                    )
                ))
                fig.update_layout(
                    title="Factor 2 Loadings",
                    xaxis_title="Loading Value",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        with tab3:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown()
            with col2:
                factor3_data = factor_loadings['Factor3'].sort_values(key=abs, ascending=False)
                fig = go.Figure(go.Bar(
                    x=factor3_data.values,
                    y=factor3_data.index,
                    orientation='h',
                    marker=dict(
                        color=factor3_data.values,
                        colorscale='RdBu',
                        cmid=0
                    )
                ))
                fig.update_layout(
                    title="Factor 3 Loadings",
                    xaxis_title="Loading Value",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
        st.subheader(" Factor Loadings Heatmap")
        fig = px.imshow(
            factor_loadings.T,
            labels=dict(x="Features", y="Factors", color="Loading"),
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            title="Factor Loadings Matrix"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
        st.subheader(" Factor Score Distributions")
        col1, col2, col3 = st.columns(3)
        for idx, col in enumerate([col1, col2, col3]):
            with col:
                factor_col = f'Factor{idx+1}'
                fig = px.histogram(
                    factor_scores,
                    x=factor_col,
                    nbins=50,
                    title=f"{factor_col} Distribution",
                    color_discrete_sequence=['#8b5cf6']
                )
                st.plotly_chart(fig, use_container_width=True)
elif selected_page == " Network Analysis":
    st.header(" Urban Network Analysis")
    network_metrics, edge_weights = load_network_analysis()
    if network_metrics is not None and edge_weights is not None:
        st.subheader(" Network Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Network Nodes", len(network_metrics))
        with col2:
            st.metric("Network Edges", len(edge_weights))
        with col3:
            st.metric("Communities", network_metrics['community'].nunique())
        with col4:
            avg_centrality = network_metrics['degree_centrality'].mean()
            st.metric("Avg Centrality", f"{avg_centrality:.3f}")
        st.markdown("---")
        st.subheader(" Interactive Network Graph")
        fig = go.Figure()
        n_nodes = len(network_metrics)
        angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
        radius = 5
        node_positions = {}
        for i, row in network_metrics.iterrows():
            x = radius * np.cos(angles[i])
            y = radius * np.sin(angles[i])
            node_positions[row['area']] = (x, y)
        for _, edge in edge_weights.iterrows():
            x0, y0 = node_positions[edge['area1']]
            x1, y1 = node_positions[edge['area2']]
            fig.add_trace(go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode='lines',
                line=dict(width=edge['weight']*3, color='lightgray'),
                hoverinfo='skip',
                showlegend=False
            ))
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        for _, row in network_metrics.iterrows():
            x, y = node_positions[row['area']]
            node_x.append(x)
            node_y.append(y)
            hover_text = f"<b>{row['area']}</b><br>" + \
                        f"PageRank: {row['pagerank']:.3f}<br>" + \
                        f"Degree Centrality: {row['degree_centrality']:.3f}<br>" + \
                        f"Betweenness: {row['betweenness_centrality']:.3f}<br>" + \
                        f"Community: {row['community']}"
            node_text.append(hover_text)
            node_size.append(row['pagerank'] * 100 + 20)
            node_color.append(row['community'])
        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=[row['area'] for _, row in network_metrics.iterrows()],
            textposition='top center',
            hovertext=node_text,
            hoverinfo='text',
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Community"),
                line=dict(width=2, color='white')
            ),
            showlegend=False
        ))
        fig.update_layout(
            title="London Area Transportation Network",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info(" **Graph Interpretation**: Node size represents PageRank influence. Colors indicate detected communities (areas with similar traffic/weather patterns). Edge thickness shows correlation strength.")
        st.markdown("---")
        st.subheader(" Centrality Metrics Comparison")
        col1, col2 = st.columns(2)
        with col1:
            top_degree = network_metrics.nlargest(5, 'degree_centrality')
            fig = go.Figure(go.Bar(
                x=top_degree['degree_centrality'],
                y=top_degree['area'],
                orientation='h',
                marker=dict(color='#3b82f6'),
                text=top_degree['degree_centrality'].round(3),
                textposition='auto'
            ))
            fig.update_layout(
                title="Top 5 Areas by Degree Centrality",
                xaxis_title="Degree Centrality",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
            top_pagerank = network_metrics.nlargest(5, 'pagerank')
            fig = go.Figure(go.Bar(
                x=top_pagerank['pagerank'],
                y=top_pagerank['area'],
                orientation='h',
                marker=dict(color='#10b981'),
                text=top_pagerank['pagerank'].round(3),
                textposition='auto'
            ))
            fig.update_layout(
                title="Top 5 Areas by PageRank",
                xaxis_title="PageRank Score",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            top_betweenness = network_metrics.nlargest(5, 'betweenness_centrality')
            fig = go.Figure(go.Bar(
                x=top_betweenness['betweenness_centrality'],
                y=top_betweenness['area'],
                orientation='h',
                marker=dict(color='#f59e0b'),
                text=top_betweenness['betweenness_centrality'].round(3),
                textposition='auto'
            ))
            fig.update_layout(
                title="Top 5 Areas by Betweenness Centrality",
                xaxis_title="Betweenness Centrality",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
            top_closeness = network_metrics.nlargest(5, 'closeness_centrality')
            fig = go.Figure(go.Bar(
                x=top_closeness['closeness_centrality'],
                y=top_closeness['area'],
                orientation='h',
                marker=dict(color='#ef4444'),
                text=top_closeness['closeness_centrality'].round(3),
                textposition='auto'
            ))
            fig.update_layout(
                title="Top 5 Areas by Closeness Centrality",
                xaxis_title="Closeness Centrality",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
        st.subheader(" Community Analysis")
        community_stats = network_metrics.groupby('community').agg({
            'area': 'count',
            'avg_weather_severity': 'mean',
            'avg_traffic_intensity': 'mean',
            'total_accidents': 'sum',
            'pagerank': 'mean'
        }).reset_index()
        community_stats.columns = ['Community', 'Num Areas', 'Avg Weather Severity', 
                                   'Avg Traffic Intensity', 'Total Accidents', 'Avg PageRank']
        st.dataframe(community_stats.style.format({
            'Avg Weather Severity': '{:.2f}',
            'Avg Traffic Intensity': '{:.2f}',
            'Total Accidents': '{:.0f}',
            'Avg PageRank': '{:.3f}'
        }), use_container_width=True)
        st.markdown("---")
        st.subheader(" Detailed Network Metrics")
        display_metrics = network_metrics[[
            'area', 'degree', 'degree_centrality', 'betweenness_centrality',
            'closeness_centrality', 'pagerank', 'clustering_coefficient', 'community'
        ]].sort_values('pagerank', ascending=False)
        st.dataframe(display_metrics.style.format({
            'degree_centrality': '{:.3f}',
            'betweenness_centrality': '{:.3f}',
            'closeness_centrality': '{:.3f}',
            'pagerank': '{:.3f}',
            'clustering_coefficient': '{:.3f}'
        }), use_container_width=True)
        st.markdown("---")
        st.subheader(" Network Insights")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown()
        with col2:
            st.markdown()
elif selected_page == " Recommendations":
    st.header(" Strategic Recommendations for Urban Traffic Management")
    st.markdown()
    st.markdown("---")
    st.subheader(" Immediate Operational Actions")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown()
        st.markdown()
        st.markdown()
    with col2:
        st.markdown()
        st.markdown()
        st.markdown()
    st.markdown("---")
    st.subheader(" Policy Recommendations")
    st.markdown()
    st.markdown()
    st.markdown()
    st.markdown("---")
    st.subheader(" Factor-Based Monitoring System")
    st.markdown()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown()
    with col2:
        st.markdown()
    with col3:
        st.markdown()
    st.markdown("---")
    st.subheader(" Implementation Roadmap")
    roadmap_data = pd.DataFrame({
        'Phase': ['Phase 1 (Months 1-3)', 'Phase 2 (Months 4-6)', 'Phase 3 (Months 7-12)', 'Phase 4 (Year 2+)'],
        'Actions': [
            'Deploy additional sensors • Implement dynamic speed limits • Launch public communication system',
            'Install adaptive traffic signals • Enhance drainage systems • Begin emergency protocol training',
            'Complete windbreak installations • Launch modal shift incentives • Full factor monitoring system',
            'Evaluate outcomes • Scale successful programs • Long-term infrastructure improvements'
        ],
        'Budget': ['£2-3M', '£5-7M', '£8-12M', '£15-20M'],
        'Priority': [' High', ' Medium', ' Medium', ' Long-term']
    })
    st.dataframe(roadmap_data, use_container_width=True)
    st.markdown("---")
    st.subheader(" Expected Outcomes")
    col1, col2 = st.columns(2)
    with col1:
        outcomes_data = pd.DataFrame({
            'Metric': ['Congestion Probability', 'Accident Risk', 'Average Speed', 'Public Satisfaction'],
            'Current': ['66.16%', '28.75%', '45 km/h', '65%'],
            'Target (Year 1)': ['55-60%', '22-25%', '50-55 km/h', '75%'],
            'Target (Year 2)': ['45-50%', '18-20%', '55-60 km/h', '85%']
        })
        st.dataframe(outcomes_data, use_container_width=True)
    with col2:
        st.markdown()
    st.markdown("---")
    st.subheader(" Key Takeaways")
    st.success()
    st.info()
st.markdown("---")
st.markdown(, unsafe_allow_html=True)
