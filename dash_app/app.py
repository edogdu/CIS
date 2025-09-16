from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import requests

# 1. Initialize the Dash application
app = Dash(__name__)

# --- Data Fetching Functions ---
# This part handles fetching data from your backend services.
# Note: You should replace 'api' and 'neo4j' with the correct service names as defined in your docker-compose.yml
# This function fetches anomaly alerts from your API
def fetch_alerts():
    response = requests.get("http://api:8090/alerts")
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    return pd.DataFrame()

# This function fetches data from Neo4j to build the graph visualization
def fetch_graph_data():
    # You would use a Neo4j driver here to execute a Cypher query
    # The query would retrieve nodes and relationships for visualization
    # Example:
    # driver = GraphDatabase.driver("bolt://neo4j:7687", auth=("neo4j", "password"))
    # with driver.session() as session:
    #     result = session.run("MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50")
    #     # Process the result into a format for your visualization library
    # return processed_data
    pass


# 2. Define the application layout
app.layout = html.Div(children=[
    html.H1(children="Explainable Anomaly Detection Dashboard"),
    
    html.Div(children="""
        A dashboard to visualize anomalies, their explanations, and the affected infrastructure.
    """),

    # Component for displaying alerts
    html.H2(children="Latest Anomalies Detected"),
    # Dash table or other component to display alerts
    html.Table(),
    
    # Component for the graph visualization
    html.H2(children="Knowledge Graph Visualization"),
    # Use Dash Cytoscape for interactive network graphs
    dcc.Graph(
        id='knowledge-graph',
        figure={} # The figure will be updated by a callback
    ),

    # Component for showing details of a selected anomaly
    html.H2(children="Anomaly Details & Explanation"),
    html.Div(id='explanation-panel'),
])

# 3. Define Callbacks for Interactivity
# Callbacks are essential for making the dashboard interactive
# They listen for changes in an input component and update an output component.

@app.callback(
    Output('knowledge-graph', 'figure'),
    [Input('anomaly-list', 'selected_row')] # Example: a table row selection
)
def update_graph_visualization(selected_anomaly):
    # This callback would fetch and display the relevant part of the knowledge graph.
    if selected_anomaly:
        # Fetch the subgraph from Neo4j based on the selected anomaly
        graph_data = fetch_graph_data(selected_anomaly)
        
        # Create a Plotly graph object from the Neo4j data
        fig = go.Figure()
        # Add nodes and edges to the figure
        
        return fig
    return go.Figure()

@app.callback(
    Output('explanation-panel', 'children'),
    [Input('anomaly-list', 'selected_row')]
)
def update_explanation_panel(selected_anomaly):
    # This callback would display the XAI output and MITRE ATT&CK mapping
    if selected_anomaly:
        # Fetch the explanation and mapping from your API
        # Return a Div with the formatted text and/or a table
        return html.Div([
            html.H3(children=f"Attack Type: {selected_anomaly['attack_type']}"),
            html.P(children=f"Top Features: {selected_anomaly['top_features']}"),
            # You could add more details here
        ])
    return html.Div("Select an anomaly to view details.")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)