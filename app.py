import json
from collections import defaultdict

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go

# Load JSON data
with open('ablation/grid_results.json', 'r') as file:
    data = json.load(file)

# Parse data: key = "weight,lower-upper"
# We'll extract weight, lower, upper from keys, and loss from values
parsed_data = defaultdict(list)  # weight -> list of (lower, upper, loss)
weights_set = set()
lowers_set = set()
uppers_set = set()

for key, loss in data.items():
    weight_part, interval_part = key.split('-')
    weight = float(weight_part.split(',')[0])
    lower = float(weight_part.split(',')[1])
    upper = float(interval_part)

    parsed_data[weight].append((lower, upper, loss))
    weights_set.add(weight)
    lowers_set.add(lower)
    uppers_set.add(upper)

weights = sorted(weights_set)
all_lowers = sorted(lowers_set)
all_uppers = sorted(uppers_set)

lower_min = min(all_lowers)
lower_max = max(all_lowers)
upper_min = min(all_uppers)
upper_max = max(all_uppers)

app = dash.Dash(__name__)
app.title = "Loss vs Weight by Interval"

app.layout = html.Div([
    html.H2("Loss vs Weight filtered by interval limits"),
    html.Div([
        html.Label("Global Lower Bound:"),
        dcc.Slider(
            id='global-lower',
            min=lower_min,
            max=lower_max,
            step=0.01,
            value=lower_min,
            marks={round(val, 2): str(round(val, 2)) for val in all_lowers},
            tooltip={"placement": "bottom"}
        ),
        html.Label("Global Upper Bound:"),
        dcc.Slider(
            id='global-upper',
            min=upper_min,
            max=upper_max,
            step=0.01,
            value=upper_max,
            marks={round(val, 2): str(round(val, 2)) for val in all_uppers},
            tooltip={"placement": "bottom"}
        )
    ], style={'margin': '20px'}),
    dcc.Graph(id='loss-vs-weight')
    
])


@app.callback(
    Output('loss-vs-weight', 'figure'),
    Input('global-lower', 'value'),
    Input('global-upper', 'value')
)
def update_graph(global_lower, global_upper):
    fig = go.Figure()
    # For each weight, find intervals fully inside [global_lower, global_upper]
    for weight in weights:
        intervals = parsed_data[weight]
        # Filter intervals whose [lower, upper] is inside [global_lower, global_upper]
        filtered_losses = [loss for (lower, upper, loss) in intervals
                           if lower >= global_lower and upper <= global_upper]
        # Aggregate loss: average or min? Let's do average if multiple intervals match
        if filtered_losses:
            avg_loss = sum(filtered_losses) / len(filtered_losses)
            fig.add_trace(go.Scatter(
                x=[weight],
                y=[avg_loss],
                mode='markers',
                name=f'Weight {weight:.2f}'
            ))

    fig.update_layout(
        title='Loss vs Weight (interval filtered)',
        xaxis_title='Weight',
        yaxis_title='Loss',
        template='plotly_white',
        height=600,
        showlegend=False
    )
    return fig


if __name__ == '__main__':
    app.run(debug=True)