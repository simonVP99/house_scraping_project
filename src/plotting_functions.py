import plotly.express as px
import pandas as pd

def generate_boxplot(df, x, y = 'price', color = 'property_type'):
    fig = px.box(df, x=x, y=y, color=color, title=f'Box Plot of {y} by {x} and {color}')

    # Update layout for better visualization
    fig.update_layout(
        xaxis_title=f'{x}',
        yaxis_title=f'{y}',
        boxmode='group'  # Group by property type
    )

    # Show the plot
    fig.show()
