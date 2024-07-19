import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="dask")
import os
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import joblib

# Load the prediction results
predictions = pd.read_csv('new_data_predictions.csv')
top_10_churn = predictions.nlargest(10, '예측이탈률')
model_performance = joblib.load('model_performance.pkl')

# Load the model to get feature importances
ensemble_model = joblib.load('retrained_model.pkl')
feature_importances = ensemble_model.feature_importances().reset_index().rename(columns={'index': '변수', 'importance': '중요도'})

# Extracting metrics from classification_report
accuracy = model_performance['accuracy']
precision = model_performance['weighted avg']['precision']
recall = model_performance['weighted avg']['recall']
f1_score = model_performance['weighted avg']['f1-score']

# Initialize the Dash app
app = Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("고객 이탈 예측 대시보드"),

    # Top 10 customers with the highest churn probability
    html.H2("예측 이탈률 상위 10명 고객"),
    html.Table([
        html.Thead(html.Tr([html.Th(col) for col in top_10_churn.columns])),
        html.Tbody([
            html.Tr([
                html.Td(top_10_churn.iloc[i][col], style={'color': 'blue'} if i < 10 else {})
                for col in top_10_churn.columns
            ]) for i in range(len(top_10_churn))
        ])
    ]),

    # Model performance metrics
    html.H2("모델 성능"),
    html.Table([
        html.Tr([html.Th('평가 지표'), html.Th('값')]),
        html.Tr([html.Td('정확도'), html.Td(f"{accuracy:.2f}")]),
        html.Tr([html.Td('정밀도'), html.Td(f"{precision:.2f}")]),
        html.Tr([html.Td('재현율'), html.Td(f"{recall:.2f}")]),
        html.Tr([html.Td('F1 점수'), html.Td(f"{f1_score:.2f}")]),
    ]),

    # Feature importances
    html.H2("변수 중요도"),
    dcc.Graph(id='feature-importances-bar', figure=px.bar(feature_importances, x='변수', y='중요도', title='변수 중요도')),

    # Dropdowns for filtering
    html.Div([
        html.Div([
            html.Label('성별 필터'),
            dcc.Dropdown(
                id='gender-filter',
                options=[{'label': gender, 'value': gender} for gender in predictions['성별'].unique()],
                value=None,
                multi=True,
                placeholder="성별 선택"
            ),
        ], style={'width': '30%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label('소득 수준'),
            dcc.Dropdown(
                id='income-filter',
                options=[{'label': income, 'value': income} for income in predictions['소득'].unique()],
                value=None,
                multi=True,
                placeholder="소득 수준 선택"
            ),
        ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '20px'}),
        
        html.Div([
            html.Label('교육 수준'),
            dcc.Dropdown(
                id='education-filter',
                options=[{'label': edu, 'value': edu} for edu in predictions['교육수준'].unique()],
                value=None,
                multi=True,
                placeholder="교육 수준 선택"
            ),
        ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '20px'}),
        
        html.Div([
            html.Label('결혼 여부'),
            dcc.Dropdown(
                id='marital-filter',
                options=[{'label': marital, 'value': marital} for marital in predictions['결혼상태'].unique()],
                value=None,
                multi=True,
                placeholder="결혼 여부 선택"
            ),
        ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '20px'}),
    ], style={'marginBottom': '20px'}),

    # Histogram of churn probabilities
    html.H2("이탈률 분포"),
    dcc.Graph(id='churn-histogram'),

    # Bar chart of churn counts by gender
    html.H2("성별에 따른 이탈자 수"),
    dcc.Graph(id='churn-gender-bar'),

    # Bar chart of churn counts by income category
    html.H2("소득 수준에 따른 이탈자 수"),
    dcc.Graph(id='churn-income-bar'),

    # Bar chart of churn counts by education level
    html.H2("교육 수준에 따른 이탈자 수"),
    dcc.Graph(id='churn-education-bar'),

    # Bar chart of churn counts by marital status
    html.H2("결혼 여부에 따른 이탈자 수"),
    dcc.Graph(id='churn-marital-bar'),

    # Display a table with the predictions
    html.H2("모든 고객 예측표"),
    dash_table.DataTable(
        id='churn-table',
        columns=[{'name': col, 'id': col} for col in predictions.columns],
        data=predictions.to_dict('records'),
        style_data_conditional=[
            {
                'if': {
                    'filter_query': '{예측이탈률} > 0.9',
                    'column_id': '예측이탈률'
                },
                'backgroundColor': 'tomato',
                'color': 'white'
            }
        ],
        sort_action='native',
        filter_action='native',
        page_size=10
    )
])

# Callback to update the visualizations based on the selected filters
@app.callback(
    [Output('churn-histogram', 'figure'),
     Output('churn-gender-bar', 'figure'),
     Output('churn-income-bar', 'figure'),
     Output('churn-education-bar', 'figure'),
     Output('churn-marital-bar', 'figure'),
     Output('churn-table', 'data')],
    [Input('gender-filter', 'value'),
     Input('income-filter', 'value'),
     Input('education-filter', 'value'),
     Input('marital-filter', 'value')]
)
def update_graphs(selected_genders, selected_income_categories, selected_educations, selected_maritals):
    filtered_predictions = predictions.copy()

    if selected_genders:
        filtered_predictions = filtered_predictions[filtered_predictions['성별'].isin(selected_genders)]
    if selected_income_categories:
        filtered_predictions = filtered_predictions[filtered_predictions['소득'].isin(selected_income_categories)]
    if selected_educations:
        filtered_predictions = filtered_predictions[filtered_predictions['교육수준'].isin(selected_educations)]
    if selected_maritals:
        filtered_predictions = filtered_predictions[filtered_predictions['결혼상태'].isin(selected_maritals)]

    # Histogram of churn probabilities
    churn_histogram = px.histogram(
        filtered_predictions, 
        x='예측이탈률', 
        nbins=50, 
        title='이탈률 분포'
    )

    # Bar chart of churn counts by gender
    churn_gender_bar = px.bar(
        filtered_predictions.groupby('성별').size().reset_index(name='Counts'), 
        x='성별', 
        y='Counts', 
        title='성별에 따른 이탈자 수'
    )

    # Bar chart of churn counts by income category
    churn_income_bar = px.bar(
        filtered_predictions.groupby('소득').size().reset_index(name='Counts'), 
        x='소득', 
        y='Counts', 
        title='소득 수준에 따른 이탈자 수'
    )

    # Bar chart of churn counts by education level
    churn_education_bar = px.bar(
        filtered_predictions.groupby('교육수준').size().reset_index(name='Counts'), 
        x='교육수준', 
        y='Counts', 
        title='교육 수준에 따른 이탈자 수'
    )

    # Bar chart of churn counts by marital status
    churn_marital_bar = px.bar(
        filtered_predictions.groupby('결혼상태').size().reset_index(name='Counts'), 
        x='결혼상태', 
        y='Counts', 
        title='결혼 여부에 따른 이탈자 수'
    )

    return churn_histogram, churn_gender_bar, churn_income_bar, churn_education_bar, churn_marital_bar, filtered_predictions.to_dict('records')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run_server(debug=True, port=port)
