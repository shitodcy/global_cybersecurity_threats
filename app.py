import os
import pandas as pd
import numpy as np
import json
from flask import Flask, render_template
import plotly
import plotly.express as px
import plotly.graph_objects as go
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

dracula = {
    'background': '#282a36', 'current_line': '#44475a', 'foreground': '#f8f8f2',
    'comment': '#6272a4', 'cyan': '#8be9fd', 'green': '#50fa7b',
    'orange': '#ffb86c', 'pink': '#ff79c9', 'purple': '#bd93f9',
    'red': '#ff5555', 'yellow': '#f1fa8c'
}

def get_and_clean_data():
    """Membaca dan membersihkan data dari file CSV dengan pembersihan yang teliti."""
    file_name = 'etc/dataseet/cleaned_encoded_global_cybersecurity_threats.csv'
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        logging.error(f"FATAL: File tidak ditemukan di path: {file_name}")
        return None

    df_cleaned = df.copy()
    df_cleaned.columns = [col.strip().lower().replace(' ', '_').replace('(', '').replace(')', '') for col in df_cleaned.columns]

    rename_map = {
        'number_of_affected_users': 'number_of_records_affected',
        'target_industry': 'sector',
        'financial_loss_in_million_$': 'financial_loss_usd'
    }
    df_cleaned.rename(columns=rename_map, inplace=True)

    if 'financial_loss_usd' in df_cleaned.columns:
        df_cleaned['financial_loss_usd'] = df_cleaned['financial_loss_usd'].astype(str).str.replace(r'[^\d.]', '', regex=True)
        df_cleaned['financial_loss_usd'] = pd.to_numeric(df_cleaned['financial_loss_usd'], errors='coerce')
        df_cleaned['financial_loss_usd'] = df_cleaned['financial_loss_usd'].fillna(0) * 1_000_000
    else:
        df_cleaned['financial_loss_usd'] = 0

    other_numeric_cols = ['number_of_records_affected', 'incident_resolution_time_in_hours', 'year']
    for col in other_numeric_cols:
        if col in df_cleaned.columns:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce').fillna(0)
    
    df_cleaned.dropna(subset=['year', 'attack_type', 'country', 'sector'], inplace=True)
    df_cleaned.drop_duplicates(inplace=True)
    
    if 'year' in df_cleaned.columns:
        df_cleaned = df_cleaned[df_cleaned['year'] > 0]
        df_cleaned['year'] = df_cleaned['year'].astype(int)
    
    return df_cleaned

def create_visualizations(df):
    """Membuat semua visualisasi (EDA dan Model) dalam satu fungsi."""
    charts = {}

    # 1. Visualisasi EDA
    charts['pie'] = px.pie(df, names='attack_type', hole=0.4, title='Proporsi Jenis Serangan')
    top_15_countries = df['country'].value_counts().nlargest(15).reset_index()
    charts['bar'] = px.bar(top_15_countries, y='country', x='count', orientation='h', title='Top 15 Negara Laporan Terbanyak', color='count', color_continuous_scale='RdBu')
    charts['bar'].update_layout(yaxis={'categoryorder':'total ascending'}, coloraxis_showscale=False)
    crosstab_df = pd.crosstab(df['sector'], df['attack_type']).reset_index().melt(id_vars='sector')
    charts['sector_composition'] = px.bar(crosstab_df, y='sector', x='value', color='attack_type', orientation='h', title='Komposisi Serangan pada Sektor', labels={'value': 'Jumlah Kejadian', 'sector': 'Sektor Target', 'attack_type': 'Jenis Serangan'})
    charts['sector_composition'].update_layout(barmode='stack', yaxis={'categoryorder':'total ascending'})
    year_counts = df['year'].value_counts().sort_index().reset_index()
    charts['yearly_trend_labels'] = year_counts['year'].tolist()
    charts['yearly_trend_data'] = year_counts['count'].tolist()

    # 2. Visualisasi Model
    feature_col, target_col = 'number_of_records_affected', 'financial_loss_usd'
    df_model = df[(df[feature_col] > 0) & (df[target_col] > 0)].copy()

    if df_model.empty:
        charts['heatmap'] = go.Figure().update_layout(title_text="Heatmap (Data tidak valid)")
        charts['regression_table_chart'] = go.Figure().update_layout(title_text="Statistik Regresi (Data tidak valid)")
        charts['prediction_chart'] = go.Figure().update_layout(title_text="Prediksi Kerugian (Data tidak valid)")
    else:
        # Heatmap Korelasi
        corr_matrix = df_model[[feature_col, target_col, 'year', 'incident_resolution_time_in_hours']].corr()
        labels_map = {'year': 'Tahun', 'number_of_records_affected': 'Jml Record Terdampak', 'financial_loss_usd': 'Kerugian Finansial', 'incident_resolution_time_in_hours': 'Waktu Respons'}
        corr_display = corr_matrix.rename(columns=labels_map, index=labels_map)
        charts['heatmap'] = go.Figure(data=go.Heatmap(z=corr_display.values, x=corr_display.columns, y=corr_display.index, colorscale='RdBu', zmid=0, text=np.round(corr_display.values, 2), texttemplate="%{text}", textfont={"size": 10}))
        charts['heatmap'].update_layout(title={'text': 'Heatmap Korelasi', 'x': 0.5}, margin=dict(l=120, r=50, t=80, b=80), xaxis_tickangle=-45)

        # Model Regresi dan Statistik
        X = df_model[[feature_col]]
        y = df_model[target_col]
        model = LinearRegression().fit(X, y)
        r2 = model.score(X, y)

        from plotly.subplots import make_subplots

        # Grafik Statistik Model Regresi
        fig_stats = make_subplots(
            rows=3, cols=1, 
            subplot_titles=("Intercept (b₀)", "Koefisien Regresi (b₁)", "R-squared (R²)"),
            vertical_spacing=0.15
        )

        # Tambahkan bar untuk setiap metrik
        fig_stats.add_trace(go.Bar(x=[model.intercept_], y=[' '], orientation='h', marker_color=dracula['cyan'], text=f"${model.intercept_:,.0f}", textposition='auto'), row=1, col=1)
        fig_stats.add_trace(go.Bar(x=[model.coef_[0]], y=[' '], orientation='h', marker_color=dracula['green'], text=f"{model.coef_[0]:.4f}", textposition='auto'), row=2, col=1)
        fig_stats.add_trace(go.Bar(x=[r2], y=[' '], orientation='h', marker_color=dracula['orange'], text=f"{r2:.4f}", textposition='auto'), row=3, col=1)

        fig_stats.update_layout(
            title={'text': 'Grafik Statistik Model Regresi', 'x': 0.5},
            showlegend=False,
            margin=dict(l=10, r=10, t=80, b=10),
            height=400 # Sesuaikan tinggi agar pas
        )
        # Sembunyikan label sumbu y karena judul subplot sudah cukup
        fig_stats.update_yaxes(showticklabels=False)

        charts['regression_table_chart'] = fig_stats

        # Grafik Prediksi Kerugian per Negara
        country_data = df.groupby('country')[[feature_col]].sum().reset_index()
        country_data['predicted_loss'] = model.predict(country_data[[feature_col]])
        country_data_sorted = country_data.sort_values('predicted_loss', ascending=False)
        
        # LOGGING DIAGNOSTIK: Cetak DataFrame untuk memastikan semua negara disertakan
        logging.info("DataFrame untuk grafik prediksi (Seluruh Negara):")
        logging.info(country_data_sorted)

        charts['prediction_chart'] = px.bar(country_data_sorted, x='country', y='predicted_loss', 
                                              title='Prediksi Kerugian Finansial per Negara (Seluruh Negara)',
                                              labels={'country': 'Negara', 'predicted_loss': 'Prediksi Kerugian (USD)'})
        
        # Memaksa urutan sumbu-x sesuai dengan data yang sudah diurutkan
        charts['prediction_chart'].update_xaxes(categoryorder='array', categoryarray=country_data_sorted['country'])

    # Atur tema untuk semua grafik
    for key, fig in charts.items():
        if isinstance(fig, go.Figure):
            fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=dracula['foreground']))
            if 'title' in fig.layout and fig.layout.title.font is not None:
                color_map = {
                    'bar': dracula['green'], 'pie': dracula['purple'], 'sector_composition': dracula['orange'],
                    'heatmap': dracula['purple'], 'regression_table_chart': dracula['pink'],
                    'prediction_chart': dracula['cyan']
                }
                if key in color_map:
                    fig.layout.title.font.color = color_map.get(key)

    return charts

@app.route('/')
def dashboard():
    df_cleaned = get_and_clean_data()
    if df_cleaned is None:
        return "<h1>Error Kritis</h1><p>File dataset tidak ditemukan.</p>", 500

    charts = create_visualizations(df_cleaned)

    def fig_to_json(fig):
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder) if fig else 'null'
    
    kpi = { 
        "total_incidents": f"{len(df_cleaned):,}", 
        "total_loss": f"${df_cleaned['financial_loss_usd'].sum():,.0f}", 
        "total_users_affected": f"{df_cleaned['number_of_records_affected'].sum():,}" 
    }

    return render_template(
        'index.html', kpi=kpi, 
        bar_chart=fig_to_json(charts.get('bar')),
        pie_chart=fig_to_json(charts.get('pie')),
        sector_composition_chart=fig_to_json(charts.get('sector_composition')),
        yearly_trend_labels=json.dumps(charts.get('yearly_trend_labels')),
        yearly_trend_data=json.dumps(charts.get('yearly_trend_data')),
        heatmap_chart=fig_to_json(charts.get('heatmap')),
        regression_table_chart=fig_to_json(charts.get('regression_table_chart')),
        prediction_chart=fig_to_json(charts.get('prediction_chart'))
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)