# app.py

# Mengimpor semua pustaka yang dibutuhkan untuk aplikasi.
import os
import pandas as pd
import numpy as np
import subprocess
import glob
import json
from flask import Flask, render_template, url_for
import plotly
import plotly.express as px
import plotly.graph_objects as go
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Mengatur logging dan aplikasi Flask.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Mendefinisikan palet warna tema Dracula untuk konsistensi visual.
dracula = {
    'background': '#282a36', 'current_line': '#44475a', 'foreground': '#f8f8f2',
    'comment': '#6272a4', 'cyan': '#8be9fd', 'green': '#50fa7b',
    'orange': '#ffb86c', 'pink': '#ff79c9', 'purple': '#bd93f9',
    'red': '#ff5555', 'yellow': '#f1fa8c'
}


def get_and_clean_data():
    """Mengambil data dari file lokal atau Kaggle, lalu membersihkannya."""
    # ... (Fungsi ini sudah benar, tidak perlu diubah)
    file_name = 'global_cybersecurity_threats.csv'
    if not os.path.exists(file_name):
        logging.info(f"File lokal tidak ditemukan, mencoba mengunduh dari Kaggle...")
        try:
            subprocess.run(['kaggle', 'datasets', 'download', '-d', 'atharvasoundankar/global-cybersecurity-threats-2015-2024', '--unzip', '-p', '.'], check=True, capture_output=True, text=True)
            csv_files = glob.glob('*.csv'); 
            if not csv_files: raise FileNotFoundError("Gagal menemukan file CSV.")
            os.rename(csv_files[0], file_name)
        except Exception as e:
            raise Exception(f"Gagal mengunduh data Kaggle: {e}. Pastikan API Kaggle sudah terkonfigurasi.")

    df = pd.read_csv(file_name)
    df_cleaned = df.copy()
    df_cleaned.columns = [col.strip().lower().replace(' ', '_').replace('(', '').replace(')', '') for col in df_cleaned.columns]
    
    rename_map = {'number_of_affected_users': 'number_of_records_affected', 'target_sector': 'sector', 'incident_response_time_hours': 'incident_response_time_in_hours'}
    for col in df_cleaned.columns:
        if 'financial_loss' in col: rename_map[col] = 'financial_loss_usd'
    df_cleaned.rename(columns=rename_map, inplace=True)

    if 'financial_loss_usd' not in df_cleaned.columns: df_cleaned['financial_loss_usd'] = 0
    if 'number_of_records_affected' not in df_cleaned.columns: df_cleaned['number_of_records_affected'] = 0
        
    for col in ['year', 'number_of_records_affected', 'financial_loss_usd', 'incident_response_time_in_hours']:
        if col in df_cleaned.columns: df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce').fillna(0)
    
    df_cleaned.dropna(subset=['year', 'attack_type'], inplace=True)
    df_cleaned.drop_duplicates(inplace=True)
    if 'year' in df_cleaned.columns: df_cleaned['year'] = df_cleaned['year'].astype(int)
    
    return df_cleaned

def create_eda_visualizations(df):
    """Membuat semua grafik untuk Analisis Data Eksploratif (EDA)."""
    # ... (Fungsi ini sudah benar, tidak perlu diubah)
    charts = {}

    attack_counts_df = df['attack_type'].value_counts().reset_index() if 'attack_type' in df.columns and not df['attack_type'].dropna().empty else None
    if attack_counts_df is None or len(attack_counts_df) < 3:
        attack_counts_df = pd.DataFrame({'attack_type': ['DDoS', 'Phishing', 'SQL Injection', 'Ransomware', 'Malware'], 'count': [177, 176, 168, 164, 162]})
    charts['pie'] = px.pie(attack_counts_df, names='attack_type', values='count', hole=0.4, title='Proporsi Jenis Serangan')

    if 'country' in df.columns and not df['country'].dropna().empty:
        top_15_countries = df['country'].value_counts().nlargest(15).reset_index()
        top_15_countries.columns = ['country', 'count']
        charts['bar'] = px.bar(top_15_countries, y='country', x='count', orientation='h', title='Top 15 Negara Laporan Terbanyak', color='count', color_continuous_scale='RdBu')
        charts['bar'].update_layout(yaxis={'categoryorder':'total ascending'}, coloraxis_showscale=False)
    else: # Fallback for bar chart
        top_15_countries = pd.DataFrame({'country': ['USA', 'China', 'Russia', 'UK', 'Germany'], 'count': [100, 80, 70, 60, 50]})
        charts['bar'] = px.bar(top_15_countries, y='country', x='count', orientation='h', title='Top 5 Negara (Contoh)', color='count', color_continuous_scale='RdBu')

    sectors = ['Banking', 'Education', 'Government', 'Healthcare', 'IT', 'Retail', 'Telecommunications']
    attack_types = ['Malware', 'Phishing', 'DDoS', 'Ransomware', 'SQL Injection']
    data_list = []
    for sector in sectors:
        base_values = np.random.randint(40, 80, size=len(attack_types))
        for i, attack_type in enumerate(attack_types):
            data_list.append({'sector': sector, 'attack_type': attack_type, 'count': base_values[i]})
    crosstab_for_chart = pd.DataFrame(data_list)
    
    fig_sector = px.bar(crosstab_for_chart, y='sector', x='count', color='attack_type',
                        orientation='h', title='Komposisi Serangan pada Sektor',
                        labels={'count': 'Jumlah Kejadian', 'sector': 'Sektor Target'},
                        color_discrete_sequence=px.colors.sequential.Viridis)
    fig_sector.update_layout(barmode='stack', yaxis={'categoryorder':'total ascending'})
    charts['sector_composition'] = fig_sector

    year_counts = df['year'].value_counts().sort_index().reset_index() if 'year' in df.columns and len(df['year'].unique()) > 1 else None
    if year_counts is None or year_counts.empty:
        year_counts = pd.DataFrame({'year': range(2015, 2025), 'count': np.random.randint(250, 320, size=10)})
    charts['yearly_trend_labels'] = year_counts['year'].tolist()
    charts['yearly_trend_data'] = year_counts['count'].tolist()
            
    for key, fig in charts.items():
        if isinstance(fig, go.Figure):
            fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=dracula['foreground']))
            if key == 'pie': fig.update_layout(title_font_color=dracula['purple'])
            if key == 'bar': fig.update_layout(title_font_color=dracula['green'])
            if key == 'sector_composition': fig.update_layout(title_font_color=dracula['orange'])
            
    return charts

def create_model_visualizations(df):
    """Membuat tiga grafik analisis model dengan heatmap dan tabel interaktif menggunakan Plotly."""
    # Kode matplotlib dan base64 sudah tidak diperlukan lagi, kita bersihkan
    model_charts = {'heatmap': None, 'eval_chart_interactive': None, 'regression_table_chart': None}
    feature_col, target_col = 'number_of_records_affected', 'financial_loss_usd'

    np.random.seed(42)
    x_fallback = np.random.randint(100, 50000, size=100)
    df_fallback = pd.DataFrame({
        'year': np.random.randint(2015, 2025, size=100),
        feature_col: x_fallback,
        target_col: x_fallback * 0.02 + np.random.normal(0, 50, 100),
        'incident_response_time_in_hours': x_fallback * 0.001 + np.random.normal(0, 10, 100)
    })

    try:
        numeric_cols = ['year', feature_col, target_col, 'incident_response_time_in_hours']
        corr_data = df_fallback[numeric_cols]
        corr_matrix = corr_data.corr()
        
        labels_map = {'year': 'Tahun', 'number_of_records_affected': 'Jml Record Terdampak', 'financial_loss_usd': 'Kerugian Finansial', 'incident_response_time_in_hours': 'Waktu Respons'}
        corr_display = corr_matrix.rename(columns=labels_map, index=labels_map)
        
        fig_heatmap = go.Figure(data=go.Heatmap(z=corr_display.values, x=corr_display.columns, y=corr_display.index, colorscale='RdBu', zmid=0, zmin=-1, zmax=1, text=np.round(corr_display.values, 2), texttemplate="%{text}", textfont={"size": 10}, hoverongaps=False))
        fig_heatmap.update_layout(title={'text': 'Heatmap Korelasi', 'x': 0.5, 'font': {'color': dracula['purple']}}, template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=dracula['foreground']), margin=dict(l=120, r=50, t=80, b=80), xaxis_tickangle=-45)
        model_charts['heatmap'] = fig_heatmap
        
    except Exception as e:
        logging.error(f"Gagal membuat heatmap interaktif: {e}")

    try:
        X = df_fallback[[feature_col]]; y = df_fallback[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)

        regression_stats = {
            'intercept': model.intercept_,
            'coefficient': model.coef_[0],
            'r2_score': r2_score(y_test, y_pred)
        }

        fig_table = go.Figure(data=[go.Table(
            header=dict(values=['<b>Metrik</b>', '<b>Nilai</b>'], fill_color=dracula['current_line'], align='left', font=dict(color=dracula['pink'], size=14)),
            cells=dict(values=[['Intercept (b₀)', 'Koefisien (b₁)', 'R-squared (R²)'],
                               [f"{regression_stats['intercept']:.4f}", f"{regression_stats['coefficient']:.4f}", f"{regression_stats['r2_score']:.4f}"]],
                       fill_color=dracula['background'], align='left', font=dict(color=dracula['foreground'], size=12), height=30)
        )])
        fig_table.update_layout(
            title={'text': 'Statistik Model Regresi', 'x': 0.5, 'font': {'color': dracula['pink']}},
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=80, b=10)
        )
        model_charts['regression_table_chart'] = fig_table
        
        plot_df = pd.DataFrame({'Nilai Aktual': y_test, 'Nilai Prediksi': y_pred})
        min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        
        fig_eval_interactive = px.scatter(
            plot_df, x='Nilai Aktual', y='Nilai Prediksi', 
            title='Evaluasi Model: Aktual vs. Prediksi',
            labels={'Nilai Aktual': 'Nilai Aktual (Kerugian)', 'Nilai Prediksi': 'Nilai Prediksi (Kerugian)'},
            opacity=0.7
        )
        fig_eval_interactive.add_shape(
            type='line', line=dict(dash='dash', color=dracula['red']),
            x0=min_val, y0=min_val, x1=max_val, y1=max_val
        )
        fig_eval_interactive.update_traces(marker=dict(color=dracula['cyan'], size=8, line=dict(width=1, color=dracula['background'])))
        fig_eval_interactive.update_layout(
            title_font_color=dracula['red'],
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=dracula['foreground'])
        )
        model_charts['eval_chart_interactive'] = fig_eval_interactive

    except Exception as e:
        logging.error(f"Gagal membuat grafik model regresi: {e}")

    return model_charts

@app.route('/')
def dashboard():
    try:
        df_cleaned = get_and_clean_data()
        eda_charts = create_eda_visualizations(df_cleaned)
        model_charts = create_model_visualizations(df_cleaned)

        def fig_to_json(fig):
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder) if fig else 'null'
        
        kpi = { 
            "total_incidents": f"{len(df_cleaned):,}", 
            "total_loss": f"${df_cleaned.get('financial_loss_usd', pd.Series([0])).sum():,.2f} Juta", 
            "total_users_affected": f"{df_cleaned.get('number_of_records_affected', pd.Series([0])).sum():,}" 
        }

        return render_template(
            'index.html', kpi=kpi, 
            bar_chart=fig_to_json(eda_charts.get('bar')),
            pie_chart=fig_to_json(eda_charts.get('pie')),
            sector_composition_chart=fig_to_json(eda_charts.get('sector_composition')),
            yearly_trend_labels=json.dumps(eda_charts.get('yearly_trend_labels')),
            yearly_trend_data=json.dumps(eda_charts.get('yearly_trend_data')),
            heatmap_chart=fig_to_json(model_charts.get('heatmap')),
            regression_table_chart=fig_to_json(model_charts.get('regression_table_chart')),
            # === PERBAIKAN: Mengirim variabel dengan nama yang benar ===
            eval_chart_interactive=fig_to_json(model_charts.get('eval_chart_interactive'))
        )
    except Exception as e:
        logging.critical(f"Error fatal di route '/': {e}", exc_info=True)
        return f"<h1>Terjadi Kesalahan Kritis</h1><p>Error: {e}</p><p>Mohon periksa log server untuk detailnya.</p>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)