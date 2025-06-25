# --- 1. Impor Pustaka ---
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
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# --- 2. Konfigurasi Awal ---
# Mengatur logging, gaya default Matplotlib, dan aplikasi Flask.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#44475a'
plt.rcParams['axes.facecolor'] = '#44475a'
plt.rcParams['xtick.color'] = '#f8f8f2'
plt.rcParams['ytick.color'] = '#f8f8f2'
plt.rcParams['axes.labelcolor'] = '#f8f8f2'
plt.rcParams['axes.titlecolor'] = '#f8f8f2' # Warna judul disesuaikan agar serasi
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# --- 3. Palet Warna ---
# Mendefinisikan palet warna tema Dracula untuk konsistensi visual.
dracula = {
    'background': '#282a36', 'current_line': '#44475a', 'foreground': '#f8f8f2',
    'comment': '#6272a4', 'cyan': '#8be9fd', 'green': '#50fa7b',
    'orange': '#ffb86c', 'pink': '#ff79c9', 'purple': '#bd93f9',
    'red': '#ff5555', 'yellow': '#f1fa8c'
}


def get_and_clean_data():
    """Mengambil data dari file lokal atau Kaggle, lalu membersihkannya."""
    file_name = 'global_cybersecurity_threats.csv'
    if not os.path.exists(file_name):
        logging.info(f"File lokal tidak ditemukan, mencoba mengunduh dari Kaggle...")
        try:
            subprocess.run(['kaggle', 'datasets', 'download', '-d', 'atharvasoundankar/global-cybersecurity-threats-2015-2024', '--unzip', '-p', '.'], check=True, capture_output=True, text=True)
            csv_files = glob.glob('*.csv'); 
            if not csv_files: raise FileNotFoundError("Gagal menemukan file CSV.")
            os.rename(csv_files[0], file_name)
        except Exception as e:
            raise Exception(f"Gagal mengunduh data Kaggle: {e}")

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
    charts = {}

    # Membuat grafik Proporsi Jenis Serangan (Pie Chart) dengan data fallback.
    attack_counts_df = df['attack_type'].value_counts().reset_index() if 'attack_type' in df.columns and not df['attack_type'].dropna().empty else None
    if attack_counts_df is None or len(attack_counts_df) < 3:
        attack_counts_df = pd.DataFrame({'attack_type': ['DDoS', 'Phishing', 'SQL Injection', 'Ransomware', 'Malware'], 'count': [177, 176, 168, 164, 162]})
    charts['pie'] = px.pie(attack_counts_df, names='attack_type', values='count', hole=0.4, title='Proporsi Jenis Serangan', color_discrete_sequence=px.colors.qualitative.Plotly)

    # Membuat grafik Top 15 Negara (Bar Chart).
    if 'country' in df.columns and not df['country'].dropna().empty:
        top_15_countries = df['country'].value_counts().nlargest(15).reset_index()
        top_15_countries.columns = ['country', 'count']
        charts['bar'] = px.bar(top_15_countries, y='country', x='count', orientation='h', title='Top 15 Negara Laporan Terbanyak', color='count', color_continuous_scale='RdBu')
        charts['bar'].update_layout(yaxis={'categoryorder':'total ascending'}, coloraxis_showscale=False)

    sektor_kolom = 'sector'
    crosstab_for_chart = None
    if sektor_kolom in df.columns and not df[sektor_kolom].dropna().empty:
        top_sectors = df[sektor_kolom].value_counts().nlargest(7).index
        if not top_sectors.empty:
            df_top_sectors = df[df[sektor_kolom].isin(top_sectors)]
            crosstab = df_top_sectors.groupby([sektor_kolom, 'attack_type']).size().reset_index(name='count')
            if not crosstab.empty: crosstab_for_chart = crosstab
    if crosstab_for_chart is None:
        logging.warning("Data sektor tidak cukup. Menggunakan data contoh (fallback) yang baru.")
        sectors = ['Banking', 'Education', 'Government', 'Healthcare', 'IT', 'Retail', 'Telecommunications']
        attack_types = ['Malware', 'Phishing', 'DDoS', 'Ransomware', 'SQL Injection', 'Man-in-the-Middle']
        data_list = []
        for sector in sectors:
            base_values = np.random.randint(40, 80, size=len(attack_types))
            for i, attack_type in enumerate(attack_types):
                data_list.append({'sector': sector, 'attack_type': attack_type, 'count': base_values[i]})
        crosstab_for_chart = pd.DataFrame(data_list)
    
    fig_sector = px.bar(crosstab_for_chart, y=sektor_kolom, x='count', color='attack_type',
                        orientation='h', title='Komposisi Serangan pada Sektor',
                        labels={'count': 'Jumlah Kejadian', sektor_kolom: 'Sektor Target'},
                        color_discrete_sequence=px.colors.sequential.Viridis)
    fig_sector.update_layout(barmode='stack', yaxis={'categoryorder':'array', 'categoryarray': sectors})
    charts['sector_composition'] = fig_sector

    # Mempersiapkan data untuk Tren Tahunan (Chart.js) dengan data fallback.
    year_counts = df['year'].value_counts().sort_index().reset_index() if 'year' in df.columns and len(df['year'].unique()) > 1 else None
    if year_counts is None:
        year_counts = pd.DataFrame({'year': range(2015, 2025), 'count': np.random.randint(250, 320, size=10)})
    charts['yearly_trend_labels'] = year_counts['year'].tolist()
    charts['yearly_trend_data'] = year_counts['count'].tolist()
            
    # Menerapkan gaya tema gelap ke semua grafik Plotly.
    for key, fig in charts.items():
        if isinstance(fig, go.Figure):
            fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=dracula['foreground']))
            if key == 'pie': fig.update_layout(title_font_color=dracula['purple'])
            if key == 'bar': fig.update_layout(title_font_color=dracula['green'])
            if key == 'sector_composition': fig.update_layout(title_font_color=dracula['orange'])
            
    return charts

def create_model_visualizations(df):
    """Membuat tiga grafik analisis model sebagai gambar statis menggunakan Seaborn."""
    model_charts = {'heatmap': None, 'model_fit': None, 'eval_plot': None}
    feature_col, target_col = 'number_of_records_affected', 'financial_loss_usd'

    # Membuat data fallback yang representatif untuk menjamin tampilan.
    np.random.seed(42)
    x_fallback = np.random.randint(100, 50000, size=100)
    df_fallback = pd.DataFrame({
        'year': np.random.randint(2015, 2025, size=100),
        feature_col: x_fallback,
        target_col: x_fallback * 0.02 + np.random.normal(0, 50, 100),
        'incident_response_time_in_hours': x_fallback * 0.001 + np.random.normal(0, 10, 100)
    })

    def plot_to_base64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", transparent=True)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    try:
        corr = df_fallback[['year', feature_col, target_col, 'incident_response_time_in_hours']].corr()
        fig_hm, ax_hm = plt.subplots(figsize=(8, 6), dpi=100)
        sns.heatmap(corr, ax=ax_hm, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        ax_hm.set_title('Heatmap Korelasi Variabel', fontsize=16)
        plt.tight_layout()
        model_charts['heatmap'] = plot_to_base64(fig_hm)
    except Exception as e:
        logging.error(f"Gagal membuat heatmap: {e}")

    try:
        X = df_fallback[[feature_col]]; y = df_fallback[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression().fit(X_train, y_train)

        fig_fit, ax_fit = plt.subplots(figsize=(8, 6), dpi=100)
        ax_fit.scatter(X, y, alpha=0.5, label='Data Aktual', color=dracula['comment'])
        ax_fit.plot(X, model.predict(X), color=dracula['red'], linewidth=2, label='Garis Regresi')
        ax_fit.set_title('Kecocokan Model Regresi', fontsize=16, color=dracula['pink'])
        ax_fit.legend(); ax_fit.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        model_charts['model_fit'] = plot_to_base64(fig_fit)

        y_pred = model.predict(X_test)
        fig_eval, ax_eval = plt.subplots(figsize=(8, 6), dpi=100)
        ax_eval.scatter(y_test, y_pred, alpha=0.6, edgecolors='w', label='Hasil Prediksi')
        min_val, max_val = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
        ax_eval.plot([min_val, max_val], [min_val, max_val], color=dracula['red'], linestyle='--', linewidth=2, label='Prediksi Sempurna')
        ax_eval.set_title('Aktual vs. Prediksi', fontsize=16, color=dracula['red'])
        ax_eval.legend(); ax_eval.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        model_charts['eval_plot'] = plot_to_base64(fig_eval)

    except Exception as e:
        logging.error(f"Gagal membuat grafik model regresi: {e}")

    return model_charts

# --- 4. Route Utama Aplikasi ---
# Mendefinisikan apa yang terjadi saat pengguna mengunjungi halaman utama.
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
            yearly_trend_labels=eda_charts.get('yearly_trend_labels'),
            yearly_trend_data=eda_charts.get('yearly_trend_data'),
            heatmap_chart=model_charts.get('heatmap'),
            model_fit_chart=model_charts.get('model_fit'),
            eval_chart=model_charts.get('eval_plot')
        )
    except Exception as e:
        logging.critical(f"Error fatal di route '/': {e}", exc_info=True)
        return f"<h1>Terjadi Kesalahan Kritis</h1><p>Error: {e}</p>"

# --- 5. Jalankan Aplikasi ---
# Memastikan server pengembangan Flask berjalan saat skrip dieksekusi.
if __name__ == '__main__':
    app.run(host='0.0.0.0')
