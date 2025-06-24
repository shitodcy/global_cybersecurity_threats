# 🌐 Dasbor Analisis Ancaman Siber Global

Selamat datang di **Dasbor Analisis Ancaman Siber Global**!  
Aplikasi web ini dibangun menggunakan **Flask** untuk memvisualisasikan data ancaman siber secara interaktif dan informatif dari tahun **2015 hingga 2024**.

## 📊 Sumber Dataset

Dataset yang digunakan dalam proyek ini tersedia di Kaggle:

🔗 [Global Cybersecurity Threats Analysis – by Dev Raai](https://www.kaggle.com/code/devraai/global-cybersecurity-threats-analysis)


## 🧰 Teknologi

- **Backend**: [Python](https://www.python.org), [Flask](https://palletsprojects.com/projects/flask/)  
- **Analisis & Manipulasi Data**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [Scikit-learn](https://scikit-learn.org/stable/)  
- **Visualisasi**: 
  - [Plotly](https://plotly.com/python/) & [Plotly Express](https://plotly.com/python/plotly-express/)
  - [Chart.js](https://www.chartjs.org/)
  - [Seaborn](https://seaborn.pydata.org/) & [Matplotlib] (https://matplotlib.org/)
- **Frontend**: [HTML5](https://html.spec.whatwg.org/multipage/), [CSS3](https://www.w3.org/TR/css/), [Bootstrap 5](https://getbootstrap.com/)

---

## 📦 Prasyarat

Pastikan Anda telah menginstal:

- [Python 3.8+](https://www.python.org/downloads/)
- [pip](https://pypi.org/project/pip/)
- Akun [Kaggle](https://www.kaggle.com/) + Token API

---

## ⚙️ Cara Menjalankan Proyek

### 1. Kloning Repositori

```bash
git clone https://github.com/shitodcy/global_cybersecurity_threats.git
cd global_cybersecurity_threats
````

### 2. Konfigurasi API Kaggle

* Login ke [Kaggle](https://www.kaggle.com/) dan buka [halaman akun Anda](https://www.kaggle.com/account).
* Klik **Create New API Token** → file `kaggle.json` akan terunduh.
* Letakkan file:

  * **Linux/macOS**: `~/.kaggle/kaggle.json`
  * **Windows**: `C:\Users\<Your-Username>\.kaggle\kaggle.json`

> *Jika belum ada folder `.kaggle`, buat secara manual.*

### 3. Instal Dependensi

```bash
# Buat dan aktifkan virtual environment (opsional tapi direkomendasikan)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate    # Windows

# Instal semua dependensi
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi Flask

```bash
python app.py
```

Buka browser Anda dan akses:
👉 `http://127.0.0.1:5000/`
