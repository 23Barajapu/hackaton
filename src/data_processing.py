import os
import re
import joblib
import json
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import timeseries_dataset_from_array
from supabase import create_client, Client
from dotenv import load_dotenv

import config

# Muat variabel.env
load_dotenv()

def _clean_numeric_string(s: str) -> float:
    """Membersihkan format angka '54 987,79'  menjadi 54987.79"""
    if not isinstance(s, str):
        return s
    s = re.sub(r'\s', '', s)  # Hapus spasi
    s = s.replace(',', '.')   # Ganti koma dengan titik
    try:
        return float(s)
    except (ValueError, TypeError):
        return np.nan

def _get_db_engine():
    """Membuat koneksi DB langsung (SQLAlchemy) untuk data besar (pelatihan)."""
    db_url = os.environ.get("SUPABASE_DB_URL_POOLER")
    if not db_url:
        raise ValueError("SUPABASE_DB_URL_POOLER tidak diatur di.env")
    # Gunakan NullPool saat terhubung ke pooler mode transaksi (Port 6543) [2, 3, 4, 5, 71, 72]
    engine = create_engine(db_url, poolclass=NullPool)
    return engine

def _get_api_client() -> Client:
    """Membuat klien API Supabase (supabase-py) untuk kueri kecil (prediksi)."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL atau SUPABASE_KEY tidak diatur di.env")
    return create_client(url, key) [12, 63, 73, 2, 74, 10, 75, 11, 76, 77]

def load_training_data() -> (pd.DataFrame, pd.DataFrame):
    """Menarik SEMUA data panen dan cuaca untuk pelatihan model."""
    print("Menarik data pelatihan dari Supabase (via koneksi DB langsung)...")
    engine = _get_db_engine()
    # Ganti 'harvest_data' dan 'weather_events' dengan nama tabel Anda di Supabase
    try:
        df_harvest = pd.read_sql_query("SELECT * FROM harvest_data", con=engine) [78, 79, 80, 81, 63, 3, 4, 82, 9, 83, 84]
        df_weather = pd.read_sql_query("SELECT * FROM weather_events", con=engine) [78, 79, 80, 81, 63, 3, 4, 82, 9, 83, 84]
        print(f"Data panen dimuat: {df_harvest.shape} baris")
        print(f"Data cuaca dimuat: {df_weather.shape} baris")
        return df_harvest, df_weather
    except Exception as e:
        print(f"Gagal menarik data: {e}")
        return pd.DataFrame(), pd.DataFrame()

def load_prediction_data(region_name: str, start_date: str) -> (pd.DataFrame, pd.DataFrame):
    """Menarik data terbaru untuk satu wilayah guna membuat prediksi."""
    print(f"Menarik data prediksi untuk {region_name} dari Supabase (via API)...")
    client = _get_api_client()
    
    # Ganti nama tabel dan kolom jika berbeda
    harvest_response = client.table("harvest_data").select("*").eq("Kabupaten/Kota", region_name).execute() [85]
    weather_response = client.table("weather_events").select("*").eq("Kabupaten/Kota", region_name).gte("Tanggal", start_date).execute() [85]
    
    return pd.DataFrame(harvest_response.data), pd.DataFrame(weather_response.data)

def load_data_from_csv():
    """Memuat data dari file CSV lokal untuk pengembangan/testing."""
    print("Memuat data dari file CSV lokal...")
    
    # Gunakan path absolut berdasarkan lokasi file ini
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # Naik satu level dari src/ ke root
    PANEN_DATA_PATH = os.path.join(project_root, 'data', 'sample_data_panen.csv')
    CUACA_DATA_PATH = os.path.join(project_root, 'data', 'sample_data_cuaca.csv')
    
    df_harvest = pd.read_csv(PANEN_DATA_PATH, sep=';')
    df_weather = pd.read_csv(CUACA_DATA_PATH, sep=';')
    df_weather[config.DATE_COLUMN] = pd.to_datetime(df_weather[config.DATE_COLUMN])
    
    print(f"Data panen dimuat: {df_harvest.shape} baris")
    print(f"Data cuaca dimuat: {df_weather.shape} baris")
    return df_harvest, df_weather

def preprocess_features(df_harvest: pd.DataFrame, df_weather: pd.DataFrame, scaler=None, is_training=True):
    """
    Inti dari pipeline ML. Mengubah data mentah menjadi sekuens yang siap untuk GRU.
    """
    print("Memulai pra-pemrosesan fitur...")
    
    # === 1. Pra-pemrosesan Data Panen (Y dan Fitur X) ===
    
    # Ganti nama kolom agar lebih mudah digunakan
    # Cek kolom yang ada terlebih dahulu
    rename_dict = {}
    if config.TARGET_COLUMN in df_harvest.columns:
        rename_dict[config.TARGET_COLUMN] = "Produktivitas"
    if config.REGION_COLUMN in df_harvest.columns:
        rename_dict[config.REGION_COLUMN] = "Wilayah"
    if "Luas Panen Tanaman Padi (ha) (Ha)" in df_harvest.columns:
        rename_dict["Luas Panen Tanaman Padi (ha) (Ha)"] = "LuasPanen"
    
    if rename_dict:
        df_harvest = df_harvest.rename(columns=rename_dict)
    
    # Jika kolom belum ada setelah rename, coba cari dengan nama yang mirip
    if "Produktivitas" not in df_harvest.columns:
        # Cari kolom yang mengandung "Produktivitas"
        prod_cols = [col for col in df_harvest.columns if "Produktivitas" in col or "produktivitas" in col.lower()]
        if prod_cols:
            df_harvest = df_harvest.rename(columns={prod_cols[0]: "Produktivitas"})
    
    if "LuasPanen" not in df_harvest.columns:
        # Cari kolom yang mengandung "Luas"
        luas_cols = [col for col in df_harvest.columns if "Luas" in col or "luas" in col.lower()]
        if luas_cols:
            df_harvest = df_harvest.rename(columns={luas_cols[0]: "LuasPanen"})

    # Bersihkan angka (misal, "54 987,79" -> 54987.79) 
    for col in ["Produktivitas", "LuasPanen"]:
        if col in df_harvest.columns:
            df_harvest[col] = df_harvest[col].apply(_clean_numeric_string)
    df_harvest = df_harvest.dropna(subset=["Produktivitas", "LuasPanen"])
    
    # Buat label (Y) hanya jika pelatihan
    if is_training:
        # Hitung Z-score produktivitas PER WILAYAH untuk menemukan anomali [86, 87, 88, 89, 90, 91]
        df_harvest['z_score'] = df_harvest.groupby('Wilayah')['Produktivitas'].transform(
            lambda x: (x - x.mean()) / x.std(ddof=0)
        )
        # Label 1 (Gagal Panen) jika Z-score di bawah ambang batas [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 2, 42]
        df_harvest['GagalPanen'] = (df_harvest['z_score'] < config.Z_SCORE_THRESHOLD).astype(int)
        
    df_harvest_proc = df_harvest.copy()

    # === 2. Pra-pemrosesan Data Cuaca (Fitur X) ===
    df_weather_proc = df_weather.rename(columns={config.REGION_COLUMN: "Wilayah"})
    df_weather_proc[config.DATE_COLUMN] = pd.to_datetime(df_weather_proc[config.DATE_COLUMN])
    
    # Normalisasi nama wilayah untuk matching dengan data panen
    def normalize_region_name(name):
        if pd.isna(name):
            return ""
        name = str(name).strip()
        # Hapus prefix umum (dengan spasi setelahnya)
        prefixes = ["Kab. ", "Kabupaten ", "Kota ", "Kotamadya "]
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):].strip()
        # Juga coba tanpa spasi
        prefixes_no_space = ["Kab.", "Kabupaten", "Kota", "Kotamadya"]
        for prefix in prefixes_no_space:
            if name.startswith(prefix) and len(name) > len(prefix):
                name = name[len(prefix):].strip()
        return name
    
    # Normalisasi nama wilayah di data cuaca
    df_weather_proc['Wilayah'] = df_weather_proc['Wilayah'].apply(normalize_region_name)
    
    # One-Hot Encoding untuk 'Cuaca Ekstrem' dan 'Dampak' 
    # Kita menggunakan str.get_dummies untuk menangani string gabungan (misal, "Hujan Lebat, Petir")
    df_weather_events = df_weather_proc[config.WEATHER_EVENT_COLUMN].str.get_dummies(sep=', ')
    df_weather_impacts = df_weather_proc[config.WEATHER_IMPACT_COLUMN].str.get_dummies(sep=' / ')
    
    df_weather_proc = pd.concat([df_weather_proc[['Wilayah', config.DATE_COLUMN]], df_weather_events, df_weather_impacts], axis=1)

    # Agregasi Harian ke Mingguan: Hitung JUMLAH kejadian per minggu
    # Set index ke Tanggal, lalu groupby Wilayah dan resample
    # Simpan Wilayah sebagai kolom terpisah sebelum set_index
    df_weather_proc_indexed = df_weather_proc.set_index(config.DATE_COLUMN)
    
    # Groupby dan resample, pastikan Wilayah tetap ada
    # Gunakan include_groups=False untuk menghindari warning
    df_weather_weekly = (df_weather_proc_indexed
                        .groupby('Wilayah', group_keys=True)
                        .resample(config.TIME_AGGREGATION_RULE, include_groups=False)
                        .sum(numeric_only=True)
                        .reset_index(level=0, drop=False))  # Reset hanya level 0 (Tanggal), keep Wilayah
    
    # Reset index untuk mendapatkan Tanggal sebagai kolom
    df_weather_weekly = df_weather_weekly.reset_index()
    
    df_weather_weekly['Tahun'] = df_weather_weekly[config.DATE_COLUMN].dt.year

    # === 3. Gabungkan Data Panen dan Cuaca ===
    # Gabungkan fitur cuaca mingguan dengan data panen tahunan
    df_merged = pd.merge(
        df_weather_weekly,
        df_harvest_proc,
        on=['Wilayah', 'Tahun'],
        how='left' # Jaga semua data cuaca, cocokkan data panen jika ada
    )
    
    # Isi data panen (LuasPanen, GagalPanen) ke semua minggu dalam tahun itu
    cols_to_fill = ['LuasPanen']
    if is_training and 'GagalPanen' in df_merged.columns:
        cols_to_fill.append('GagalPanen')
    
    # Hanya fill kolom yang ada
    existing_cols_to_fill = [col for col in cols_to_fill if col in df_merged.columns]
    if existing_cols_to_fill:
        df_merged[existing_cols_to_fill] = df_merged.groupby('Wilayah')[existing_cols_to_fill].ffill().bfill()
        df_merged = df_merged.dropna(subset=existing_cols_to_fill) # Hapus minggu tanpa data panen
    
    # === 4. Normalisasi Fitur (X) ===
    # Tentukan fitur (X) dan label (y)
    # Hapus kolom non-fitur
    cols_to_drop = ['Wilayah', config.DATE_COLUMN, 'Tahun', 'Produktivitas', 'Rekap Produksi Padi (ton)']
    if is_training:
        cols_to_drop.append('z_score')
    features_df = df_merged.drop(columns=cols_to_drop, errors='ignore')
    
    # Pastikan semua kolom fitur ada, meskipun tidak ada di data baru
    if scaler and is_training is False:
        # Untuk prediksi: pastikan kolom cocok dengan saat pelatihan
        if hasattr(scaler, 'feature_names_in_'):
            known_features = scaler.feature_names_in_
            features_df = features_df.reindex(columns=known_features, fill_value=0)
    
    if is_training:
        # Buat dan 'fit' scaler HANYA pada data pelatihan
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = scaler.fit_transform(features_df)
    else:
        # 'Transform' data prediksi menggunakan scaler yang sudah ada
        scaled_features = scaler.transform(features_df)
        
    labels = df_merged['GagalPanen'].values if (is_training and 'GagalPanen' in df_merged.columns) else None

    # === 5. Buat Sekuens (Windowing) ===
    print(f"Membuat sekuens (windowing) dengan panjang {config.SEQUENCE_LENGTH}...")
    dataset = timeseries_dataset_from_array(
        data=scaled_features,
        targets=labels if is_training else None,
        sequence_length=config.SEQUENCE_LENGTH,
        sequence_stride=config.SEQUENCE_STRIDE,
        batch_size=config.BATCH_SIZE if is_training else 1, # Batch 1 untuk prediksi
        shuffle=is_training # Acak hanya saat pelatihan
    )
    
    return dataset, scaler, labels if is_training else None