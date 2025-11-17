"""
Skrip untuk melakukan prediksi gagal panen menggunakan model yang sudah dilatih.
"""
import os
import json
import joblib
import numpy as np
import tensorflow as tf
import pandas as pd
import data_processing as dp
import config

def load_model_and_artifacts():
    """Memuat model, scaler, dan config yang sudah dilatih."""
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(f"Model tidak ditemukan di {config.MODEL_PATH}. Jalankan train.py terlebih dahulu.")
    
    model = tf.keras.models.load_model(config.MODEL_PATH)
    scaler = joblib.load(config.SCALER_PATH)
    
    with open(config.CONFIG_PATH, 'r') as f:
        model_config = json.load(f)
    
    return model, scaler, model_config

def predict_harvest_failure(region_name: str, start_date: str = None, use_csv: bool = True):
    """
    Memprediksi kemungkinan gagal panen untuk suatu wilayah.
    
    Args:
        region_name: Nama kabupaten/kota
        start_date: Tanggal mulai untuk data cuaca (format: 'YYYY-MM-DD')
        use_csv: Jika True, gunakan data CSV lokal. Jika False, gunakan Supabase API
    
    Returns:
        dict: Hasil prediksi dengan probabilitas dan klasifikasi
    """
    print(f"Memuat model dan artefak...")
    model, scaler, model_config = load_model_and_artifacts()
    threshold = model_config.get('optimal_threshold', 0.5)
    
    print(f"Memprediksi untuk wilayah: {region_name}")
    
    # Muat data prediksi
    if use_csv:
        # Untuk development: filter dari CSV
        df_harvest, df_weather = dp.load_data_from_csv()
        
        # Filter data panen
        df_harvest = df_harvest[df_harvest[config.REGION_COLUMN] == region_name]
        
        # Normalisasi nama wilayah untuk matching yang lebih fleksibel
        # Hapus "Kab.", "Kota", dll untuk matching
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
        
        # Filter data cuaca dengan matching fleksibel
        df_weather_normalized = df_weather.copy()
        df_weather_normalized['_normalized_region'] = df_weather[config.REGION_COLUMN].apply(normalize_region_name)
        region_normalized = normalize_region_name(region_name)
        
        # Coba exact match dulu
        weather_mask = df_weather[config.REGION_COLUMN] == region_name
        # Jika tidak ada, coba dengan normalized name
        if not weather_mask.any():
            weather_mask = df_weather_normalized['_normalized_region'] == region_normalized
        # Jika masih tidak ada, coba contains (region_name di dalam cuaca region)
        if not weather_mask.any():
            weather_mask = df_weather[config.REGION_COLUMN].str.contains(region_name, case=False, na=False)
        # Jika masih tidak ada, coba reverse contains (cuaca region di dalam region_name)
        if not weather_mask.any():
            weather_mask = df_weather_normalized['_normalized_region'].str.contains(region_normalized, case=False, na=False)
        # Jika masih tidak ada, coba reverse - apakah region_name mengandung normalized cuaca region
        if not weather_mask.any():
            # Cek apakah ada normalized cuaca region yang ada di region_name
            for idx, norm_cuaca in enumerate(df_weather_normalized['_normalized_region']):
                if region_normalized and norm_cuaca and (region_normalized in norm_cuaca or norm_cuaca in region_normalized):
                    weather_mask.iloc[idx] = True
        
        if start_date:
            df_weather[config.DATE_COLUMN] = pd.to_datetime(df_weather[config.DATE_COLUMN])
            df_weather = df_weather[weather_mask & (df_weather[config.DATE_COLUMN] >= start_date)]
        else:
            df_weather = df_weather[weather_mask]
        
    else:
        # Untuk production: ambil dari Supabase
        if not start_date:
            # Default: 12 minggu terakhir
            from datetime import datetime, timedelta
            start_date = (datetime.now() - timedelta(weeks=12)).strftime('%Y-%m-%d')
        
        df_harvest, df_weather = dp.load_prediction_data(region_name, start_date)
    
    if df_harvest.empty or df_weather.empty:
        return {
            'error': f'Data tidak ditemukan untuk wilayah {region_name}. Panen: {len(df_harvest)} baris, Cuaca: {len(df_weather)} baris',
            'region': region_name
        }
    
    # Preprocess data
    print("Memproses data...")
    dataset, _, _ = dp.preprocess_features(
        df_harvest,
        df_weather,
        scaler=scaler,
        is_training=False
    )
    
    # Prediksi
    print("Menjalankan prediksi...")
    predictions = model.predict(dataset, verbose=0)
    
    # Ambil prediksi terakhir (paling recent)
    latest_prediction = float(predictions[-1][0])
    is_failure = latest_prediction >= threshold
    
    # Interpretasi
    risk_level = "Tinggi" if latest_prediction >= 0.7 else "Sedang" if latest_prediction >= threshold else "Rendah"
    
    # Import modul rekomendasi
    import recommendations as rec
    
    # Dapatkan alasan dan rekomendasi
    if is_failure:
        reasons = rec.get_failure_reasons(latest_prediction, df_weather, df_harvest)
    else:
        reasons = rec.get_success_reasons(latest_prediction, df_weather, df_harvest)
    
    mitigation = rec.get_mitigation_recommendations(latest_prediction, risk_level, df_weather)
    weather_forecast = rec.get_weather_forecast(df_weather, months=3)
    
    result = {
        'region': region_name,
        'probability': round(latest_prediction, 4),
        'threshold': threshold,
        'prediction': 'Gagal Panen' if is_failure else 'Normal',
        'risk_level': risk_level,
        'confidence': 'Tinggi' if abs(latest_prediction - threshold) > 0.2 else 'Sedang',
        'reasons': reasons,
        'mitigation_recommendations': mitigation,
        'weather_forecast': weather_forecast
    }
    
    return result

def predict_batch(regions: list, use_csv: bool = True):
    """
    Memprediksi untuk beberapa wilayah sekaligus.
    
    Args:
        regions: List nama kabupaten/kota
        use_csv: Jika True, gunakan data CSV lokal
    
    Returns:
        list: List hasil prediksi untuk setiap wilayah
    """
    results = []
    for region in regions:
        try:
            result = predict_harvest_failure(region, use_csv=use_csv)
            results.append(result)
        except Exception as e:
            results.append({
                'region': region,
                'error': str(e)
            })
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict harvest failure')
    parser.add_argument('--region', type=str, required=True, help='Nama kabupaten/kota')
    parser.add_argument('--start-date', type=str, help='Tanggal mulai (YYYY-MM-DD)')
    parser.add_argument('--csv', action='store_true', help='Gunakan data CSV lokal')
    
    args = parser.parse_args()
    
    result = predict_harvest_failure(
        args.region,
        start_date=args.start_date,
        use_csv=args.csv
    )
    
    print("\n" + "=" * 60)
    print("HASIL PREDIKSI")
    print("=" * 60)
    print(json.dumps(result, indent=2, ensure_ascii=False))

