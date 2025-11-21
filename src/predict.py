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
        'weather_forecast': weather_forecast,
        'web_summary': build_web_summary(
            prediction_label='Gagal Panen' if is_failure else 'Normal',
            probability=latest_prediction,
            risk_level=risk_level,
            df_weather=df_weather,
            df_harvest=df_harvest,
            reasons=reasons,
            mitigation=mitigation,
            weather_forecast=weather_forecast
        )
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

def build_web_summary(
    prediction_label: str,
    probability: float,
    risk_level: str,
    df_weather: pd.DataFrame,
    df_harvest: pd.DataFrame,
    reasons: list,
    mitigation: list,
    weather_forecast: dict
) -> dict:
    """
    Membentuk ringkasan deskriptif berbasis data aktual untuk ditampilkan di UI.
    """
    period_label = _get_weather_period_label(df_weather)
    weather_stats = _build_weather_stats(df_weather)
    analysis_content = _build_analysis_content(reasons, df_harvest)
    weather_summary = _build_weather_summary(df_weather)
    conclusion_text = (
        f"Prediksi akhir: {prediction_label} dengan probabilitas "
        f"{probability * 100:.1f}% dan level risiko {risk_level}."
    )
    mitigation_sections = _build_mitigation_sections(mitigation)
    forecast_lines = _build_forecast_lines(weather_forecast)

    return {
        "sections": [
            {
                "title": "Hasil Prediksi AI",
                "content": [
                    f"{prediction_label} (Probabilitas {probability * 100:.1f}%, Risiko {risk_level})"
                ]
            },
            {
                "title": "Analisis & Alasan Prediksi",
                "content": analysis_content
            },
            {
                "title": f"Data Cuaca Historis ({period_label})",
                "content": weather_stats
            },
            {
                "title": "Ringkasan Cuaca",
                "content": [weather_summary]
            },
            {
                "title": "Kesimpulan Detail",
                "content": [conclusion_text]
            },
            {
                "title": "Rekomendasi Mitigasi Penanganan",
                "subsections": mitigation_sections
            },
            {
                "title": "Prakiraan Cuaca BMKG (3 Bulan Kedepan)",
                "content": forecast_lines
            }
        ]
    }

def _get_weather_period_label(df_weather: pd.DataFrame) -> str:
    if df_weather.empty or config.DATE_COLUMN not in df_weather.columns:
        return "Data tidak tersedia"
    start_date = df_weather[config.DATE_COLUMN].min()
    end_date = df_weather[config.DATE_COLUMN].max()
    if pd.isna(start_date) or pd.isna(end_date):
        return "Data tidak tersedia"
    start_label = start_date.strftime("%d %B %Y")
    end_label = end_date.strftime("%d %B %Y")
    if start_label == end_label:
        return start_label
    return f"{start_label} - {end_label}"

def _build_weather_stats(df_weather: pd.DataFrame) -> list:
    stats = []
    stats.append(_format_numeric_metric(df_weather, ["suhu", "temperature"], "Suhu Rata-rata", "°C"))
    stats.append(_format_numeric_metric(df_weather, ["lembap", "humidity"], "Kelembapan", "%"))
    stats.append(_format_numeric_metric(df_weather, ["hujan", "precip"], "Curah Hujan", "mm"))
    stats.append(_format_numeric_metric(df_weather, ["angin", "wind"], "Angin", " km/jam"))
    return stats

def _format_numeric_metric(df: pd.DataFrame, keywords: list, label: str, unit: str) -> str:
    if df.empty:
        return f"{label}: Data tidak tersedia"
    column = _find_column_by_keywords(df.columns, keywords)
    if column:
        series = pd.to_numeric(df[column], errors='coerce').dropna()
        if not series.empty:
            value = series.mean()
            return f"{label}: {value:.1f}{unit}".rstrip()
    return f"{label}: Data tidak tersedia"

def _find_column_by_keywords(columns, keywords):
    for col in columns:
        col_lower = col.lower()
        for key in keywords:
            if key in col_lower:
                return col
    return None

def _build_analysis_content(reasons: list, df_harvest: pd.DataFrame) -> list:
    analysis = reasons.copy() if reasons else []
    if not df_harvest.empty and 'Produktivitas' in df_harvest.columns:
        prod_mean = df_harvest['Produktivitas'].dropna().mean()
        if pd.notna(prod_mean):
            analysis.append(f"Rata-rata produktivitas historis: {prod_mean:.2f} ku/ha.")
    if not analysis:
        analysis.append("Tidak ada analisis tambahan yang tersedia.")
    return analysis

def _build_weather_summary(df_weather: pd.DataFrame) -> str:
    if df_weather.empty or config.WEATHER_EVENT_COLUMN not in df_weather.columns:
        return "Data cuaca ekstrem tidak tersedia."
    events = (
        df_weather[config.WEATHER_EVENT_COLUMN]
        .dropna()
        .str.split(', ', expand=True)
        .stack()
    )
    if events.empty:
        return "Tidak ada catatan cuaca ekstrem pada periode ini."
    counts = events.value_counts()
    top_event = counts.index[0]
    top_count = counts.iloc[0]
    return f"Kejadian paling sering: {top_event} ({top_count} kali) selama periode historis."

def _build_mitigation_sections(mitigation: list) -> list:
    if not mitigation:
        return [{"subtitle": "Umum", "content": ["Tidak ada rekomendasi yang tersedia."]}]
    return [{"subtitle": "Mitigasi Prioritas", "content": mitigation}]

def _build_forecast_lines(weather_forecast: dict) -> list:
    if not weather_forecast or 'forecast' not in weather_forecast:
        return ["Data prakiraan belum tersedia."]
    lines = []
    for item in weather_forecast.get('forecast', []):
        name = item.get('nama_bulan', 'Periode')
        events = item.get('prediksi_kejadian', 0)
        notes = item.get('catatan', '')
        extreme = item.get('cuaca_ekstrem', {})
        if extreme:
            dominant = ', '.join(list(extreme.keys())[:2])
            lines.append(f"{name}: {events} potensi kejadian ekstrem (Dominan: {dominant}). {notes}")
        else:
            lines.append(f"{name}: {events} potensi kejadian ekstrem. {notes}")
    if not lines:
        lines.append(weather_forecast.get('note', 'Data prakiraan belum tersedia.'))
    return lines

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

