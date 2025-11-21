"""
Modul untuk memberikan rekomendasi dan analisis berdasarkan prediksi gagal panen.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import data_processing as dp
import config

def get_failure_reasons(probability: float, df_weather: pd.DataFrame, df_harvest: pd.DataFrame) -> list:
    """
    Mendapatkan alasan-alasan potensial gagal panen berdasarkan probabilitas dan data cuaca.
    
    Args:
        probability: Probabilitas gagal panen (0-1)
        df_weather: DataFrame data cuaca
        df_harvest: DataFrame data panen
    
    Returns:
        list: Daftar alasan potensial
    """
    reasons = []
    
    if probability >= 0.7:
        reasons.append("Probabilitas gagal panen sangat tinggi (>70%)")
    elif probability >= 0.5:
        reasons.append("Probabilitas gagal panen tinggi (>50%)")
    
    # Analisis data cuaca ekstrem
    if not df_weather.empty:
        # Hitung frekuensi cuaca ekstrem
        if config.WEATHER_EVENT_COLUMN in df_weather.columns:
            weather_events = df_weather[config.WEATHER_EVENT_COLUMN].str.split(', ', expand=True).stack()
            event_counts = weather_events.value_counts()
            
            if 'Hujan Lebat' in event_counts.index and event_counts['Hujan Lebat'] > 10:
                reasons.append(f"Frekuesi hujan lebat tinggi ({event_counts['Hujan Lebat']} kejadian)")
            
            if 'Angin Kencang' in event_counts.index and event_counts['Angin Kencang'] > 5:
                reasons.append(f"Frekuesi angin kencang tinggi ({event_counts['Angin Kencang']} kejadian)")
            
            if 'Puting Beliung' in event_counts.index and event_counts['Puting Beliung'] > 3:
                reasons.append(f"Frekuesi puting beliung tinggi ({event_counts['Puting Beliung']} kejadian)")
        
        # Analisis dampak
        if config.WEATHER_IMPACT_COLUMN in df_weather.columns:
            impacts = df_weather[config.WEATHER_IMPACT_COLUMN].str.split(' / ', expand=True).stack()
            impact_counts = impacts.value_counts()
            
            if 'Banjir / Genangan' in impact_counts.index and impact_counts['Banjir / Genangan'] > 5:
                reasons.append(f"Frekuesi banjir/genangan tinggi ({impact_counts['Banjir / Genangan']} kejadian)")
            
            if 'Tanah Longsor' in impact_counts.index and impact_counts['Tanah Longsor'] > 2:
                reasons.append(f"Frekuesi tanah longsor ({impact_counts['Tanah Longsor']} kejadian)")
    
    # Analisis produktivitas historis
    if not df_harvest.empty and 'Produktivitas' in df_harvest.columns:
        avg_productivity = df_harvest['Produktivitas'].mean()
        if avg_productivity < 50:
            reasons.append(f"Produktivitas historis rendah (rata-rata {avg_productivity:.2f} ku/ha)")
    
    if not reasons:
        reasons.append("Tidak ada indikasi masalah signifikan berdasarkan data yang tersedia")
    
    return reasons

def get_success_reasons(probability: float, df_weather: pd.DataFrame, df_harvest: pd.DataFrame) -> list:
    """
    Mendapatkan alasan-alasan keberhasilan panen.
    
    Args:
        probability: Probabilitas gagal panen (0-1)
        df_weather: DataFrame data cuaca
        df_harvest: DataFrame data panen
    
    Returns:
        list: Daftar alasan keberhasilan
    """
    reasons = []
    
    if probability < 0.3:
        reasons.append("Probabilitas gagal panen sangat rendah (<30%)")
    
    # Analisis kondisi cuaca
    if not df_weather.empty:
        if config.WEATHER_EVENT_COLUMN in df_weather.columns:
            weather_events = df_weather[config.WEATHER_EVENT_COLUMN].str.split(', ', expand=True).stack()
            event_counts = weather_events.value_counts()
            
            if 'Hujan Lebat' in event_counts.index and event_counts['Hujan Lebat'] < 5:
                reasons.append("Frekuesi cuaca ekstrem rendah")
    
    # Analisis produktivitas
    if not df_harvest.empty and 'Produktivitas' in df_harvest.columns:
        avg_productivity = df_harvest['Produktivitas'].mean()
        if avg_productivity >= 55:
            reasons.append(f"Produktivitas historis baik (rata-rata {avg_productivity:.2f} ku/ha)")
    
    if not reasons:
        reasons.append("Kondisi normal berdasarkan data yang tersedia")
    
    return reasons

def get_mitigation_recommendations(probability: float, risk_level: str, df_weather: pd.DataFrame) -> list:
    """
    Mendapatkan rekomendasi mitigasi berdasarkan risiko gagal panen.
    
    Args:
        probability: Probabilitas gagal panen
        risk_level: Level risiko (Tinggi/Sedang/Rendah)
        df_weather: DataFrame data cuaca
    
    Returns:
        list: Daftar rekomendasi mitigasi
    """
    recommendations = []
    
    if risk_level == "Tinggi" or probability >= 0.7:
        recommendations.append("[ALERT] TINDAKAN SEGERA DIPERLUKAN")
        recommendations.append("- Tingkatkan monitoring lahan secara intensif (setiap 2-3 hari)")
        recommendations.append("- Siapkan sistem drainase darurat untuk mengatasi genangan air")
        recommendations.append("- Pertimbangkan memanen lebih awal jika tanaman sudah cukup matang")
        recommendations.append("- Hubungi dinas pertanian setempat untuk bantuan teknis")
        recommendations.append("- Pertimbangkan asuransi pertanian untuk melindungi kerugian")
    
    if risk_level == "Sedang" or (probability >= 0.5 and probability < 0.7):
        recommendations.append("[NOTICE] PERHATIAN KHUSUS DIPERLUKAN")
        recommendations.append("- Monitor kondisi lahan secara rutin (setiap minggu)")
        recommendations.append("- Pastikan sistem irigasi dan drainase berfungsi dengan baik")
        recommendations.append("- Lakukan pemupukan sesuai jadwal dan dosis yang tepat")
        recommendations.append("- Waspada terhadap hama dan penyakit tanaman")
        recommendations.append("- Siapkan rencana cadangan jika kondisi memburuk")
    
    # Rekomendasi berdasarkan cuaca ekstrem
    if not df_weather.empty and config.WEATHER_EVENT_COLUMN in df_weather.columns:
        weather_events = df_weather[config.WEATHER_EVENT_COLUMN].str.split(', ', expand=True).stack()
        event_counts = weather_events.value_counts()
        
        if 'Hujan Lebat' in event_counts.index and event_counts['Hujan Lebat'] > 5:
            recommendations.append("- Perbaiki saluran drainase untuk mengatasi hujan lebat")
            recommendations.append("- Hindari penanaman di area yang rawan banjir")
            recommendations.append("- Gunakan varietas padi yang tahan genangan")
        
        if 'Angin Kencang' in event_counts.index and event_counts['Angin Kencang'] > 3:
            recommendations.append("- Tanam tanaman pelindung di sekeliling lahan")
            recommendations.append("- Perkuat struktur penopang tanaman jika diperlukan")
        
        if 'Puting Beliung' in event_counts.index:
            recommendations.append("- Waspada terhadap puting beliung, siapkan rencana evakuasi")
    
    # Rekomendasi umum perawatan lahan
    recommendations.append("[GUIDE] PERAWATAN RUTIN LAHAN PADI:")
    recommendations.append("- Lakukan penyiangan gulma secara berkala")
    recommendations.append("- Pantau ketersediaan air irigasi")
    recommendations.append("- Lakukan pemupukan berimbang (N, P, K)")
    recommendations.append("- Kontrol hama dan penyakit dengan pestisida yang tepat")
    recommendations.append("- Gunakan benih berkualitas dan varietas yang sesuai")
    recommendations.append("- Lakukan rotasi tanaman untuk menjaga kesuburan tanah")
    recommendations.append("- Terapkan sistem tanam jajar legowo untuk hasil optimal")
    
    return recommendations

def get_weather_forecast(df_weather: pd.DataFrame, months: int = 3) -> dict:
    """
    Membuat prediksi cuaca untuk 3 bulan ke depan berdasarkan pola historis.
    
    Args:
        df_weather: DataFrame data cuaca historis
        months: Jumlah bulan ke depan untuk diprediksi (default: 3)
    
    Returns:
        dict: Prediksi cuaca per bulan
    """
    if df_weather.empty or config.DATE_COLUMN not in df_weather.columns:
        return {
            "forecast": [],
            "note": "Data cuaca tidak tersedia untuk prediksi"
        }
    
    df_weather_copy = df_weather.copy()
    df_weather_copy[config.DATE_COLUMN] = pd.to_datetime(df_weather_copy[config.DATE_COLUMN])
    df_weather_copy['Bulan'] = df_weather_copy[config.DATE_COLUMN].dt.month
    df_weather_copy['Tahun'] = df_weather_copy[config.DATE_COLUMN].dt.year
    
    # Hitung rata-rata kejadian cuaca ekstrem per bulan
    monthly_stats = []
    current_date = datetime.now()
    
    for i in range(months):
        target_month = (current_date.month + i - 1) % 12 + 1
        target_year = current_date.year + ((current_date.month + i - 1) // 12)
        
        # Ambil data historis untuk bulan yang sama
        historical_data = df_weather_copy[df_weather_copy['Bulan'] == target_month]
        
        if not historical_data.empty:
            # Hitung statistik
            total_events = len(historical_data)
            
            # Hitung jenis cuaca ekstrem
            if config.WEATHER_EVENT_COLUMN in historical_data.columns:
                events = historical_data[config.WEATHER_EVENT_COLUMN].str.split(', ', expand=True).stack()
                event_counts = events.value_counts().to_dict()
            else:
                event_counts = {}
            
            monthly_stats.append({
                "bulan": target_month,
                "tahun": target_year,
                "nama_bulan": current_date.replace(month=target_month, year=target_year).strftime("%B %Y"),
                "prediksi_kejadian": total_events,
                "cuaca_ekstrem": event_counts,
                "catatan": f"Berdasarkan rata-rata {len(historical_data)} kejadian historis di bulan {target_month}"
            })
        else:
            monthly_stats.append({
                "bulan": target_month,
                "tahun": target_year,
                "nama_bulan": current_date.replace(month=target_month, year=target_year).strftime("%B %Y"),
                "prediksi_kejadian": 0,
                "cuaca_ekstrem": {},
                "catatan": "Tidak ada data historis untuk bulan ini"
            })
    
    return {
        "forecast": monthly_stats,
        "note": "Prediksi berdasarkan pola historis data cuaca ekstrem. Prediksi ini bersifat estimasi dan dapat berubah."
    }

