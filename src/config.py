import numpy as np
import tensorflow as tf

# --- Variabel Target & Kunci ---
# Nama kolom berdasarkan file CSV Anda 
TARGET_COLUMN = "Produktivitas Tanaman Padi (ku/ha) (Ku/ha)"
REGION_COLUMN = "Kabupaten/Kota"
DATE_COLUMN = "Tanggal"
WEATHER_EVENT_COLUMN = "Cuaca Ekstrem"
WEATHER_IMPACT_COLUMN = "Dampak"

# --- Parameter Pra-pemrosesan ---
# Menggunakan ambang batas Z-score yang umum untuk anomali "parah"
# (Z < -1.5) [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 2, 42]
Z_SCORE_THRESHOLD = -1.5

# Agregasi data cuaca harian menjadi mingguan
TIME_AGGREGATION_RULE = 'W'

# Panjang sekuens (misal, 12 minggu) untuk input GRU
SEQUENCE_LENGTH = 12
# Seberapa banyak menggeser jendela (1 minggu)
SEQUENCE_STRIDE = 1
# Ukuran Batch
BATCH_SIZE = 32

# --- Parameter Pelatihan ---
# 20% dari data akan digunakan untuk validasi
VALIDATION_SPLIT = 0.2
EPOCHS = 50
# Tentukan metrik yang paling penting untuk peringatan dini: Recall [43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 39, 4, 21, 61, 62]
TUNER_OBJECTIVE = tf.keras.metrics.Recall(name="val_recall")
# Jika menggunakan string:
# TUNER_OBJECTIVE = kt.Objective("val_recall", direction="max") [63, 64, 65, 50, 51, 66, 67, 68, 69, 70, 21]

# Ambang batas probabilitas (0.0 - 1.0) untuk klasifikasi akhir.
# Nilai default 0.5 di-override oleh skrip train.py.
OPTIMAL_THRESHOLD = 0.5

# --- Path Penyimpanan Artefak ---
MODEL_PATH = "models/gru_model.keras"
SCALER_PATH = "models/feature_scaler.joblib"
CONFIG_PATH = "models/model_config.json" # Untuk menyimpan threshold