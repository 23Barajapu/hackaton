"""
Skrip untuk melatih model GRU untuk prediksi gagal panen.
Mendukung hyperparameter tuning dengan KerasTuner dan pelatihan final.
"""
import os
import json
import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras_tuner import HyperModel, RandomSearch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import data_processing as dp
import model
import config

# Set seed untuk reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class GRUHyperModel(HyperModel):
    """Wrapper untuk model GRU yang dapat dituning dengan KerasTuner."""
    def __init__(self, input_shape):
        self.input_shape = input_shape
    
    def build(self, hp):
        return model.build_model(self.input_shape, hp)

def train_model(use_tuner=False, max_trials=10):
    """
    Melatih model GRU untuk prediksi gagal panen.
    
    Args:
        use_tuner: Jika True, gunakan KerasTuner untuk mencari hyperparameter terbaik
        max_trials: Jumlah trial untuk hyperparameter tuning
    """
    print("=" * 60)
    print("MEMULAI PELATIHAN MODEL")
    print("=" * 60)
    
    # === 1. Muat dan Preprocess Data ===
    print("\n[1/5] Memuat data...")
    # Gunakan fungsi load_data_from_csv untuk development
    # Atau load_training_data() untuk production dari Supabase
    df_harvest, df_weather = dp.load_data_from_csv()
    
    if df_harvest.empty or df_weather.empty:
        print("ERROR: Data kosong! Pastikan file CSV ada di folder data/")
        return
    
    print("\n[2/5] Memproses fitur...")
    dataset, scaler, labels = dp.preprocess_features(
        df_harvest, 
        df_weather, 
        scaler=None, 
        is_training=True
    )
    
    # Dapatkan input shape dari dataset
    sample_batch = next(iter(dataset))
    input_shape = (sample_batch[0].shape[1], sample_batch[0].shape[2])  # (sequence_length, n_features)
    print(f"Input shape: {input_shape}")
    
    # === 2. Split Data ===
    print("\n[3/5] Membagi data train/validation...")
    dataset_size = len(list(dataset))
    val_size = int(dataset_size * config.VALIDATION_SPLIT)
    train_size = dataset_size - val_size
    
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)
    
    print(f"Train samples: {train_size}")
    print(f"Validation samples: {val_size}")
    
    # === 3. Hyperparameter Tuning (Opsional) ===
    if use_tuner:
        print("\n[4/5] Menjalankan Hyperparameter Tuning...")
        hypermodel = GRUHyperModel(input_shape)
        
        tuner = RandomSearch(
            hypermodel,
            objective=config.TUNER_OBJECTIVE,
            max_trials=max_trials,
            executions_per_trial=1,
            directory='tuner_results',
            project_name='gru_harvest_failure'
        )
        
        tuner.search(
            train_dataset,
            validation_data=val_dataset,
            epochs=config.EPOCHS,
            verbose=1
        )
        
        # Dapatkan model terbaik
        best_hp = tuner.get_best_hyperparameters()[0]
        print("\nHyperparameter terbaik:")
        print(f"  units_1: {best_hp.get('units_1')}")
        print(f"  dropout_1: {best_hp.get('dropout_1')}")
        print(f"  units_2: {best_hp.get('units_2')}")
        print(f"  dropout_2: {best_hp.get('dropout_2')}")
        print(f"  learning_rate: {best_hp.get('learning_rate')}")
        
        final_model = tuner.get_best_models()[0]
    else:
        print("\n[4/5] Membangun model dengan hyperparameter default...")
        # Gunakan hyperparameter default
        hp = None
        final_model = model.build_model(input_shape, hp)
    
    # === 4. Pelatihan Final ===
    print("\n[5/5] Melatih model final...")
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            config.MODEL_PATH,
            monitor='val_recall',
            save_best_only=True,
            mode='max'
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    history = final_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # === 5. Evaluasi dan Threshold Optimization ===
    print("\nEvaluasi model...")
    val_predictions = final_model.predict(val_dataset)
    val_labels = np.concatenate([y for x, y in val_dataset], axis=0)
    
    # Cari threshold optimal berdasarkan F1-score
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in np.arange(0.3, 0.8, 0.05):
        pred_binary = (val_predictions >= threshold).astype(int)
        from sklearn.metrics import f1_score
        f1 = f1_score(val_labels, pred_binary)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\nThreshold optimal: {best_threshold:.3f} (F1-score: {best_f1:.3f})")
    
    # Evaluasi dengan threshold optimal
    final_predictions = (val_predictions >= best_threshold).astype(int)
    print("\nClassification Report:")
    print(classification_report(val_labels, final_predictions))
    print("\nConfusion Matrix:")
    print(confusion_matrix(val_labels, final_predictions))
    
    # === 6. Simpan Artefak ===
    print("\nMenyimpan model dan artefak...")
    os.makedirs('models', exist_ok=True)
    
    # Simpan model
    final_model.save(config.MODEL_PATH)
    print(f"Model disimpan di: {config.MODEL_PATH}")
    
    # Simpan scaler
    joblib.dump(scaler, config.SCALER_PATH)
    print(f"Scaler disimpan di: {config.SCALER_PATH}")
    
    # Simpan config (termasuk threshold optimal)
    model_config = {
        'optimal_threshold': float(best_threshold),
        'input_shape': input_shape,
        'sequence_length': config.SEQUENCE_LENGTH,
        'validation_f1': float(best_f1)
    }
    with open(config.CONFIG_PATH, 'w') as f:
        json.dump(model_config, f, indent=2)
    print(f"Config disimpan di: {config.CONFIG_PATH}")
    
    print("\n" + "=" * 60)
    print("PELATIHAN SELESAI!")
    print("=" * 60)
    
    return final_model, history, best_threshold

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train GRU model for harvest failure prediction')
    parser.add_argument('--tune', action='store_true', help='Run hyperparameter tuning')
    parser.add_argument('--trials', type=int, default=10, help='Number of tuning trials')
    
    args = parser.parse_args()
    
    train_model(use_tuner=args.tune, max_trials=args.trials)

