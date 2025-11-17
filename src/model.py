import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from keras_tuner import HyperParameters

def build_model(input_shape, hp=None):
    """Membangun arsitektur model GRU yang dapat dituning."""
    
    # Gunakan HyperParameters default jika tidak disediakan (untuk pelatihan akhir)
    if hp is None:
        hp = HyperParameters()
        hp.Fixed('units_1', 64)
        hp.Fixed('dropout_1', 0.2)
        hp.Fixed('units_2', 32)
        hp.Fixed('dropout_2', 0.2)
        hp.Fixed('learning_rate', 1e-3)

    model = Sequential()
    
    # Input Layer
    model.add(Input(shape=input_shape))
    
    # GRU Layer 1
    model.add(GRU(
        units=hp.Int('units_1', min_value=32, max_value=128, step=32),
        return_sequences=True  # True karena kita menumpuk lapisan GRU lain
    ))
    model.add(Dropout(hp.Float('dropout_1', 0.1, 0.4, step=0.1)))
    
    # GRU Layer 2
    model.add(GRU(
        units=hp.Int('units_2', min_value=16, max_value=64, step=16),
        return_sequences=False # False karena lapisan berikutnya adalah Dense
    ))
    model.add(Dropout(hp.Float('dropout_2', 0.1, 0.4, step=0.1)))
    
    # Output Layer
    # 'sigmoid' untuk klasifikasi biner (GagalPanen ya/tidak)
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile Model
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        ),
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

def build_tuner_model(hp):
    """Fungsi wrapper untuk KerasTuner."""
    raise NotImplementedError("Gunakan kelas HyperModel di train.py untuk input_shape dinamis")