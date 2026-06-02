import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_seed(42)
print('TensorFlow version:', tf.__version__)


def make_dataset(n_samples: int = 7000, seq_len: int = 16):
    x = np.random.choice([-1.0, 1.0], size=(n_samples, seq_len, 1)).astype(np.float32)
    cumsum = x.cumsum(axis=1)
    y = (cumsum > 0).astype(np.float32)
    return x, y


X, y = make_dataset()
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print('Train:', X_train.shape, y_train.shape)
print('Val  :', X_val.shape, y_val.shape)
print('Test :', X_test.shape, y_test.shape)

# Mini-check: данные и диапазон меток
assert X_train.ndim == 3 and X_train.shape[2] == 1
assert y_train.ndim == 3 and y_train.shape[2] == 1
assert set(np.unique(y_train)).issubset({0.0, 1.0})
print('Mini-check данных пройден.')

def build_model(input_shape):
    inputs = tf.keras.layers.Input(shape=input_shape)
    # TODO 3: добавьте LSTM с return_sequences=True
    x = tf.keras.layers.LSTM(units=64, return_sequences=True)(inputs)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)
    # TODO 4: скомпилируйте модель (adam, binary_crossentropy, accuracy)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return model


model = build_model(X_train.shape[1:])
model.summary()

# Mini-check: выход должен быть последовательностью
assert model.output_shape[-1] == 1
assert len(model.output_shape) == 3, f'Ожидался 3D-выход, получили {model.output_shape}'
print('Mini-check модели пройден.')


def train_model(model, X_train, y_train, X_val, y_val, epochs: int = 10):
    # TODO 5: обучите модель с validation_data и batch_size=64
    history = model.fit(X_train, y_train,
                        batch_size=64,
                        epochs=epochs,
                        validation_data=(X_val, y_val))
    return history


history = train_model(model, X_train, y_train, X_val, y_val)

# Mini-check: история обучения
for key in ['loss', 'val_loss', 'accuracy', 'val_accuracy']:
    assert key in history.history, f'В history нет ключа {key}'
print('Mini-check обучения пройден.')

def evaluate_model(model, X_test, y_test):
    loss, token_acc = model.evaluate(X_test, y_test, verbose=0)
    probs = model.predict(X_test, verbose=0)
    preds = (probs >= 0.5).astype(int)
    seq_acc = np.mean(np.all(y_test == preds, axis=1))
    return {
        'loss': float(loss),
        'token_accuracy': float(token_acc),
        'sequence_accuracy': float(seq_acc),
        'preds': preds,
        'probs': probs,
    }


metrics = evaluate_model(model, X_test, y_test)
print({k: v for k, v in metrics.items() if k in ('loss', 'token_accuracy', 'sequence_accuracy')})

# Mini-check: порог качества
assert metrics['preds'].shape == y_test.shape
if metrics['token_accuracy'] >= 0.90:
    print('Целевой порог достигнут: token_accuracy >= 0.90')
else:
    print('Порог не достигнут. Проверьте шаги генерации данных и return_sequences.')

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title('Token Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

idx = 0
print('x      :', X_test[idx, :, 0].astype(int))
print('target :', y_test[idx, :, 0].astype(int))
print('pred   :', metrics['preds'][idx, :, 0].astype(int))
