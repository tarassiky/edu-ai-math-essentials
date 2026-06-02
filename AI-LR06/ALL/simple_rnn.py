import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_seed(42)
print('TensorFlow version:', tf.__version__)

def make_dataset(n_samples: int = 8000, seq_len: int = 12):
    x = np.random.choice([-1.0, 1.0], size=(n_samples, seq_len, 1)).astype(np.float32)
    sums = np.sum(x, axis=(1, 2)) 
    y = (sums > 0).astype(np.int32) 
    return x, y


X, y = make_dataset()
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print('Train:', X_train.shape, y_train.shape)
print('Val  :', X_val.shape, y_val.shape)
print('Test :', X_test.shape, y_test.shape)

# Mini-check: корректность данных
assert X_train.ndim == 3 and X_train.shape[2] == 1, 'Ожидается форма (N, T, 1)'
assert y_train.ndim == 1, 'Ожидается форма меток (N,)'
assert set(np.unique(y_train)).issubset({0.0, 1.0}), 'Метки должны быть бинарными'
print('Mini-check данных пройден.')

def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.SimpleRNN(32),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


model = build_model(input_shape=X_train.shape[1:])
model.summary()

# Mini-check: корректность выходной формы
assert model.output_shape == (None, 1), f'Неожиданный output_shape: {model.output_shape}'
print('Mini-check модели пройден.')

def train_model(model, X_train, y_train, X_val, y_val, epochs: int = 4):
    # TODO 5: обучите модель с validation_data, epochs и batch_size=64
    history = model.fit(X_train, y_train,
                        batch_size=64,
                        epochs=epochs,
                        validation_data=(X_val, y_val))
    return history


history = train_model(model, X_train, y_train, X_val, y_val)

# Mini-check: история обучения содержит ключевые метрики
for key in ['loss', 'val_loss', 'accuracy', 'val_accuracy']:
    assert key in history.history, f'В history нет ключа {key}'
print('Mini-check обучения пройден.')

def evaluate_model(model, X_test, y_test):
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    probs = model.predict(X_test, verbose=0).reshape(-1)
    preds = (probs >= 0.5).astype(int)
    return {
        'loss': float(loss),
        'accuracy': float(acc),
        'preds': preds,
        'probs': probs,
    }


metrics = evaluate_model(model, X_test, y_test)
print({k: v for k, v in metrics.items() if k in ('loss', 'accuracy')})

# Mini-check: качество и корректность форм
assert metrics['probs'].shape[0] == X_test.shape[0]
assert metrics['preds'].shape[0] == X_test.shape[0]
assert set(np.unique(metrics['preds'])).issubset({0, 1})

if metrics['accuracy'] >= 0.90:
    print('Целевой порог достигнут: accuracy >= 0.90')
else:
    print('Порог не достигнут. Рекомендация: проверьте TODO и mini-check выше.')

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

for i in range(5):
    print(f'sample={i:02d} true={int(y_test[i])} prob={metrics["probs"][i]:.3f} pred={int(metrics["preds"][i])}')
