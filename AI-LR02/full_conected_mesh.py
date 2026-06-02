# Последовательная модель НС
from tensorflow.keras.models import Sequential
# Основные слои
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
# Утилиты для to_categorical()
from tensorflow.keras import utils
# Алгоритмы оптимизации
from tensorflow.keras.optimizers import Adam
# Отрисовка графиков
import matplotlib.pyplot as plt
# Разделение данных
from sklearn.model_selection import train_test_split
# Для загрузки датасета
from sklearn.datasets import load_wine
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout

# === Загрузка и описание данных ===
x_data = load_wine()['data']
y_data = load_wine()['target']

print('Размерность x_data -', x_data.shape)
print('Размерность y_data -', y_data.shape)
print()

print('Данные по первому вину:', x_data[0])
print('Класс вина:', y_data[0])

# === Подготовка данных ===
y_data = utils.to_categorical(y_data, 3)

# Разбиение данных (условия менять нельзя!)
x_all, x_test, y_all, y_test = train_test_split(
    x_data, y_data,
    test_size=0.1,
    shuffle=True,
    random_state=6
)

x_train, x_val, y_train, y_val = train_test_split(
    x_all, y_all,
    test_size=0.1,
    shuffle=True,
    random_state=6
)

# === Нормализация ===
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

# === Создание модели ===
model = Sequential([
    Input(shape=(13,)),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])


# Компиляция
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Информация о модели
model.summary()

# === Обучение модели ===
history = model.fit(
    x_train, y_train,
    epochs=100,
    batch_size=8,
    validation_data=(x_val, y_val),
    verbose=1
)

# === Оценка ===
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f'\nТочность на тестовой выборке: {test_acc * 100:.2f}%')

# === Графики ===
plt.figure(figsize=(10,4))
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title('Точность обучения')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10,4))
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Функция потерь')
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
