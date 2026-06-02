# Отчет по лабораторной работе

## Тема: Классификация данных с использованием нейронных сетей (Dataset: Wine)

### 1. Цель работы

Изучить процесс построения, обучения и оценки модели нейронной сети для задачи многоклассовой классификации на примере стандартного датасета Wine из библиотеки sklearn. Закрепить практические навыки предобработки данных, разделения выборки, построения и настройки архитектуры нейронной сети с использованием библиотеки TensorFlow Keras.

### 2. Описание данных

В работе используется датасет **Wine**, содержащий химические характеристики различных сортов вина. Датасет находится в модуле `sklearn.datasets` и загружается при помощи функции `load_wine()`.

* Количество объектов: 178
* Количество признаков: 13 (вещественные значения)
* Количество классов: 3 (типы вина)

Каждая строка данных представляет собой измерения химических свойств одного образца вина.

### 3. Подготовка данных

Для подготовки данных выполнены следующие этапы:

1. Загрузка исходных данных `x_data` и меток классов `y_data`.
2. Преобразование меток классов к виду **One-Hot Encoding** при помощи `utils.to_categorical()`.
3. Разделение данных на три выборки согласно шаблону:

   * **Обучающая выборка** (81%)
   * **Валидационная выборка** (9%)
   * **Тестовая выборка** (10%)

   Разделение выполнено с использованием `train_test_split()` и фиксированного параметра `random_state=6`.
4. Нормализация признаков с использованием `StandardScaler()` из `sklearn.preprocessing`.

### 4. Архитектура модели

Модель построена с использованием класса `Sequential`. Архитектура включает:

| № слоя | Тип слоя                    | Количество нейронов | Функция активации |
| ------ | --------------------------- | ------------------- | ----------------- |
| 1      | Dense + BatchNorm + Dropout | 128                 | ReLU              |
| 2      | Dense + BatchNorm + Dropout | 64                  | ReLU              |
| 3      | Dense                       | 32                  | ReLU              |
| 4      | Dense (выходной)            | 3                   | Softmax           |

Оптимизатор: **Adam** (learning rate = 0.001)

Функция потерь: **categorical_crossentropy**

Метрика обучения: **accuracy**

### 5. Обучение модели

Модель обучалась на 100 эпохах, размер батча — 8. В процессе обучения контролировалась метрика точности и функция потерь как на обучающей, так и на валидационной выборках.

### 6. Результаты

```
Размерность x_data - (178, 13)
Размерность y_data - (178,)

Данные по первому вину: [1.423e+01 1.710e+00 2.430e+00 1.560e+01 1.270e+02 2.800e+00 3.060e+00
 2.800e-01 2.290e+00 5.640e+00 1.040e+00 3.920e+00 1.065e+03]
Класс вина: 0
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense (Dense)                   │ (None, 128)            │         1,792 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization             │ (None, 128)            │           512 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 128)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 64)             │         8,256 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_1           │ (None, 64)             │           256 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_1 (Dropout)             │ (None, 64)             │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 32)             │         2,080 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_3 (Dense)                 │ (None, 3)              │            99 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 12,995 (50.76 KB)
 Trainable params: 12,611 (49.26 KB)
 Non-trainable params: 384 (1.50 KB)
Epoch 1/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.7014 - loss: 0.7192 - val_accuracy: 0.6250 - val_loss: 0.9198
Epoch 2/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.8958 - loss: 0.3996 - val_accuracy: 0.9375 - val_loss: 0.7252
Epoch 3/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9514 - loss: 0.2213 - val_accuracy: 0.8750 - val_loss: 0.5871
Epoch 4/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9236 - loss: 0.2212 - val_accuracy: 0.8750 - val_loss: 0.4945
Epoch 5/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9375 - loss: 0.2138 - val_accuracy: 0.8750 - val_loss: 0.4183
Epoch 6/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9653 - loss: 0.1512 - val_accuracy: 0.8750 - val_loss: 0.3532
Epoch 7/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9583 - loss: 0.1318 - val_accuracy: 0.8750 - val_loss: 0.3050
Epoch 8/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9375 - loss: 0.1349 - val_accuracy: 0.9375 - val_loss: 0.2619
Epoch 9/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9653 - loss: 0.1023 - val_accuracy: 0.9375 - val_loss: 0.2300
Epoch 10/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9375 - loss: 0.1795 - val_accuracy: 0.9375 - val_loss: 0.2039
Epoch 11/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9861 - loss: 0.0749 - val_accuracy: 0.9375 - val_loss: 0.1806
Epoch 12/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9583 - loss: 0.1347 - val_accuracy: 0.9375 - val_loss: 0.1416
Epoch 13/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9722 - loss: 0.0849 - val_accuracy: 0.9375 - val_loss: 0.1372
Epoch 14/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9861 - loss: 0.0653 - val_accuracy: 0.9375 - val_loss: 0.1163
Epoch 15/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9444 - loss: 0.1266 - val_accuracy: 0.9375 - val_loss: 0.1164
Epoch 16/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9722 - loss: 0.0742 - val_accuracy: 0.9375 - val_loss: 0.1316
Epoch 17/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9653 - loss: 0.0716 - val_accuracy: 0.9375 - val_loss: 0.1391
Epoch 18/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9722 - loss: 0.0871 - val_accuracy: 0.9375 - val_loss: 0.1674
Epoch 19/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9514 - loss: 0.0984 - val_accuracy: 0.9375 - val_loss: 0.1632
Epoch 20/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9861 - loss: 0.0572 - val_accuracy: 0.9375 - val_loss: 0.1378
Epoch 21/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9792 - loss: 0.1056 - val_accuracy: 0.9375 - val_loss: 0.1199
Epoch 22/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9931 - loss: 0.0713 - val_accuracy: 0.9375 - val_loss: 0.1213
Epoch 23/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9722 - loss: 0.0815 - val_accuracy: 0.8750 - val_loss: 0.2744
Epoch 24/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9653 - loss: 0.0867 - val_accuracy: 0.8750 - val_loss: 0.2315
Epoch 25/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9444 - loss: 0.1364 - val_accuracy: 0.9375 - val_loss: 0.1347
Epoch 26/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9653 - loss: 0.0953 - val_accuracy: 0.9375 - val_loss: 0.0994
Epoch 27/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9792 - loss: 0.0501 - val_accuracy: 0.9375 - val_loss: 0.0772
Epoch 28/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9931 - loss: 0.0380 - val_accuracy: 0.9375 - val_loss: 0.0743
Epoch 29/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9792 - loss: 0.0379 - val_accuracy: 0.9375 - val_loss: 0.0903
Epoch 30/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9931 - loss: 0.0271 - val_accuracy: 0.9375 - val_loss: 0.0991
Epoch 31/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9792 - loss: 0.0938 - val_accuracy: 0.9375 - val_loss: 0.1262
Epoch 32/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9653 - loss: 0.0657 - val_accuracy: 0.9375 - val_loss: 0.1018
Epoch 33/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9861 - loss: 0.0677 - val_accuracy: 0.9375 - val_loss: 0.1094
Epoch 34/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9861 - loss: 0.0738 - val_accuracy: 0.9375 - val_loss: 0.0901
Epoch 35/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9792 - loss: 0.0722 - val_accuracy: 0.9375 - val_loss: 0.0855
Epoch 36/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 1.0000 - loss: 0.0150 - val_accuracy: 0.9375 - val_loss: 0.1017
Epoch 37/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9861 - loss: 0.0457 - val_accuracy: 0.9375 - val_loss: 0.1251
Epoch 38/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9792 - loss: 0.0615 - val_accuracy: 1.0000 - val_loss: 0.0516
Epoch 39/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9792 - loss: 0.0623 - val_accuracy: 0.9375 - val_loss: 0.0664
Epoch 40/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9931 - loss: 0.0655 - val_accuracy: 0.9375 - val_loss: 0.0864
Epoch 41/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9653 - loss: 0.0578 - val_accuracy: 1.0000 - val_loss: 0.0447
Epoch 42/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9792 - loss: 0.0452 - val_accuracy: 1.0000 - val_loss: 0.0290
Epoch 43/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9722 - loss: 0.1305 - val_accuracy: 1.0000 - val_loss: 0.0182
Epoch 44/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9931 - loss: 0.0273 - val_accuracy: 1.0000 - val_loss: 0.0179
Epoch 45/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9375 - loss: 0.1460 - val_accuracy: 0.9375 - val_loss: 0.0896
Epoch 46/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9931 - loss: 0.0252 - val_accuracy: 0.9375 - val_loss: 0.0938
Epoch 47/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9861 - loss: 0.0543 - val_accuracy: 0.9375 - val_loss: 0.0947
Epoch 48/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9931 - loss: 0.0502 - val_accuracy: 0.9375 - val_loss: 0.1140
Epoch 49/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 1.0000 - loss: 0.0141 - val_accuracy: 0.9375 - val_loss: 0.1097
Epoch 50/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9931 - loss: 0.0282 - val_accuracy: 0.9375 - val_loss: 0.0650
Epoch 51/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9722 - loss: 0.0655 - val_accuracy: 0.9375 - val_loss: 0.0800
Epoch 52/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9792 - loss: 0.0643 - val_accuracy: 0.9375 - val_loss: 0.0841
Epoch 53/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9722 - loss: 0.0760 - val_accuracy: 0.8750 - val_loss: 0.2440
Epoch 54/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9583 - loss: 0.0724 - val_accuracy: 0.8750 - val_loss: 0.2802
Epoch 55/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9514 - loss: 0.1337 - val_accuracy: 0.9375 - val_loss: 0.1090
Epoch 56/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9931 - loss: 0.0291 - val_accuracy: 0.9375 - val_loss: 0.0814
Epoch 57/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9722 - loss: 0.0555 - val_accuracy: 0.9375 - val_loss: 0.0852
Epoch 58/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9722 - loss: 0.0825 - val_accuracy: 1.0000 - val_loss: 0.0630
Epoch 59/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9931 - loss: 0.0251 - val_accuracy: 1.0000 - val_loss: 0.0528
Epoch 60/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9375 - loss: 0.1550 - val_accuracy: 0.9375 - val_loss: 0.0868
Epoch 61/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9861 - loss: 0.0660 - val_accuracy: 0.9375 - val_loss: 0.1177
Epoch 62/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9792 - loss: 0.0591 - val_accuracy: 0.9375 - val_loss: 0.0885
Epoch 63/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9722 - loss: 0.0878 - val_accuracy: 0.9375 - val_loss: 0.0831
Epoch 64/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9931 - loss: 0.0371 - val_accuracy: 0.9375 - val_loss: 0.0720
Epoch 65/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9583 - loss: 0.1757 - val_accuracy: 1.0000 - val_loss: 0.0481
Epoch 66/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9931 - loss: 0.0280 - val_accuracy: 1.0000 - val_loss: 0.0435
Epoch 67/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 1.0000 - loss: 0.0118 - val_accuracy: 1.0000 - val_loss: 0.0346
Epoch 68/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 1.0000 - loss: 0.0158 - val_accuracy: 1.0000 - val_loss: 0.0374
Epoch 69/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9861 - loss: 0.0361 - val_accuracy: 1.0000 - val_loss: 0.0520
Epoch 70/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9583 - loss: 0.1042 - val_accuracy: 0.9375 - val_loss: 0.0863
Epoch 71/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9583 - loss: 0.1085 - val_accuracy: 1.0000 - val_loss: 0.0530
Epoch 72/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9861 - loss: 0.0422 - val_accuracy: 1.0000 - val_loss: 0.0396
Epoch 73/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9931 - loss: 0.0525 - val_accuracy: 1.0000 - val_loss: 0.0366
Epoch 74/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9931 - loss: 0.0206 - val_accuracy: 1.0000 - val_loss: 0.0355
Epoch 75/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 1.0000 - loss: 0.0065 - val_accuracy: 1.0000 - val_loss: 0.0350
Epoch 76/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9861 - loss: 0.0637 - val_accuracy: 1.0000 - val_loss: 0.0464
Epoch 77/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9861 - loss: 0.0333 - val_accuracy: 1.0000 - val_loss: 0.0418
Epoch 78/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 1.0000 - loss: 0.0117 - val_accuracy: 1.0000 - val_loss: 0.0419
Epoch 79/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9792 - loss: 0.0307 - val_accuracy: 1.0000 - val_loss: 0.0400
Epoch 80/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9931 - loss: 0.0227 - val_accuracy: 1.0000 - val_loss: 0.0179
Epoch 81/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9931 - loss: 0.0417 - val_accuracy: 1.0000 - val_loss: 0.0230
Epoch 82/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9861 - loss: 0.0626 - val_accuracy: 1.0000 - val_loss: 0.0459
Epoch 83/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9861 - loss: 0.0364 - val_accuracy: 1.0000 - val_loss: 0.0519
Epoch 84/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9861 - loss: 0.0316 - val_accuracy: 1.0000 - val_loss: 0.0548
Epoch 85/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9931 - loss: 0.0242 - val_accuracy: 1.0000 - val_loss: 0.0454
Epoch 86/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9931 - loss: 0.0239 - val_accuracy: 1.0000 - val_loss: 0.0334
Epoch 87/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9931 - loss: 0.0189 - val_accuracy: 1.0000 - val_loss: 0.0413
Epoch 88/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9931 - loss: 0.0203 - val_accuracy: 1.0000 - val_loss: 0.0582
Epoch 89/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9722 - loss: 0.0580 - val_accuracy: 1.0000 - val_loss: 0.0388
Epoch 90/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9931 - loss: 0.0322 - val_accuracy: 1.0000 - val_loss: 0.0222
Epoch 91/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 1.0000 - loss: 0.0047 - val_accuracy: 1.0000 - val_loss: 0.0175
Epoch 92/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9931 - loss: 0.0423 - val_accuracy: 1.0000 - val_loss: 0.0121
Epoch 93/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9931 - loss: 0.0185 - val_accuracy: 1.0000 - val_loss: 0.0104
Epoch 94/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9931 - loss: 0.0227 - val_accuracy: 1.0000 - val_loss: 0.0110
Epoch 95/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 1.0000 - loss: 0.0152 - val_accuracy: 1.0000 - val_loss: 0.0109
Epoch 96/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9931 - loss: 0.0240 - val_accuracy: 1.0000 - val_loss: 0.0156
Epoch 97/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 1.0000 - loss: 0.0200 - val_accuracy: 1.0000 - val_loss: 0.0293
Epoch 98/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9861 - loss: 0.0283 - val_accuracy: 1.0000 - val_loss: 0.0259
Epoch 99/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.9931 - loss: 0.0377 - val_accuracy: 1.0000 - val_loss: 0.0551
Epoch 100/100
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 1.0000 - loss: 0.0051 - val_accuracy: 0.9375 - val_loss: 0.0680

Точность на тестовой выборке: 94.44%
```

### 7. Выводы

В ходе лабораторной работы была успешно построена и обучена нейронная сеть для задачи многоклассовой классификации. Были изучены этапы предобработки данных, построения последовательной модели нейронной сети и анализа результатов. Полученная точность говорит о том, что модель успешно обобщает данные и способна классифицировать новые объекты.

### 8. Приложение: Листинг программы

Код программы приведён в файле ```full_conected_mesh.py```


