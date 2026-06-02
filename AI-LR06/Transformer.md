# Лабораторная 1: Transformer encoder для order-sensitive toy classification

## Контракт данных

Вход:
- padded последовательности целых token ids,
- форма `X -> (N, T)`.

Выход:
- бинарная метка `y -> (N,)`.

Правило метки:
- `y = 1`, если `7` стоит раньше `3`;
- `y = 0`, если `3` стоит раньше `7`.

## Таблица форм тензоров

| Сущность | Смысл | Форма |
|---|---|---|
| `X_train` | padded token ids | `(N, T)` |
| `padding_mask` | полезные позиции | `(N, T)` |
| `embeddings` | token + position embeddings | `(N, T, E)` |
| `attention_scores` | веса внимания по головам | `(N, H, T, T)` |
| `pooled` | один вектор на объект | `(N, E)` |
| `y_pred` | вероятность класса | `(N, 1)` |

## Шпаргалка по обозначениям и формам

- `N` — число объектов.
- `T` — длина последовательности после padding.
- `E` — размер embedding / model dimension.
- `H` — число attention heads.
- `PAD = 0`.

Практический фокус этой ЛР:
- проверить shapes;
- проверить mask;
- проверить, что attention не уходит в padded хвост.

## Контракт модели

Модель должна состоять из таких блоков:
1. `TokenAndPositionEmbedding`
2. `TransformerEncoderBlock`
3. masked average pooling
4. classifier head для binary classification

Отдельно нужен tracing-path, который возвращает `attention_scores` хотя бы для одного encoder block.

## Мини-теория

Минимальный encoder block:

$$
H_1 = \mathrm{LayerNorm}(X + \mathrm{MHA}(X))
$$

$$
H_2 = \mathrm{LayerNorm}(H_1 + \mathrm{FFN}(H_1))
$$

Если positional embedding отсутствует, то attention знает “какие токены были”, но слабо знает “где они были”.
Для этой лабораторной это критично, потому что метка зависит именно от порядка.

## Ручной разбор одного примера

Сравните две последовательности:

```text
A = [7, 5, 2, 3]
B = [3, 5, 2, 7]
```

Если смотреть только на набор токенов, они одинаковы.
Но по правилу задачи:
- `A -> 1`
- `B -> 0`

Значит, модели нужен позиционный сигнал.

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

SEED = 7
PAD_ID = 0
KEY_TOKEN = 7
VALUE_TOKEN = 3
VOCAB_SIZE = 16
SEQ_LEN = 12
MIN_LEN = 4
TRAIN_SIZE = 4000
TEST_SIZE = 1000
EMBED_DIM = 32
NUM_HEADS = 2
FF_DIM = 64
BATCH_SIZE = 64
EPOCHS = 6

keras.utils.set_random_seed(SEED)
np.set_printoptions(linewidth=120)
```

## Checkpoint 1: данные

На этом этапе нужно убедиться в трёх вещах:
- padded последовательности имеют общую длину `T`;
- токены `7` и `3` всегда присутствуют в полезной части;
- label зависит именно от порядка этих токенов.


```python
filler_tokens = np.array(
    [token for token in range(1, VOCAB_SIZE) if token not in (KEY_TOKEN, VALUE_TOKEN)],
    dtype=np.int32,
)


def generate_order_dataset(num_samples, seq_len=SEQ_LEN, min_len=MIN_LEN, seed=SEED):
    """Генерирует синтетический набор для задачи чувствительности к порядку.

    Args:
      num_samples: Число последовательностей для генерации.
      seq_len: Максимальная длина последовательности после дополнения.
      min_len: Минимальная полезная длина до дополнения.
      seed: Зерно случайности для воспроизводимости.

    Returns:
      Кортеж `(X, y, lengths)`, где:
        `X`: Матрица токенов формы `(num_samples, seq_len)`.
        `y`: Вектор бинарных меток формы `(num_samples,)`.
        `lengths`: Полезные длины до дополнения формы `(num_samples,)`.

    Raises:
      ValueError: Если `min_len` или `seq_len` заданы некорректно.
    """
    if min_len < 2 or seq_len < min_len:
        raise ValueError('Ожидается 2 <= min_len <= seq_len.')

    rng = np.random.default_rng(seed)
    X = np.full((num_samples, seq_len), PAD_ID, dtype=np.int32)
    y = np.zeros((num_samples,), dtype=np.int32)
    lengths = np.zeros((num_samples,), dtype=np.int32)

    for i in range(num_samples):
        
        length = int(rng.integers(min_len, seq_len + 1)) 
        
        tokens = rng.choice(filler_tokens, size=length) # собираем последовательность длиной length из filler_tokens
        pos_key, pos_value = rng.choice(length, size=2, replace=False) # выбираем случайные места для key и value
        
        tokens[pos_key] = KEY_TOKEN # добавляем ключ на позицию, которую случайно выбрали
        tokens[pos_value] = VALUE_TOKEN # добавляем значение на позицию, которую случайно выбрали
        
        X[i, :length] = tokens # записываем значимые значения в i-ой строке с 0 до length
        y[i] = int(pos_key < pos_value) # метка, которая определяет, ключ встретился раньше значения или нет
        
        lengths[i] = length # пометка до какого индекса идут значимые значения

    return X, y, lengths


X_all, y_all, lengths_all = generate_order_dataset(TRAIN_SIZE + TEST_SIZE)
X_train, X_test, y_train, y_test, len_train, len_test = train_test_split(
    X_all,
    y_all,
    lengths_all,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=y_all,
)

print('X_train shape:', X_train.shape)
print('X_test shape :', X_test.shape)
print('y_train mean :', y_train.mean())

manual_a = np.array([[7, 5, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int32)
manual_b = np.array([[3, 5, 2, 7, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int32)

print('manual A label should be 1')
print('manual B label should be 0')
print('useful lengths sample:', len_train[:5])
print('class balance      :', np.bincount(y_train))
```

## Checkpoint 2: embeddings + mask

Здесь нужно проверить две базовые идеи:
- `TokenAndPositionEmbedding` возвращает `(batch, time, embed_dim)`;
- padding mask сохраняется и потом попадёт в attention и pooling.

```python
class TokenAndPositionEmbedding(layers.Layer):
    """Складывает токенные и позиционные векторы.

    Args:
      maxlen: Максимальная длина последовательности.
      vocab_size: Размер словаря токенов.
      embed_dim: Размерность векторного представления.
      **kwargs: Дополнительные аргументы базового слоя Keras.

    Returns:
      Экземпляр слоя встраивания токенов и позиций.

    Raises:
      ValueError: Если `embed_dim` меньше 1.
    """

    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        if embed_dim < 1:
            raise ValueError('embed_dim должен быть положительным.')
        self.token_emb = layers.Embedding(vocab_size, embed_dim, mask_zero=True)
        self.pos_emb = layers.Embedding(maxlen, embed_dim)
        self.supports_masking = True

    def call(self, inputs):
        """Возвращает сумму токенных и позиционных векторов.

        Args:
          inputs: Целочисленные токены формы `(batch, time)`.

        Returns:
          Тензор формы `(batch, time, embed_dim)`.

        Raises:
          NotImplementedError: Пока шаг `TODO 2` не реализован.
        """
        
        positions = tf.range(start=0, limit=tf.shape(inputs)[-1], delta=1) # задаем диапазон, стартуем с 0 и до количсетва тензора inputs не включая его самого, разделитель 1(по умолчанию)
        
        token_embeddings = self.token_emb(inputs) # получем token использую Embedding реализованный раннее
        position_embeddings = self.pos_emb(positions) # получем position использую Embedding реализованный раннее
        
        result = token_embeddings + position_embeddings # получаем сумму

        return result

    def compute_mask(self, inputs, mask=None):
        """Переадресует маску непустых токенов дальше по графу.

        Args:
          inputs: Целочисленные токены формы `(batch, time)`.
          mask: Входная маска базового уровня.

        Returns:
          Булева маска формы `(batch, time)`.

        Raises:
          NotImplementedError: Пока шаг `TODO 2` не реализован.
        """
        
        return self.token_emb.compute_mask(inputs) # получает на вход послежовательность и обрабатывает ее, возвращая True на местах реальных значений и False на местах 0


def masked_average(x, mask):
    """Вычисляет среднее по времени с учётом маски.

    Args:
      x: Тензор признаков формы `(batch, time, embed_dim)`.
      mask: Булева маска полезных позиций формы `(batch, time)`.

    Returns:
      Усреднённый тензор формы `(batch, embed_dim)`.

    Raises:
      ValueError: Если входной тензор имеет неверный ранг.
    """
    if x.shape.rank != 3:
        raise ValueError('Ожидается ранг 3 для тензора признаков.')
    mask = tf.cast(mask, x.dtype)
    mask = tf.expand_dims(mask, axis=-1)
    summed = tf.reduce_sum(x * mask, axis=1)
    counts = tf.reduce_sum(mask, axis=1)
    return summed / tf.maximum(counts, 1.0)


class TransformerEncoderBlock(layers.Layer):
    """Минимальный блок кодировщика трансформера.

    Args:
      embed_dim: Размерность входных признаков.
      num_heads: Число голов внимания.
      ff_dim: Размер скрытого слоя позиционно-независимой сети.
      rate: Доля выключаемых нейронов в прореживании.
      **kwargs: Дополнительные аргументы базового слоя Keras.

    Returns:
      Экземпляр блока кодировщика.

    Raises:
      ValueError: Если `embed_dim` не делится на `num_heads`.
    """

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        if embed_dim % num_heads != 0:
            raise ValueError('embed_dim должен делиться на num_heads без остатка.')
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=rate,
        )
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation='relu'),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.supports_masking = True

    def call(self, inputs, mask=None, training=None, return_attention_scores=False):
        """Выполняет прямой проход блока кодировщика.

        Args:
          inputs: Входной тензор формы `(batch, time, embed_dim)`.
          mask: Булева маска формы `(batch, time)`.
          training: Признак режима обучения.
          return_attention_scores: Вернуть ли дополнительно веса внимания.

        Returns:
          Либо выходной тензор формы `(batch, time, embed_dim)`,
          либо кортеж `(output, attention_scores)`.

        Raises:
          NotImplementedError: Пока шаг `TODO 3` не реализован.
        """

        if mask is not None:
          query_mask = tf.cast(mask[:, :, None], dtype=bool) # Маска запросов. None - позволит позже транслировать (broadcast) с маской ключей.
          key_mask = tf.cast(mask[:, None, :], dtype=bool) # Маска ключей. None - создает измерение для транслирования.
          attention_mask = query_mask & key_mask # Создаем 2D-маску.

        attention_output, attention_scores = self.att(
                query=inputs,                 # 1. Что ищем
                value=inputs,                 # 2. Что достаем
                key=inputs,                   # 3. По чему ищем
                attention_mask=attention_mask,# 4. Куда смотреть нельзя
                return_attention_scores=True) # 5. Просим вернуть веса
            
        attention_output = self.dropout1(attention_output, training=training) # dropout 1
        x = self.layernorm1(inputs + attention_output)  # Residual + LayerNorm 1
        ffn_output = self.ffn(x) # Пропускаем через позиционно-независимую сеть из двух плотных слоев для нелинейного преобразования признаков
        ffn_output = self.dropout2(ffn_output, training=training) # dropout 2
        x = self.layernorm2(x + ffn_output)   # Residual + LayerNorm 2
        
        if return_attention_scores:
            return x, attention_scores
        
        return x

    def compute_mask(self, inputs, mask=None):
        """Пробрасывает временную маску на следующий слой.

        Args:
          inputs: Входной тензор признаков.
          mask: Входная булева маска.

        Returns:
          Та же маска без изменений.

        Raises:
          RuntimeError: Не выбрасывается в штатном режиме.
        """
        return mask

sample_tokens = X_train[:2]
sample_mask = sample_tokens != PAD_ID

sample_embedding_layer = TokenAndPositionEmbedding(SEQ_LEN, VOCAB_SIZE, EMBED_DIM)
sample_encoder_block = TransformerEncoderBlock(EMBED_DIM, NUM_HEADS, FF_DIM)

sample_embeddings = sample_embedding_layer(sample_tokens)
sample_encoded, sample_scores = sample_encoder_block(
    sample_embeddings,
    mask=sample_mask,
    return_attention_scores=True,
)

print('sample_embeddings:', sample_embeddings.shape)
print('sample_encoded   :', sample_encoded.shape)
print('sample_scores    :', sample_scores.shape)
```

## Checkpoint 3: encoder block + classifier

После этого блока модель должна:
- принимать `tokens -> (batch, time)`;
- превращать их в embeddings с позициями;
- прогонять через `TransformerEncoderBlock`;
- сворачивать всё в одну вероятность `y_pred -> (batch, 1)` через `sigmoid`.

```python
keras.utils.set_random_seed(SEED)

transformer_inputs = keras.Input(shape=(SEQ_LEN,), dtype='int32', name='tokens')
padding_mask = layers.Lambda(lambda x: tf.not_equal(x, PAD_ID), name='padding_mask')(transformer_inputs)

embedding_layer = TokenAndPositionEmbedding(
    vocab_size=VOCAB_SIZE,      # Размер словаря
    maxlen=SEQ_LEN,             # Максимальный размер
    embed_dim=EMBED_DIM         # Размерность выходных данных слоя встраивания
)

encoder_layer = TransformerEncoderBlock(
    embed_dim=EMBED_DIM,        # Количество точек внимания
    num_heads=NUM_HEADS,        # Размерность выходных данных первого плотного слоя в двухслойной нейронной сети прямого распространения
    ff_dim=FF_DIM,              # Активация для первого плотного слоя в двухслойной нейронной сети прямого распространения
)

x = embedding_layer(transformer_inputs) # Подаем входные данные
x = encoder_layer(x, mask=padding_mask) # Подаем входные данные и маску

output = tf.keras.layers.GlobalAveragePooling1D()(x, mask=padding_mask) # создаем объект (пустые скобки), далее «вызываем» его, передавая данные(входные данные, маска)

x = tf.keras.layers.Dense(64, activation='relu')(output) # Ищем нелинейные завиимости в нашем слое с помощью relu
x = tf.keras.layers.Dropout(0.5)(x) # Борьба с переобучением, на случайном этапе отключая 50% нейронов
predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x) # Сжимает все вычисления в одно единственное число от 0 до 1

model = tf.keras.Model(inputs=transformer_inputs, outputs=predictions) # Соединаем наши данные в итоговую модель

# Скомпилировали модель
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()
```

## Checkpoint 4: обучение

Перед запуском `fit` проверьте:
- classifier head действительно бинарный;
- loss = `binary_crossentropy`;
- train и validation будут сравниваться отдельно от финального test.

```python
history = model.fit(
    x=X_train,               # Ваши входные данные (матрица индексов токенов)
    y=y_train,               # Правильные ответы (0 или 1)
    batch_size=64,           # Размер порции данных для одного шага градиента
    epochs=10,               # Сколько раз модель полностью пройдет по данным
    validation_split=0.2,    # Отделить 20% данных на проверку (модель их не видит при обучении)
    shuffle=True             # Перемешивать данные перед каждой эпохой
)

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
```

## Checkpoint 5: attention trace и критерии завершения

Перед сдачей здесь должны одновременно выполняться все условия:
- `test_acc >= 0.95`;
- два ручных примера с перестановкой `7` и `3` дают разные предсказания;
- heatmap строится только по non-PAD части последовательности.

```python
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f'test_loss = {test_loss:.4f}')
print(f'test_acc  = {test_acc:.4f}')

paired_examples = np.array(
    [
        [7, 5, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, 5, 2, 7, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=np.int32,
)

paired_probs = model.predict(paired_examples, verbose=0).ravel()
for seq, prob in zip(paired_examples, paired_probs):
    label = int(prob >= 0.5)
    print(seq, '-> prob=', round(float(prob), 4), 'label=', label)

# TODO 6.1-6.3: Компактное решение
sample_index = 0 # Берем первый пример из тестовой выборки
sample_tokens = X_test[sample_index:sample_index + 1] # Берем срез от sample_index до sample_index+1, сохраняя размерность (1, seq_len)
sample_length = (sample_tokens[0] != PAD_ID).sum() # Берем первый и единственный элемент батча, сравниваем с ID и получаем маску (True/False) после чего суммируем полученные ответы.

# Получаем encoder и эмбеддинги
encoder = model.get_layer('transformer_encoder_block_1') # Достаем encoder блок из обученной модели
embeddings = model.get_layer('token_and_position_embedding_1')(sample_tokens) # Вызываем embedding слой
mask = tf.not_equal(sample_tokens, PAD_ID) # True для реальных токенов, False для padding

# Получаем attention scores
_, attention_scores = encoder(embeddings, mask=mask, return_attention_scores=True) # Вызываем encoder: embeddings - входные векторы, mask - маска для игнорирования padding, return_attention_scores=True - просим encoder вернуть веса внимания, _ - игнорируем выход encoder (закодированные векторы), так как нас интересуют только веса.

# Конвертируем в numpy, усредняем и обрезаем
attention_np = attention_scores.numpy()  # Преводим в массив Numpy
mean_attention = attention_np[0].mean(axis=0)[:sample_length, :sample_length] # attention_np[0] - берем первый (и единственный) пример из батча, .mean(axis=0) - усредняем по головам, [:sample_length, :sample_length] - обрезаем до реальной длины.

print(f'Attention shape: {mean_attention.shape}') 
token_labels = [str(token) for token in sample_tokens[0][:sample_length]] # Создаем текстовые метки для каждого токена, sample_tokens[0] - первый пример из батча, [:sample_length] - берем только реальные токены (без padding), str(token) - преобразуем число в строку.

plt.figure(figsize=(6, 5))
plt.imshow(mean_attention, cmap='magma', aspect='auto')
plt.colorbar(label='attention weight')
plt.xticks(range(sample_length), token_labels)
plt.yticks(range(sample_length), token_labels)
plt.xlabel('key positions')
plt.ylabel('query positions')
plt.title('Mean self-attention over heads')
plt.tight_layout()
plt.show()
```
