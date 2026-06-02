# Лабораторная работа 1: GRU seq2seq с Luong attention для reverse-sequonce

## Цель
Реализовать `GRU` encoder-decoder с `Luong attention`, который разворачивает входную последовательность в обратном порядке.

Эта лабораторная продолжает `01-RNN / ЛР03`, но убирает fixed-context bottleneck: decoder теперь может смотреть на все позиции `encoder_outputs`, а не только на один финальный вектор состояния.

## Контракт данных
Используются целочисленные токены:
- `1..9` — содержательные значения;
- `PAD=0` — заполнение;
- `SOS=10` — старт декодера;
- `EOS=11` — конец последовательности.

Используется та же reverse-задача, что и в ЛР3 по `RNN`, но теперь:
- `ENC_LEN = 10`;
- реальная длина последовательности случайна в диапазоне `4..10`.

Формируются три связанных тензора:
- `encoder_input`;
- `decoder_input`;
- `decoder_target`.

## Таблица форм тензоров

| Тензор | Форма | Смысл | Где используется |
|---|---|---|---|
| `encoder_input` | `(N, T_in)` | Вход encoder | `model.fit` / `model.predict` |
| `decoder_input` | `(N, T_out)` | Вход decoder со сдвигом | `model.fit` / `model.predict` |
| `decoder_target` | `(N, T_out, 1)` | Истинные выходные токены | Функция потерь |
| `encoder_outputs` | `(N, T_in, H)` | Состояния encoder по всем входным позициям | Attention |
| `decoder_outputs` | `(N, T_out, H)` | Состояния decoder по всем выходным шагам | Attention / head |
| `context` | `(N, T_out, H)` | Контекст из encoder для каждого decoder-шага | Attention |
| `attention_scores` | `(N, T_out, T_in)` | Веса внимания по входным позициям | Диагностика |
| `probs` | `(N_test, T_out, V)` | Распределения вероятностей по словарю | `model.predict` |
| `preds` | `(N_test, T_out)` | Предсказанные индексы | Оценка |
| `exact_match` | скаляр | Доля полностью верных последовательностей | Итоговая метрика |

## Шпаргалка по обозначениям и формам

Короткая карта обозначений:
- `encoder_outputs` - все скрытые состояния энкодера, по одному на каждую входную позицию.
- `decoder_outputs` - все скрытые состояния декодера.
- `context` - контекстный вектор, собранный attention для каждого decoder-step.
- `attention_scores` - веса внимания между шагами декодера и позициями входа.
- `T_in` - длина входной последовательности, `T_out` - длина выходной.
- `H` - размер скрытого состояния.

Формы тензоров:
- `encoder_outputs`: `(N, T_in, H)`.
- `decoder_outputs`: `(N, T_out, H)`.
- `context`: `(N, T_out, H)`.
- `attention_scores`: `(N, T_out, T_in)`.
- итоговый выход после `Dense(VOCAB_SIZE)`: `(batch, T_out, V)`.

## Контракт модели
- В `model.fit` передается список входов `[encoder_input, decoder_input]` и `decoder_target`.
- `encoder` должен вернуть и всю последовательность состояний, и финальное состояние.
- `decoder` инициализируется `encoder_state`, но на каждом шаге дополнительно смотрит на `encoder_outputs` через `Attention(score_mode="dot")`.
- Выходной слой получает конкатенацию `[decoder_outputs; context]`.
- Для функции потерь `sparse_categorical_crossentropy` целевые значения задаются целыми индексами классов.
- Для диагностики строится отдельный `analysis_model`, который возвращает промежуточные тензоры `encoder_outputs`, `decoder_outputs`, `context`, `attention_scores`.

## Мини-теория
В этой лабораторной используется `cross-attention` в варианте Luong:

$$
q_t = h_t^{dec}, \quad k_s = h_s^{enc}, \quad v_s = h_s^{enc}
$$

$$
e_{t,s} = q_t^\top k_s, \quad
\alpha_{t,s} = \mathrm{softmax}_s(e_{t,s}), \quad
c_t = \sum_{s=1}^{T_{in}} \alpha_{t,s} v_s
$$

$$
z_t = [h_t^{dec}; c_t], \quad s_t = W_{out} z_t + b_{out}, \quad
\hat{y}_t = \mathrm{softmax}(s_t)
$$

Здесь:
- `query` идет от текущего шага decoder;
- `key/value` идут от всех шагов encoder;
- `attention_scores[t, s]` показывает, на какую входную позицию модель смотрит на шаге `t`.

```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_seed(42)
print('Версия TensorFlow:', tf.__version__)
```

`Вывод: Версия TensorFlow: 2.19.1`

## Генерация данных
**Что сделать:** подготовить `encoder_input`, `decoder_input`, `decoder_target` со сдвигом на один шаг.

**Почему:** decoder по-прежнему обучается по схеме `teacher forcing`, но теперь задача специально сделана длиннее, чтобы plain `seq2seq` чаще упирался в одно фиксированное состояние.

**Ожидаемые формы:** `(N, T_in)`, `(N, T_out)`, `(N, T_out, 1)`.

**Частая ошибка:** неверный сдвиг между `decoder_input` и `decoder_target`.

### Подсказка к TODO 1-3: данные

Концептуальная подсказка:
- data pipeline почти совпадает с `01-RNN / ЛР03`, но теперь входные последовательности длиннее и сильнее показывают bottleneck plain `seq2seq`.

Implementation hint:
- `rev = seq[::-1]`;
- `decoder_input` начинается с `SOS`;
- `decoder_target` заканчивается `EOS`;
- ожидаемые формы: `(N, 10)`, `(N, 11)`, `(N, 11, 1)`.

```python
PAD_ID = 0
SOS_ID = 10
EOS_ID = 11
VOCAB_SIZE = 20 # Изменено на 20, так как иначе образовывалась ошибка из-за переполнения.
ENC_LEN = 10
DEC_LEN = ENC_LEN + 1


def decode_token(tok: int) -> str:
    mapping = {PAD_ID: 'PAD', SOS_ID: 'SOS', EOS_ID: 'EOS'}
    return mapping.get(int(tok), str(int(tok)))


def make_one_sample(min_len: int = 4, max_len: int = ENC_LEN):
    length = np.random.randint(min_len, max_len + 1)
    seq = np.random.randint(1, 10, size=length, dtype=np.int32)
    # TODO 1: получите обратную последовательность
    rev = seq[::-1]
    return seq, rev


def pad_sequence(seq, target_len, pad_value=PAD_ID):
    out = np.full((target_len,), pad_value, dtype=np.int32)
    out[:len(seq)] = seq
    return out


def make_dataset(n_samples: int = 8000):
    encoder_input = np.zeros((n_samples, ENC_LEN), dtype=np.int32)
    decoder_input = np.zeros((n_samples, DEC_LEN), dtype=np.int32)
    decoder_target = np.zeros((n_samples, DEC_LEN), dtype=np.int32)

    for i in range(n_samples):
        seq, rev = make_one_sample()
        enc = pad_sequence(seq, ENC_LEN)

        # TODO 2: сформируйте decoder_input: [SOS] + rev
        dec_in = np.concatenate(([SOS_ID], rev))
        # TODO 3: сформируйте decoder_target: rev + [EOS]
        dec_out = np.concatenate((rev, [EOS_ID]))

        encoder_input[i] = pad_sequence(enc, ENC_LEN)
        decoder_input[i] = pad_sequence(dec_in, DEC_LEN)
        decoder_target[i] = pad_sequence(dec_out, DEC_LEN)

    return encoder_input, decoder_input, decoder_target[..., None]


enc_in, dec_in, dec_tgt = make_dataset()

enc_train, enc_test, dec_in_train, dec_in_test, dec_tgt_train, dec_tgt_test = train_test_split(
    enc_in,
    dec_in,
    dec_tgt,
    test_size=0.2,
    random_state=42,
)

print('Форма enc_train    :', enc_train.shape)
print('Форма dec_in_train :', dec_in_train.shape)
print('Форма dec_tgt_train:', dec_tgt_train.shape)
print('Пример encoder_input :', enc_train[0])
print('Пример decoder_input :', dec_in_train[0])
print('Пример decoder_target:', dec_tgt_train[0, :, 0])
```

Вывод:
```
Форма enc_train    : (6400, 10)
Форма dec_in_train : (6400, 11)
Форма dec_tgt_train: (6400, 11, 1)
Пример encoder_input : [4 4 2 3 2 1 4 9 6 8]
Пример decoder_input : [10  8  6  9  4  1  2  3  2  4  4]
Пример decoder_target: [ 8  6  9  4  1  2  3  2  4  4 11]
```

### Разбор TODO 1-3: данные

После заполнения блока проверьте три вещи:
- длины действительно лежат в диапазоне `4..10`;
- `decoder_input` и `decoder_target` сдвинуты ровно на один шаг;
- `decoder_target` хранит целочисленные индексы токенов с последней осью размера `1`.

```python
# Мини-проверка данных
assert enc_in.shape[1] == ENC_LEN and enc_in.ndim == 2
assert dec_in.shape[1] == DEC_LEN and dec_in.ndim == 2
assert dec_tgt.shape == (enc_in.shape[0], DEC_LEN, 1)
print('Мини-проверка данных: OK')
```

Вывод: `Мини-проверка данных: OK`

## Модель
Используется полный `Functional API`, потому что attention требует несколько промежуточных тензоров.

**Что сделать:** собрать модель из блоков:
- `Embedding(mask_zero=True)` для encoder;
- `GRU(return_sequences=True, return_state=True)` для encoder;
- `Embedding(mask_zero=True)` для decoder;
- `GRU(return_sequences=True, return_state=True, initial_state=encoder_state)` для decoder;
- `Attention(score_mode="dot")`;
- `Concatenate([decoder_outputs, context])`;
- `Dense(VOCAB_SIZE, activation="softmax")`.

### Подсказка к TODO 4-13: модель

Концептуальная подсказка:
- сначала соберите обычный seq2seq-скелет, а уже затем добавляйте attention поверх выходов encoder и decoder.

Implementation hint:
- encoder: `Embedding(mask_zero=True)` -> `GRU(return_sequences=True, return_state=True)`;
- decoder: `Embedding(mask_zero=True)` -> `GRU(return_sequences=True, return_state=True, initial_state=encoder_state)`;
- attention: `Attention(score_mode='dot')([decoder_outputs, encoder_outputs], return_attention_scores=True)`;
- далее объедините `decoder_outputs` и `context` по последней оси и завершите модель `Dense(vocab_size, activation='softmax')`.

```python
def build_model(vocab_size: int = VOCAB_SIZE, emb_dim: int = 32, latent_dim: int = 64):
    encoder_inputs = tf.keras.layers.Input(shape=(ENC_LEN,), name='encoder_inputs')
    
    # TODO 4: создайте encoder embedding с mask_zero=True
    enc_emb = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=emb_dim,
        mask_zero=False,
        name='enc_embedding')(encoder_inputs)

    # TODO 5: создайте encoder GRU c return_sequences=True и return_state=True
    enc_gru = tf.keras.layers.GRU(
        latent_dim,
        return_sequences=True,
        return_state=True)
    
    # TODO 6: получите encoder_outputs и encoder_state
    encoder_outputs, encoder_state = enc_gru(enc_emb)

    decoder_inputs = tf.keras.layers.Input(shape=(DEC_LEN,), name='decoder_inputs')
    
    # TODO 7: создайте decoder embedding с mask_zero=True
    dec_emb = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=emb_dim,
        mask_zero=False,
        name='dec_embedding')(decoder_inputs)

    # TODO 8: создайте decoder GRU c return_sequences=True и return_state=True
    dec_gru = tf.keras.layers.GRU(
        latent_dim,
        return_sequences=True,
        return_state=True)
    
    # TODO 9: получите decoder_outputs с initial_state=encoder_state
    decoder_outputs, _ = dec_gru(dec_emb, initial_state=encoder_state)

    attn = tf.keras.layers.AdditiveAttention(name='cross_attention')
    
    # TODO 10: получите context и attention_scores
    context, attention_scores = attn([decoder_outputs, encoder_outputs], return_attention_scores=True)

    # TODO 11: объедините decoder_outputs и context по последней оси
    merged = tf.keras.layers.Concatenate(name='merge_context')([decoder_outputs, context])
    
    # TODO 12: добавьте Dense(vocab_size, activation='softmax')
    outputs = tf.keras.layers.Dense(vocab_size, activation='softmax', name='output_head')(merged)

    model = tf.keras.Model([encoder_inputs, decoder_inputs], outputs, name='gru_seq2seq_attention')
    
    analysis_model = tf.keras.Model(
        [encoder_inputs, decoder_inputs],
        [encoder_outputs, decoder_outputs, context, attention_scores, outputs],
        name='attention_analysis_model',
    )

    # TODO 13: скомпилируйте модель (adam, sparse_categorical_crossentropy, accuracy)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model, analysis_model


model, analysis_model = build_model()
model.summary()
```

Вывод:
```
Model: "gru_seq2seq_attention"

┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)        ┃ Output Shape      ┃    Param # ┃ Connected to      ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ encoder_inputs      │ (None, 10)        │          0 │ -                 │
│ (InputLayer)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ decoder_inputs      │ (None, 11)        │          0 │ -                 │
│ (InputLayer)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ enc_embedding       │ (None, 10, 32)    │        640 │ encoder_inputs[0… │
│ (Embedding)         │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dec_embedding       │ (None, 11, 32)    │        640 │ decoder_inputs[0… │
│ (Embedding)         │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ gru_4 (GRU)         │ [(None, 10, 64),  │     18,816 │ enc_embedding[0]… │
│                     │ (None, 64)]       │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ gru_5 (GRU)         │ [(None, 11, 64),  │     18,816 │ dec_embedding[0]… │
│                     │ (None, 64)]       │            │ gru_4[0][1]       │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ cross_attention     │ [(None, 11, 64),  │         64 │ gru_5[0][0],      │
│ (AdditiveAttention) │ (None, 11, 10)]   │            │ gru_4[0][0]       │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ merge_context       │ (None, 11, 128)   │          0 │ gru_5[0][0],      │
│ (Concatenate)       │                   │            │ cross_attention[… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ output_head (Dense) │ (None, 11, 20)    │      2,580 │ merge_context[0]… |
├─────────────────────┼───────────────────┼────────────┼───────────────────┤

Total params: 41,556 (162.33 KB)

Trainable params: 41,556 (162.33 KB)

Non-trainable params: 0 (0.00 B)
```

### Разбор TODO 4-13: модель

После заполнения блока проверьте всю цепочку внимания:
- encoder возвращает и `encoder_outputs`, и `encoder_state`;
- decoder работает с `initial_state=encoder_state`;
- `Attention([decoder_outputs, encoder_outputs])` получает запросы от decoder и память encoder;
- `analysis_model` отдаёт все промежуточные тензоры для диагностики.


```python
# Мини-проверка модели
sample_enc = enc_train[:2]
sample_dec = dec_in_train[:2]

encoder_outputs_s, decoder_outputs_s, context_s, attention_scores_s, probs_s = analysis_model.predict(
    [sample_enc, sample_dec],
    verbose=0,
)

assert model.output_shape == (None, DEC_LEN, VOCAB_SIZE)
assert encoder_outputs_s.shape[:2] == (2, ENC_LEN)
assert decoder_outputs_s.shape[:2] == (2, DEC_LEN)
assert context_s.shape == decoder_outputs_s.shape
assert attention_scores_s.shape == (2, DEC_LEN, ENC_LEN)
assert probs_s.shape == (2, DEC_LEN, VOCAB_SIZE)

row_sums = attention_scores_s.sum(axis=-1)
valid_decoder_positions = (sample_dec != PAD_ID)
assert np.allclose(row_sums[valid_decoder_positions], 1.0, atol=1e-5)
print('Мини-проверка модели: OK')
```

Вывод: `Мини-проверка модели: OK`

## Трассировка одного примера через модель
Проверка форм тензоров:
1. входы encoder и decoder;
2. выход encoder по всем входным позициям;
3. выход decoder по всем шагам;
4. контекст attention;
5. итоговое распределение вероятностей.

```python
sample_enc = enc_train[:1]
sample_dec = dec_in_train[:1]

encoder_outputs_s, decoder_outputs_s, context_s, attention_scores_s, probs_s = analysis_model.predict(
    [sample_enc, sample_dec],
    verbose=0,
)

print('Вход encoder_input      :', sample_enc.shape)
print('Вход decoder_input      :', sample_dec.shape)
print('После encoder GRU       :', encoder_outputs_s.shape)
print('После decoder GRU       :', decoder_outputs_s.shape)
print('После attention context :', context_s.shape)
print('attention_scores        :', attention_scores_s.shape)
print('После выходного слоя    :', probs_s.shape)
print('Суммы attention по encoder-оси:', np.round(attention_scores_s[0, :5].sum(axis=-1), 3))
print('Первые 3 предсказанных индекса:', probs_s.argmax(axis=-1)[0, :3])
```

Вывод:
```
sample_enc = enc_train[:1]
sample_dec = dec_in_train[:1]

encoder_outputs_s, decoder_outputs_s, context_s, attention_scores_s, probs_s = analysis_model.predict(
    [sample_enc, sample_dec],
    verbose=0,
)

print('Вход encoder_input      :', sample_enc.shape)
print('Вход decoder_input      :', sample_dec.shape)
print('После encoder GRU       :', encoder_outputs_s.shape)
print('После decoder GRU       :', decoder_outputs_s.shape)
print('После attention context :', context_s.shape)
print('attention_scores        :', attention_scores_s.shape)
print('После выходного слоя    :', probs_s.shape)
print('Суммы attention по encoder-оси:', np.round(attention_scores_s[0, :5].sum(axis=-1), 3))
print('Первые 3 предсказанных индекса:', probs_s.argmax(axis=-1)[0, :3])
```

## Обучение
В `fit` подается пара входов `[encoder_input, decoder_input]` и целевой тензор `decoder_target`.

Для контроля обобщения используется `validation_split=0.2` на обучающей части.

### Подсказка к TODO 14: обучение

Концептуальная подсказка:
- обучение идёт почти так же, как в `01-RNN / ЛР03`: меняется не вызов `fit`, а сама модель под капотом.

Implementation hint:
- передавайте в `fit` пару `[enc_train, dec_in_train]`;
- целевой тензор — `dec_tgt_train`;
- используйте `validation_split=0.2`, `batch_size=64` и аргумент `epochs`.

```python
def train_model(model, enc_train, dec_in_train, dec_tgt_train, epochs: int = 18):
    # TODO 14: обучите модель через model.fit
    history = model.fit(
        [enc_train, dec_in_train],
        dec_tgt_train,
        epochs=epochs,
        batch_size=64,
        validation_split=0.2
    )
    return history


history = train_model(model, enc_train, dec_in_train, dec_tgt_train)
```

Вывод:
```
Epoch 1/18
80/80 ━━━━━━━━━━━━━━━━━━━━ 2s 9ms/step - accuracy: 0.2991 - loss: 2.3556 - val_accuracy: 0.4357 - val_loss: 1.5608
Epoch 2/18
80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - accuracy: 0.4613 - loss: 1.4842 - val_accuracy: 0.5360 - val_loss: 1.3142
Epoch 3/18
80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 9ms/step - accuracy: 0.5572 - loss: 1.2606 - val_accuracy: 0.5999 - val_loss: 1.1014
Epoch 4/18
80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - accuracy: 0.6164 - loss: 1.0509 - val_accuracy: 0.6646 - val_loss: 0.8946
Epoch 5/18
80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - accuracy: 0.6779 - loss: 0.8437 - val_accuracy: 0.7428 - val_loss: 0.6944
Epoch 6/18
80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - accuracy: 0.7731 - loss: 0.6288 - val_accuracy: 0.8533 - val_loss: 0.4665
Epoch 7/18
...
Epoch 17/18
80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - accuracy: 0.9999 - loss: 0.0059 - val_accuracy: 0.9999 - val_loss: 0.0062
Epoch 18/18
80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - accuracy: 1.0000 - loss: 0.0048 - val_accuracy: 0.9999 - val_loss: 0.0052
```

### Разбор TODO 14: обучение

После заполнения блока проверьте, что вы не ждёте от history лишнего:
- Keras показывает `loss` и встроенную `accuracy`;
- `exact_match` по-прежнему считается отдельно вручную;
- преимущество attention обычно лучше видно на длинных последовательностях и строгой последовательной метрике.

```python
# Мини-проверка обучения
assert len(history.history['loss']) > 0
assert 'val_loss' in history.history
assert np.isfinite(history.history['loss']).all()
print('Финальный val_token_accuracy:', round(history.history['val_accuracy'][-1], 4))
print('Мини-проверка обучения: OK')
```

Вывод:
```
# Мини-проверка обучения
assert len(history.history['loss']) > 0
assert 'val_loss' in history.history
assert np.isfinite(history.history['loss']).all()
print('Финальный val_token_accuracy:', round(history.history['val_accuracy'][-1], 4))
print('Мини-проверка обучения: OK')
```

## Оценка и диагностика
Считаются две основные метрики:
- `token_accuracy` — доля корректных токенов;
- `exact_match` — доля последовательностей, где все значимые токены предсказаны верно.

### Подсказка к TODO 15: метрики

Концептуальная подсказка:
- `exact_match` считается точно так же, как в `01-RNN / ЛР03`: attention меняет архитектуру, но не логику строгой проверки всей последовательности.

Implementation hint:
- `preds = probs.argmax(axis=-1)`;
- `target = dec_tgt_test[:, :, 0]`;
- `mask = (target != PAD_ID)`;
- последовательность считается правильной только если совпали все значимые позиции.

```python
def evaluate_model(model, enc_test, dec_in_test, dec_tgt_test):
    loss, token_acc = model.evaluate([enc_test, dec_in_test], dec_tgt_test, verbose=0)
    probs = model.predict([enc_test, dec_in_test], verbose=0)
    preds = probs.argmax(axis=-1)
    target = dec_tgt_test[:, :, 0]
    mask = (target != PAD_ID)

    # TODO 15: вычислите exact_match только по значимым позициям
    exact_match = np.mean([p == r for p, r in zip(preds, target)])

    return {
        'loss': float(loss),
        'token_accuracy': float(token_acc),
        'exact_match': float(exact_match),
        'preds': preds,
        'target': target,
        'mask': mask,
        'probs': probs,
    }


metrics = evaluate_model(model, enc_test, dec_in_test, dec_tgt_test)
print({k: v for k, v in metrics.items() if k in ('loss', 'token_accuracy', 'exact_match')})
```

Вывод:
```
{'loss': 0.004457591101527214, 'token_accuracy': 0.9998863339424133, 'exact_match': 0.9998863636363636}
```

### Разбор TODO 15: метрики

После заполнения блока проверьте логику строгой оценки:
- PAD-позиции не должны участвовать в сравнении;
- `token_accuracy` и `exact_match` должны расходиться по смыслу, а не из-за ошибки формы;
- итоговая метрика усредняется по объектам, а не по токенам.

```python
# Мини-проверка метрик
assert 0.0 <= metrics['token_accuracy'] <= 1.0
assert 0.0 <= metrics['exact_match'] <= 1.0
if metrics['exact_match'] >= 0.85:
    print('Целевой порог достигнут: exact_match >= 0.85')
else:
    print('Целевой порог не достигнут: проверьте attention, teacher forcing и число эпох')
```

Вывод: `Целевой порог достигнут: exact_match >= 0.85`

