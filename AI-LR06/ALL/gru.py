import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_seed(42)
print('TensorFlow version:', tf.__version__)


PAD_ID = 0
SOS_ID = 10
EOS_ID = 12
VOCAB_SIZE = 18
ENC_LEN = 6
DEC_LEN = ENC_LEN + 1


def make_one_sample(min_len: int = 3, max_len: int = ENC_LEN):
    length = np.random.randint(min_len, max_len + 1)
    seq = np.random.randint(1, 10, size=length, dtype=np.int32)
    # TODO 1: получите обратную последовательность
    rev = seq[::-1]
    return seq, rev


def pad_sequence(seq, target_len, pad_value=PAD_ID):
    out = np.full((target_len,), pad_value, dtype=np.int32)
    out[:len(seq)] = seq
    return out


def make_dataset(n_samples: int = 7000):
    encoder_input = np.zeros((n_samples, ENC_LEN), dtype=np.int32)
    decoder_input = np.zeros((n_samples, DEC_LEN), dtype=np.int32)
    decoder_target = np.zeros((n_samples, DEC_LEN), dtype=np.int32)

    for i in range(n_samples):
        seq, rev = make_one_sample()
        enc = pad_sequence(seq, ENC_LEN)

        # TODO 2: сформируйте вход декодера как [SOS] + rev
        dec_in = [SOS_ID] + rev.tolist()
        # TODO 3: сформируйте целевой выход декодера как rev + [EOS]
        dec_out = rev.tolist() + [EOS_ID]

        encoder_input[i] = pad_sequence(enc, ENC_LEN)
        decoder_input[i] = pad_sequence(dec_in, DEC_LEN)
        decoder_target[i] = pad_sequence(dec_out, DEC_LEN)

    return encoder_input, decoder_input, decoder_target[..., None]


enc_in, dec_in, dec_tgt = make_dataset()

enc_train, enc_temp, dec_in_train, dec_in_temp, dec_tgt_train, dec_tgt_temp = train_test_split(
    enc_in, dec_in, dec_tgt, test_size=0.3, random_state=42
)
enc_val, enc_test, dec_in_val, dec_in_test, dec_tgt_val, dec_tgt_test = train_test_split(
    enc_temp, dec_in_temp, dec_tgt_temp, test_size=0.5, random_state=42
)

print('Train:', enc_train.shape, dec_in_train.shape, dec_tgt_train.shape)
print('Val  :', enc_val.shape, dec_in_val.shape, dec_tgt_val.shape)
print('Test :', enc_test.shape, dec_in_test.shape, dec_tgt_test.shape)

# Mini-check: формы и сдвиг decoder
assert enc_train.ndim == 2 and dec_in_train.ndim == 2
assert dec_tgt_train.ndim == 3 and dec_tgt_train.shape[2] == 1

sample_idx = 0
print('demo decoder_input :', dec_in_train[sample_idx])
print('demo decoder_target:', dec_tgt_train[sample_idx, :, 0])
print('Mini-check данных пройден.')

def build_model(vocab_size: int = VOCAB_SIZE, emb_dim: int = 16, latent_dim: int = 48):
    encoder_inputs = tf.keras.layers.Input(shape=(ENC_LEN,), name='encoder_inputs')
    # TODO 4: embedding для encoder
    enc_emb = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=emb_dim,
        input_length=ENC_LEN,
        name='encoder_embedding'
    )(encoder_inputs)
    # TODO 5: GRU encoder с return_state=True
    _, state = tf.keras.layers.GRU(
        latent_dim,
        return_state=True, 
        name='encoder_gru'
    )(enc_emb)

    decoder_inputs = tf.keras.layers.Input(shape=(DEC_LEN,), name='decoder_inputs')
    # TODO 6: embedding для decoder
    dec_emb = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=emb_dim,
        input_length=DEC_LEN,
        name='decoder_embedding'
    )(decoder_inputs)
    # TODO 7: GRU decoder с initial_state=state и return_sequences=True
    dec_out = tf.keras.layers.GRU(
        latent_dim,
        return_sequences=True,
        name='decoder_gru'
    )(dec_emb, initial_state=state)

    # TODO 8: Dense + softmax для распределения по словарю
    logits = tf.keras.layers.Dense(
        vocab_size,
        activation='softmax',
        name='dense_out'
    )(dec_out)

    model = tf.keras.Model([encoder_inputs, decoder_inputs], logits)
    # TODO 9: compile (adam, sparse_categorical_crossentropy, accuracy)
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return model


model = build_model()
model.summary()

# Mini-check: выход модели
assert len(model.output_shape) == 3
assert model.output_shape[-1] == VOCAB_SIZE
print('Mini-check модели пройден.')

def train_model(model, enc_train, dec_in_train, dec_tgt_train, enc_val, dec_in_val, dec_tgt_val, epochs: int = 18):
    # TODO 10: обучите модель с validation_data и batch_size=64
    history = model.fit(
        [enc_train, dec_in_train], dec_tgt_train,
        batch_size=64,
        epochs=epochs,
        validation_data=([enc_val, dec_in_val], dec_tgt_val)
    )
    return history


history = train_model(
    model,
    enc_train,
    dec_in_train,
    dec_tgt_train,
    enc_val,
    dec_in_val,
    dec_tgt_val,
)

# Mini-check: история обучения
for key in ['loss', 'val_loss', 'accuracy', 'val_accuracy']:
    assert key in history.history, f'В history нет ключа {key}'
print('Mini-check обучения пройден.')


def evaluate_model(model, enc_test, dec_in_test, dec_tgt_test):
    loss, token_acc = model.evaluate([enc_test, dec_in_test], dec_tgt_test, verbose=0)
    probs = model.predict([enc_test, dec_in_test], verbose=0)
    preds = probs.argmax(axis=-1)
    target = dec_tgt_test[:, :, 0]
    mask = (target != PAD_ID)
    # TODO 11: exact match по значимым токенам (target != PAD_ID)
    exact_match = np.mean(np.all((preds == target) | ~mask, axis=1))
    
    return {
        'loss': float(loss),
        'token_accuracy': float(token_acc),
        'exact_match': float(exact_match),
        'preds': preds,
        'target': target,
    }


metrics = evaluate_model(model, enc_test, dec_in_test, dec_tgt_test)
print({k: v for k, v in metrics.items() if k in ('loss', 'token_accuracy', 'exact_match')})

# Mini-check: формы и порог
assert metrics['preds'].shape == metrics['target'].shape
if metrics['exact_match'] >= 0.80:
    print('Целевой порог достигнут: exact_match >= 0.80')
else:
    print('Порог не достигнут. Проверьте сдвиг decoder_input/decoder_target и маску PAD.')

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

for i in range(3):
    print('---')
    print('encoder_input :', enc_test[i])
    print('target       :', metrics['target'][i])
    print('pred         :', metrics['preds'][i])
