import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers,models
from tensorflow.keras.preprocessing.sequence import pad_sequences

num_words=10000
max_len=200

(x_train,y_train),(x_test,y_test)=tf.keras.datasets.imdb.load_data(num_words=num_words)
x_train=pad_sequences(x_train,maxlen=max_len)
x_test=pad_sequences(x_test,maxlen=max_len)

model=models.Sequential([
    layers.Embedding(num_words,32),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dense(16,activation='relu'),
    layers.Dense(1,activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history=model.fit(
    x_train,
    y_train,
    epochs=3,
    batch_size=128,
    validation_data=(x_test,y_test)
)

loss,acc=model.evaluate(x_test,y_test,verbose=2)
print(f"\n Test Accuracy:{acc:.4f}")

word_index=tf.keras.datasets.imdb.get_word_index()
reverse_word_index={
    value:key for key,value in word_index.items()
}

def decode_review(encoded):
    return " ".join([
        reverse_word_index.get(i-3,"?")
        for i in encoded
        if i>=3
    ])
    
index=np.random.randint(len(x_test))
prediction=model.predict(
    x_test[index:index+1],
    verbose=0
)[0][0]

print("\n Sample reviews: ")
print(decode_review(x_test[index]))

print("\n Actual Sentiment: ",
      "Positive" if y_test[index]==1 else "Negative")
print("\n Predicted Sentiment: ",
      "Positive" if prediction>=0.5 else "Negative")

plt.figure(figsize=(15,8))

plt.plot(
    history.history['accuracy'],
    label='Trained Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title('IMDB Sentiment Classification')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
