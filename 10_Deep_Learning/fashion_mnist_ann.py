"""Fashion-MNIST üzerinde küçük ANN sınıflandırıcısı."""
def train(epochs: int = 3) -> None:
    import tensorflow as tf
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(.25),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(x_train, y_train, epochs=epochs, batch_size=128, validation_split=.1)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print({"test_loss": round(loss, 4), "test_accuracy": round(accuracy, 4)})
    model.save("fashion_mnist.keras")

if __name__ == "__main__":
    train()
