"""Fashion-MNIST için tekrarlanabilir TensorFlow derin öğrenme deneyi."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

def set_reproducibility(seed: int = 42) -> None:
    import tensorflow as tf
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

def load_and_prepare(validation_size=5000):
    import tensorflow as tf
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_validation = x_train[-validation_size:]
    y_validation = y_train[-validation_size:]
    x_train = x_train[:-validation_size]
    y_train = y_train[:-validation_size]
    return (x_train, y_train), (x_validation, y_validation), (x_test, y_test)

def build_model(learning_rate=1e-3, dropout=.30):
    import tensorflow as tf
    inputs = tf.keras.layers.Input(shape=(28, 28), name="image")
    x = tf.keras.layers.Reshape((28, 28, 1))(inputs)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax", name="class_probability")(x)
    model = tf.keras.Model(inputs, outputs, name="fashion_mnist_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def callbacks(output: Path, patience=3):
    import tensorflow as tf
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=.5, patience=1, min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            output / "best_model.keras", monitor="val_accuracy", save_best_only=True
        ),
        tf.keras.callbacks.CSVLogger(output / "training_log.csv"),
    ]

def save_training_curves(history, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="validation")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="validation")
    axes[1].set_title("Accuracy"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "training_curves.png", dpi=170)
    plt.close(fig)

def save_prediction_examples(model, x_test, y_test, output: Path) -> None:
    probability = model.predict(x_test[:25], verbose=0)
    prediction = probability.argmax(axis=1)
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    for index, ax in enumerate(axes.flat):
        ax.imshow(x_test[index], cmap="gray")
        color = "green" if prediction[index] == y_test[index] else "red"
        ax.set_title(
            f"G:{CLASS_NAMES[y_test[index]]}\nT:{CLASS_NAMES[prediction[index]]}",
            fontsize=8, color=color,
        )
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output / "prediction_examples.png", dpi=170)
    plt.close(fig)

def run(epochs: int, batch_size: int, output: Path) -> None:
    import tensorflow as tf
    from sklearn.metrics import classification_report, confusion_matrix
    set_reproducibility()
    output.mkdir(parents=True, exist_ok=True)
    train, validation, test = load_and_prepare()
    x_train, y_train = train
    x_val, y_val = validation
    x_test, y_test = test
    model = build_model()
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=callbacks(output), verbose=2,
    )
    probability = model.predict(x_test, verbose=0)
    prediction = probability.argmax(axis=1)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    report = classification_report(
        y_test, prediction, target_names=CLASS_NAMES, output_dict=True
    )
    metrics = {
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
        "parameter_count": int(model.count_params()),
        "epochs_trained": len(history.history["loss"]),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, prediction).tolist(),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    model.save(output / "final_model.keras")
    save_training_curves(history, output)
    save_prediction_examples(model, x_test, y_test, output)
    print(f"Test accuracy: {accuracy:.4f} | Parametre: {model.count_params():,}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Fashion-MNIST CNN")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "outputs")
    args = parser.parse_args()
    run(args.epochs, args.batch_size, args.output)

if __name__ == "__main__":
    main()
