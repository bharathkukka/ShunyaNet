# python
import tensorflow as tf


def build_model(input_dim: int = 32 * 32 * 3, num_classes: int = 10,
                hidden_units: int = 256, dropout_rate: float = 0.0) -> tf.keras.Model:

    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(hidden_units, activation="relu"),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(num_classes)  # logits; use from_logits=True
    ])


def load_cifar10_flat():
    """Loads CIFAR-10 and flattens images to vectors."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = (x_train.astype("float32") / 255.0).reshape((x_train.shape[0], -1))
    x_test = (x_test.astype("float32") / 255.0).reshape((x_test.shape[0], -1))
    y_train = y_train.squeeze().astype("int64")
    y_test = y_test.squeeze().astype("int64")
    return (x_train, y_train), (x_test, y_test)


def main():
    (x_train, y_train), (x_test, y_test) = load_cifar10_flat()

    model = build_model(hidden_units=512, dropout_rate=0.2)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=10,
        validation_split=0.1,
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
