"""Dense autoencoder (TensorFlow backend) for customer-embedding segmentation.

Optional dependency: importing this module is always safe; training raises a
business-readable install message when TensorFlow is absent. Reproducible via
seeded initializers, shuffle, and validation split; early stopping prevents
overfit and the training history is returned for the model card.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from segmentation.common import require_optional
from utils.exceptions import InsufficientDataError

MIN_SAMPLES = 200


@dataclass
class AutoencoderResult:
    """Trained encoder, Keras history dict, latent codes, and config."""

    encoder: object
    history: dict = field(default_factory=dict)
    latent: np.ndarray = field(default_factory=lambda: np.array([]))
    latent_dim: int = 8
    epochs_run: int = 0


def train_autoencoder(
    X: np.ndarray,
    latent_dim: int = 8,
    hidden: tuple[int, ...] = (64, 32),
    epochs: int = 100,
    batch_size: int = 256,
    patience: int = 10,
    validation_split: float = 0.15,
    random_state: int = 42,
) -> AutoencoderResult:
    """Train a symmetric dense autoencoder and return encoder + codes + history."""
    require_optional("tensorflow", "tensorflow")
    import tensorflow as tf

    X = np.asarray(X, dtype=np.float32)
    if len(X) < MIN_SAMPLES:
        raise InsufficientDataError(
            f"Autoencoder training needs at least {MIN_SAMPLES} customers "
            f"(only {len(X)} available). Use a classical model instead."
        )
    tf.keras.utils.set_random_seed(random_state)
    input_dim = X.shape[1]
    latent_dim = max(2, min(latent_dim, input_dim - 1 if input_dim > 2 else 1))

    inputs = tf.keras.Input(shape=(input_dim,))
    encoded = inputs
    for width in hidden:
        encoded = tf.keras.layers.Dense(
            width,
            activation="relu",
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=random_state),
        )(encoded)
    latent = tf.keras.layers.Dense(
        latent_dim,
        activation="linear",
        name="latent",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=random_state),
    )(encoded)
    decoded = latent
    for width in reversed(hidden):
        decoded = tf.keras.layers.Dense(
            width,
            activation="relu",
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=random_state),
        )(decoded)
    outputs = tf.keras.layers.Dense(input_dim, activation="linear")(decoded)

    autoencoder = tf.keras.Model(inputs, outputs)
    encoder = tf.keras.Model(inputs, latent)
    autoencoder.compile(optimizer="adam", loss="mse")
    stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )
    history = autoencoder.fit(
        X,
        X,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        shuffle=True,
        callbacks=[stop],
        verbose=0,
    )
    codes = np.asarray(encoder.predict(X, verbose=0))
    return AutoencoderResult(
        encoder=encoder,
        history={k: [float(v) for v in vals] for k, vals in history.history.items()},
        latent=codes,
        latent_dim=latent_dim,
        epochs_run=len(history.history.get("loss", [])),
    )
