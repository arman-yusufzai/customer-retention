import numpy as np
import tensorflow as tf
from scipy import sparse
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

from src.features import make_preprocessor


def dense(a):
    return a.toarray() if sparse.issparse(a) else np.asarray(a)


def build_nn(n_features, l2=1e-3, dropout=0.3, lr=1e-3, seed=42):
    tf.keras.utils.set_random_seed(seed)
    reg = tf.keras.regularizers.l2(l2)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=reg),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(16, activation="relu", kernel_regularizer=reg),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy")
    return model


def fit_nn(X_tr, y_tr, seed=42, **kw):
    """Fit preprocessing + network on X_tr. Early stopping uses a 10% slice of X_tr only."""
    pre = make_preprocessor()
    X_a, X_v, y_a, y_v = train_test_split(X_tr, y_tr, test_size=0.1,
                                          stratify=y_tr, random_state=seed)
    pre.fit(X_a)
    A, V = dense(pre.transform(X_a)), dense(pre.transform(X_v))
    model = build_nn(A.shape[1], seed=seed, **kw)
    early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                             restore_best_weights=True)
    hist = model.fit(A, y_a, validation_data=(V, y_v), epochs=200, batch_size=128,
                     callbacks=[early], verbose=0)
    return pre, model, hist


def predict_nn(pre, model, X):
    return model.predict(dense(pre.transform(X)), verbose=0).ravel()


def cv_nn(X, y, cv, **kw):
    rows = []
    for tr, te in cv.split(X, y):
        pre, model, hist = fit_nn(X.iloc[tr], y.iloc[tr], **kw)
        p_tr = predict_nn(pre, model, X.iloc[tr])
        p_te = predict_nn(pre, model, X.iloc[te])
        rows.append({"train_auc": roc_auc_score(y.iloc[tr], p_tr),
                     "auc": roc_auc_score(y.iloc[te], p_te),
                     "pr_auc": average_precision_score(y.iloc[te], p_te),
                     "epochs": len(hist.history["loss"])})
    return rows