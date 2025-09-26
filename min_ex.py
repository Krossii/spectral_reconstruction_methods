import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, initializers, optimizers
import matplotlib.pyplot as plt

# ---------------------------
# Utilities: Breit-Wigner generator (mock spectral functions)
# ---------------------------

def breit_wigner(omega, A, M, Gamma):
    omega = np.asarray(omega)
    num = 4.0 * A * Gamma * omega
    den = (M*M + Gamma*Gamma - omega*omega)**2 + (4.0 * (Gamma**2) * (omega**2))
    return num / (den + 1e-16)


def random_spectrum(omega_grid, n_bws=1, ranges=None):
    """Build sum of n_bws Breit-Wigner peaks sampling parameters uniformly.
    ranges: dict with 'A', 'M', 'Gamma' ranges each as (min,max).
    """
    if ranges is None:
        ranges = {'A':(0.1,1.0), 'M':(0.5,3.0), 'Gamma':(0.05,0.4)}
    spec = np.zeros_like(omega_grid)
    for _ in range(n_bws):
        A = np.random.uniform(*ranges['A'])
        M = np.random.uniform(*ranges['M'])
        G = np.random.uniform(*ranges['Gamma'])
        spec += breit_wigner(omega_grid, A, M, G)
    # ensure non-negative and finite
    spec = np.nan_to_num(spec, nan=0.0, posinf=1e6, neginf=0.0)
    spec[spec < 0] = 0.0
    return spec

# ---------------------------
# Kernel builder
# Two modes supported:
#  - 'momentum': K(p, omega) = omega / (pi (omega^2 + p^2)) * d_omega
#  - 'position': K(t, omega) = exp(-omega * t) * d_omega   (zero-temperature Euclidean time)
# ---------------------------

def build_kernel(p_or_t_grid, omega_grid, mode='momentum'):
    """
    p_or_t_grid: numpy array of shape (N_pts,) representing either momenta p or times t
    omega_grid: numpy array (N_omega,)
    mode: 'momentum' or 'position'
    returns: kernel matrix of shape (N_pts, N_omega)
    """
    omega_grid = np.asarray(omega_grid, dtype=np.float64)
    d_omega = float(omega_grid[1] - omega_grid[0]) if omega_grid.size > 1 else 1.0

    if mode == 'momentum':
        p_grid = np.asarray(p_or_t_grid, dtype=np.float64)
        # K(p, w) = w / (pi (w^2 + p^2)) * d_omega
        W = omega_grid[np.newaxis, :] / (np.pi * (omega_grid[np.newaxis, :]**2 + p_grid[:, np.newaxis]**2))
        K = W * d_omega
    elif mode == 'position':
        t_grid = np.asarray(p_or_t_grid, dtype=np.float64)
        # K(t, w) = exp(-w t) * d_omega
        # ensure t >= 0
        if np.any(t_grid < 0):
            raise ValueError('All t must be >= 0 for zero-temperature kernel')
        K = np.exp(-np.outer(t_grid, omega_grid)) * d_omega
    else:
        raise ValueError("mode must be 'momentum' or 'position'")

    # guard against NaNs/Infs
    K = np.nan_to_num(K, nan=0.0, posinf=1e6, neginf=0.0).astype(np.float32)
    return K

# ---------------------------
# PoNet FC builder (stabilized)
# - Uses He normal initializer
# - Adds BatchNormalization after big dense layers
# - Optional dropout
# - Smaller parameter defaults can be used to avoid OOM/NANs
# ---------------------------

def build_ponet_fc(input_dim=100, output_dim=500,
                   center_variant='paper',
                   dropout_rate=0.0,
                   l2_reg=1e-6):
    """
    center_variant: 'paper' uses the sizes from Table III (large).
                    'small' uses much smaller layers for stability.
    Returns a compiled tf.keras Model (not compiled here to keep flexibility).
    """
    init = initializers.HeNormal()
    reg = regularizers.l2(l2_reg) if l2_reg else None

    inp = layers.Input(shape=(input_dim,), name='propagator_input')
    x = layers.Activation('relu')(inp)

    if center_variant == 'paper':
        # sizes from Table III: might be very large -> risk of NaNs / OOM
        sizes = [6700, 12168, 1024]
    else:
        # a smaller architecture
        sizes = [2048, 4096, 1024]

    for i, s in enumerate(sizes):
        x = layers.Dense(s, kernel_initializer=init, kernel_regularizer=reg, name=f'fc_{s}')(x)
        # batchnorm helps numerical stability
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)

    out = layers.Dense(output_dim, activation='linear', kernel_initializer=init, name='rho_output')(x)

    model = models.Model(inputs=inp, outputs=out, name='PoNet_FC_stable')
    return model

# ---------------------------
# Loss helpers: spectral MSE and propagator MSE using kernel matrix
# Improvements done:
# - Cast kernel to tf.constant(float32)
# - Use tf.math.reduce_mean on safe finite tensors
# - Clip y_pred to reasonable range when computing propagator to avoid Inf
# ---------------------------

def spectral_mse(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    se = tf.math.squared_difference(y_true, y_pred)
    se = tf.where(tf.math.is_finite(se), se, tf.zeros_like(se))
    return tf.reduce_mean(se)


def make_propagator_mse_loss(kernel_np):
    K = tf.constant(kernel_np.astype(np.float32))  # (N_pts, Nw)

    def propagator_mse(y_true_rho, y_pred_rho):
        # enforce float32
        y_true_rho = tf.cast(y_true_rho, tf.float32)
        y_pred_rho = tf.cast(y_pred_rho, tf.float32)
        # clip predictions to avoid huge values
        y_pred_rho = tf.clip_by_value(y_pred_rho, 0.0, 1e6)

        # G = rho @ K^T  -> (batch, N_pts)
        G_pred = tf.matmul(y_pred_rho, tf.transpose(K))
        G_true = tf.matmul(y_true_rho, tf.transpose(K))

        diff = tf.math.squared_difference(G_true, G_pred)
        diff = tf.where(tf.math.is_finite(diff), diff, tf.zeros_like(diff))
        return tf.reduce_mean(diff)

    return propagator_mse


def combined_loss(alpha, kernel_np):
    prop_loss = make_propagator_mse_loss(kernel_np)
    def loss(y_true, y_pred):
        return spectral_mse(y_true, y_pred) + alpha * prop_loss(y_true, y_pred)
    return loss

# ---------------------------
# Dataset creation
# ---------------------------

def generate_dataset(n_samples=20000, Np=100, Nw=500, max_bws=3, noise_sigma=1e-4, seed=123,
                     kernel_mode='momentum'):
    np.random.seed(seed)
    # grids
    p_or_t_grid = np.arange(Np)
    omega_grid = np.linspace(1e-4, 10.0, Nw)   # start > 0 to avoid exact zeros
    K = build_kernel(p_or_t_grid, omega_grid, mode=kernel_mode)  # shape (Np, Nw)

    X = np.zeros((n_samples, Np), dtype=np.float32)   # propagators
    Y = np.zeros((n_samples, Nw), dtype=np.float32)   # spectra (ground truth)

    for i in range(n_samples):
        nbw = np.random.randint(1, max_bws+1)
        rho = random_spectrum(omega_grid, n_bws=nbw)
        G = K @ rho
        # add gaussian noise to propagator (smaller sigma to avoid instability)
        G_noisy = G + np.random.normal(scale=noise_sigma, size=G.shape)
        # clip and nan-safe
        G_noisy = np.nan_to_num(G_noisy, nan=0.0, posinf=1e6, neginf=0.0)
        X[i,:] = G_noisy.astype(np.float32)
        Y[i,:] = rho.astype(np.float32)

    # optional normalization: scale X to unit std per feature to help training
    mean_X = np.mean(X, axis=0, keepdims=True)
    std_X = np.std(X, axis=0, keepdims=True) + 1e-12
    X = (X - mean_X) / std_X

    return (X, Y, p_or_t_grid, omega_grid, K)

# ---------------------------
# Example training run
# ---------------------------
if __name__ == '__main__':
    # hyperparams 
    Np = 100
    Nw = 500 
    n_train = 50000
    n_val = 10000
    batch_size = 64
    n_epochs = 100
    alpha = 1.0         # much smaller alpha to avoid huge gradients
    lr = 1e-5           # smaller learning rate
    clipnorm = 1.0

    # ---- choose kernel mode: 'momentum' or 'position' ----
    kernel_mode = 'position'

    # generate dataset
    X, Y, p_or_t_grid, omega_grid, K = generate_dataset(n_samples=n_train+n_val,
                                                         Np=Np, Nw=Nw, max_bws=1,
                                                         noise_sigma=1e-4, seed=42,
                                                         kernel_mode=kernel_mode)
    X_train, X_val = X[:n_train], X[n_train:]
    Y_train, Y_val = Y[:n_train], Y[n_train:]

    # build model (use 'small' variant for stability; switch to 'paper' if you have large GPU)
    model = build_ponet_fc(input_dim=Np, output_dim=Nw, center_variant='small', dropout_rate=0.05)
    model.summary()

    # compile with combined loss L_rho + alpha L_G
    loss_fn = combined_loss(alpha=alpha, kernel_np=K)
    opt = optimizers.Adam(learning_rate=lr, clipnorm=clipnorm)
    model.compile(optimizer=opt, loss=loss_fn, metrics=[spectral_mse])

    # add callbacks
    cb = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-8),
          tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)]

    # train
    hist = model.fit(X_train, Y_train,
                     validation_data=(X_val, Y_val),
                     epochs=n_epochs,
                     batch_size=batch_size,
                     callbacks=cb)

    # evaluate on a sample
    idx = np.random.randint(0, X_val.shape[0])
    x_sample = X_val[idx:idx+1]
    rho_true = Y_val[idx]
    rho_pred = model.predict(x_sample)[0]

    # plot comparison
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.plot(omega_grid, rho_true, label='true rho', linewidth=1.2)
    plt.plot(omega_grid, rho_pred, label='pred rho', linewidth=1.0, alpha=0.8)
    plt.legend()
    plt.xlabel('omega')
    plt.ylabel('rho(omega)')
    plt.title('PoNet FC reconstruction example (stable)')
    plt.subplot(1,3,2)
    plt.plot(hist.history['loss'], label='train loss')
    plt.plot(hist.history['val_loss'], label='val loss')
    plt.yscale('log')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Training history')
    plt.subplot(1,3,3)
    plt.scatter(p_or_t_grid, x_sample[0], label='input G', marker = 'x', color='orange', s=10)
    plt.scatter(p_or_t_grid, K @ rho_pred, label='reconstructed G', marker = 'x', color='green', s=10, alpha=0.7)
    plt.xlabel('t' if kernel_mode=='position' else 'p')
    plt.ylabel('G')
    plt.legend()
    plt.title('Input propagator')
    plt.tight_layout()
    plt.savefig('min_ex.png', dpi=300)