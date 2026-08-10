"""
on_the_fly_data.py

Streaming (on-the-fly) mock-data generator for the spectral-reconstruction
supervised network, matching the training strategy described in
Kades et al., "Spectral Reconstruction with Deep Neural Networks"
(arXiv:1905.04305), Section III A and Appendix C:

  - Parameters (A, M, Gamma) for each Breit-Wigner peak are drawn uniformly
    at random from the bounds in Table I on EVERY batch, rather than from a
    fixed, pre-generated dataset saved to disk.
  - The number of Breit-Wigner peaks per sample is randomized between 1 and
    n_bw_max (paper: up to 3), so the network sees a mixture of 1-, 2-, and
    3-peak spectra during training, as described in Appendix C.
  - Noise is a fresh i.i.d. Gaussian draw per sample per batch (Eq. 8), not
    baked into a saved file.
  - The momentum-space Kallen-Lehmann kernel (Eq. 1) is used to map
    rho(omega) -> G(p), matching the paper's actual numerical setup
    (as opposed to the Euclidean-time/finite-T kernels used elsewhere
    in this codebase).

Because every batch is freshly sampled from a continuous parameter space,
the same (theta, epsilon) pair is essentially never repeated across
training -- this is what the paper means when it says the risk of
overfitting is "practically non-existent" (Sec. III A).
"""

import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from typing import Tuple, List, Dict


# ---------------------------------------------------------------------------
# Physics: kernel (Eq. 1) and Breit-Wigner spectral function (Eq. 7)
# ---------------------------------------------------------------------------

def KL_kernel_Position_Vacuum(
        Position, 
        Omega
        ):
    Position = Position[:, np.newaxis]  # Reshape Position as column to allow broadcasting
    ker = np.exp(-Omega * np.abs(Position)) + np.exp(-Omega*(len(Position)-Position))
    return ker

def KL_kernel_Position_FiniteT(
        Position, 
        Omega,
        T
        ):
    Position = Position[:, np.newaxis]  # Reshape Position as column to allow broadcasting
    with np.errstate(divide='ignore'):
        ker = np.cosh(Omega * (Position-1/(2*T))) / np.sinh(Omega/2/T)

        # set all entries in ker to 1 where Position is modulo 1/T and the entry is nan, because of numerical instability for large Omega
        ker[np.isnan(ker) & (Position % (1/T) == 0)] = 1
        #set all other nan entries to 0
        ker[np.isnan(ker)] = 0
    return ker

def KL_kernel_Omega(
        KL,
        x,
        Omega,
        args=[]
        ):
    ret=KL(x, Omega, *args)
    ret[:,Omega==0]=1
    ret=Omega * ret
    # set for all Omega=0 to 1
    if len(args)==0:
        ret[:,Omega==0]=0
    else:
        ret[:,Omega==0]=2*args[0]
    return ret

def Di(
        KL,
        rhoi,
        delomega
        ):
    # Ensure both tensors are of the same data type (float32)
    KL = tf.cast(KL, dtype=tf.float32)  # Cast KL to float32
    rhoi = tf.cast(rhoi, dtype=tf.float32)  # Cast rhoi to float32
    delomega = tf.cast(delomega, dtype=tf.float32)  # Cast delomega to float32

    # Perform matrix multiplication
    if rhoi.shape.rank == 1: # Ensure rhoi is 2D for multiplication
        rhoi = tf.expand_dims(rhoi, axis=0)
    
    dis = tf.matmul(KL, rhoi, transpose_b = True)  # Shape will be [Nt, batch_size]
    dis = tf.transpose(dis) #transpose to [batch_size, Nt]
    dis = dis * delomega  # Multiply by delomega
    return dis



def breit_wigner(w: np.ndarray, a: float, m: float, g: float) -> np.ndarray:
    return 4 * a * g * w / ((m ** 2 + g ** 2 - w ** 2) ** 2 + 4 * g ** 2 * w ** 2)


# ---------------------------------------------------------------------------
# Parameter volume 
# ---------------------------------------------------------------------------

@dataclass
class ParameterVolume:
    A: Tuple[float, float] = (0.1, 1.0)
    M: Tuple[float, float] = (0.5, 3.0)
    Gamma: Tuple[float, float] = (0.1, 0.4)
    delta_M: Tuple[float, float] = (0.0, 2.5)  # min/max mass separation, multi-peak only


VOL_O = ParameterVolume() 

# ---------------------------------------------------------------------------
# Streaming generator
# ---------------------------------------------------------------------------

class OnTheFlySpectralDataGenerator:
    """
    Draws fresh (rho, G_noisy, sigma) samples on every call instead of
    reading a fixed dataset from disk. Use `as_tf_dataset` for training
    and `sample_fixed_set` to build a reproducible, held-out test set
    (the paper does use a fixed set of 1000 samples per BW count for
    benchmarking -- it's only the *training* data that's generated
    on the fly).
    """

    def __init__(
            self,
            tau: np.ndarray,               # tau grid, shape [Ntau]
            omega: np.ndarray,           # omega grid, shape [Nomega]
            volume: ParameterVolume = VOL_O,
            n_bw_max: int = 3,
            noise_width: float = 1e-3,   # Eq. 8: fixed-width additive Gaussian noise
            seed: int = 0,
            ):
        self.tau = tau
        self.omega = omega
        self.delta_omega = omega[1] - omega[0]
        self.volume = volume
        self.n_bw_max = n_bw_max
        self.noise_width = noise_width
        self.rng = np.random.default_rng(seed)


        self.kernel = KL_kernel_Position_Vacuum(self.tau, self.omega) ### only for one kernel currently!!!

    def _sample_masses(self, n_bw: int) -> np.ndarray:
        """Draw n_bw masses honoring the min/max peak separation delta_M."""
        lo, hi = self.volume.M
        if n_bw == 1:
            return self.rng.uniform(lo, hi, size=1)

        dlo, dhi = self.volume.delta_M
        for _ in range(100):  # simple rejection sampling
            m1 = self.rng.uniform(lo, hi)
            masses = [m1]
            for _ in range(n_bw - 1):
                masses.append(masses[-1] + self.rng.uniform(dlo, dhi))
            masses = np.array(masses)
            if np.all(masses >= lo) and np.all(masses <= hi):
                return masses
        return np.clip(masses, lo, hi)  # fallback if rejection sampling struggles

    def _sample_one(self):
        n_bw = int(self.rng.integers(1, self.n_bw_max + 1))
        A = self.rng.uniform(*self.volume.A, size=n_bw)
        Gamma = self.rng.uniform(*self.volume.Gamma, size=n_bw)
        M = self._sample_masses(n_bw)

        rho = np.zeros_like(self.omega)
        for a, m, g in zip(A, M, Gamma):
            rho += breit_wigner(self.omega, a, m, g)

        G = Di(self.kernel, rho, self.delta_omega)                                    # [Eq. 2]
        G = np.squeeze(G.numpy(), axis=0)
        sigma = np.full_like(G, self.noise_width)
        eps = self.rng.normal(0.0, self.noise_width, size=G.shape)
        G_noisy = G + eps                                        # [Eq. 8]

        return rho.astype(np.float32), G_noisy.astype(np.float32), sigma.astype(np.float32)

    def generator(self):
        """Infinite generator of single (rho, G_noisy, sigma) samples."""
        while True:
            yield self._sample_one()

    def as_tf_dataset(self, batch_size: int) -> tf.data.Dataset:
        """
        Streaming tf.data.Dataset. Every batch is freshly sampled -- there is
        no underlying fixed-size dataset, so this is infinite and must be
        combined with `.take(steps_per_epoch)` (see note in networkTrainer
        integration below).
        """
        output_signature = (
            tf.TensorSpec(shape=(len(self.omega),), dtype=tf.float32),  # rho (X)
            tf.TensorSpec(shape=(len(self.tau),), dtype=tf.float32),      # G_noisy (y)
            tf.TensorSpec(shape=(len(self.tau),), dtype=tf.float32),      # sigma (z)
        )
        ds = tf.data.Dataset.from_generator(self.generator, output_signature=output_signature)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def sample_fixed_set(self, n_samples: int, seed: int = 0) -> List[Dict[str, np.ndarray]]:
        """
        Build a reproducible, fixed-size set (for validation/testing, or if
        you want a deterministic benchmark set like the paper's 1000-sample
        test sets per BW count in Appendix C). Uses its own seeded RNG so
        results don't depend on how many training batches were already drawn.
        """
        local = OnTheFlySpectralDataGenerator(
            self.tau, self.omega, self.volume, self.n_bw_max, self.noise_width, seed=seed
        )
        return [
            dict(zip(("fct", "corr", "noise"), local._sample_one()))
            for _ in range(n_samples)
        ]


if __name__ == "__main__":
    tau = np.linspace(0.0, 10.0, 100)   
    omega = np.linspace(0.0, 10.0, 500) 

    gen = OnTheFlySpectralDataGenerator(tau, omega, volume=VOL_O, n_bw_max=3,
                                         noise_width=1e-3, seed=0)

    train_dat = gen.as_tf_dataset(batch_size=128)
    for step, (rho, G_noisy, sigma) in enumerate(train_dat.take(3)):
        print(f"batch {step}: rho {rho.shape}, G {G_noisy.shape}, sigma {sigma.shape}")

    test_set = gen.sample_fixed_set(n_samples=1000, seed=42)
    print(f"fixed test set: {len(test_set)} samples, "
          f"e.g. fct shape {test_set[0]['fct'].shape}, corr shape {test_set[0]['corr'].shape}")