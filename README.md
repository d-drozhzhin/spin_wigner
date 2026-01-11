# Wigner transformation for $SU(2)$ group

Implementation is based on papers [Champion, Wang, Parker, & Blok](https://journals.aps.org/prx/abstract/10.1103/vbh4-lysv) and [Brif & Mann](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.59.971).


## $SU(2)$ group

$SU(2)$ is a Lie group with the following Lie algebra:

$$\{\hat{J}_x, \hat{J}_y, \hat{J}_z\}$$

$$\left[\hat{J}_a, \hat{J}_b\right] = \iota \varepsilon_{abc}~\hat{J}_c$$

Such group has infinite amount of unitary irreducible representations that are distinguished by index $j$.
Their Hilbert space $\mathcal{H}_j$ is spanned by the orthonormal basis $|j, j{-}n\rangle$, where $n = 0,1,2,\dots,2j$. Thus, it can be used to represent qudit with dimension $d=2j{+}1$.

Phase space is described with spherical angles $\theta$ and $\phi$.
In this document point on the sphere with $(\theta, \phi)$ is also denoted as $\Omega$.
And the displacement in phase space has the form:

$$\hat{D}(\Omega) = \hat{D}(\theta, \phi) = e^{-\iota \phi \hat{J}_z} ~ e^{-\iota \theta \hat{J}_y}$$


## Wigner transform for spin systems
Wigner transform of the density matrix is called the Wigner function.
It is defined as following:

$$W_{\hat{\rho}}(\Omega) = \text{Tr}\left( \hat{\rho} ~ \hat{\Delta}(\Omega)  \right)$$

Here:

$$\hat{\Delta}(\Omega) = \hat{D}(\Omega) \hat{M} \hat{D}^\dagger(\Omega)$$

is the Stratonovich-Weyl kernel and $\hat{M}$ - finite dimensional analogue of Parity operator:

$$\hat{M} = \sum_{n=0}^{2j+1} \sum_{l=0}^{2j} \frac {2l + 1} {2j + 1} ~ C_{j,j-n;l,0}^{j,j{-}n} ~ \ket{j,j{-}n} \bra{j,j{-}n}$$

$$\text{Tr}\left( \hat{M} \right) = 1 \qquad \text{Tr}\left( \hat{M}^2 \right) = 2j{+}1 = d$$

### Wigner transform integral:

$$\int_{X} d\mu(\Omega) W_{\hat{A}}(\Omega) = \text{Tr}\left( \hat{A} \right) \qquad d\mu(\Omega) = \frac{2j{+}1}{4\pi} ~ \sin\theta d\theta ~ d\phi$$


### Upper bound for Wigner transform
$$\Big| W_{\hat{A}}(\Omega) \Big|^2 = \left| \text{Tr}\left( \hat{A} ~ \hat{\Delta}(\Omega) \right) \right|^2 \le \text{Tr}\left( \hat{A}^2 \right) ~ \text{Tr}\left( \hat{\Delta}(\Omega)^2 \right) = \text{Tr}\left( \hat{A}^2 \right) ~ \text{Tr}\left( \hat{M}^2 \right) = \text{Tr}\left( \hat{A}^2 \right) ~ (2j{+}1)$$

$$\Big| W_{\hat{A}}(\Omega) \Big| \le \sqrt{ \text{Tr}\left( \hat{A}^2 \right) ~ (2j{+}1) }$$

$$\Big| W_{\hat{\rho}}(\Omega) \Big| \le \sqrt{ 2j{+}1 }$$

### Backward Wigner transform
$$\hat{A} = \int_{X} d\mu(\Omega) W_{\hat{A}}(\Omega) ~ \hat{\Delta}(\Omega)$$


## Wigner tomography

### Longtitudial angle $\phi$ tomography:

$$W(\theta, \phi) = \sum_{m_0,m_1=-j}^{+j} W_{m_0,m_1}(\theta) ~ e^{\iota (m_0-m_1) \phi} = \sum_{\mu=-2j}^{+2j} W_\mu(\theta) ~ e^{\iota \mu \phi}$$

$$\sum_{m=-j}^{+j} e^{\iota \mu \phi_m} e^{-\iota \mu' \phi_m} = (2j{+}1) ~ \delta_{\mu,\mu'} \qquad \phi_m = \phi_0 + \frac{2\pi}{2j{+}1}m$$

$$W_\mu(\theta) = \frac{1}{2j{+}1} \sum_{m=-j}^{+j} W(\theta, \phi_m) e^{-i \mu \phi_m} \qquad W_{-\mu}(\theta) = W_\mu^*(\theta)$$

Hence the integration trick emerges:

$$\int_0^{2\pi} W(\theta, \phi) ~ d\phi = \frac{2\pi}{2j{+}1} \sum_{m=-j}^{+j} W (\theta, \phi_m)$$

### Latitudial anlge $\theta$ tomography:

$$W(\theta, \phi) = \sum_{l_0,l_1=-j}^{+j} W_{l_0,l_1}(\phi) ~ e^{\iota (l_0-l_1) \theta} = \sum_{\lambda=0}^{2j} W_\lambda(\phi) ~ \cos(\lambda\theta)$$

$$\sum_{l=-j}^{+j} \cos(\lambda\theta_l) \cos(\lambda'\theta_l) = \frac{2j{+}1}{2{-}\delta_{\lambda,0}} ~ \delta_{\lambda,\lambda'} \qquad \theta_l = \frac{\pi}{2} + \frac{\pi}{2j{+}1}l$$

$$W_\lambda(\phi) = \frac{2{-}\delta_{\lambda,0}}{2j{+}1} \sum_{l=-j}^{+j} W(\theta_l, \phi) \cos(\lambda\theta_l)$$

Hence the integration trick emerges:

$$\int_0^{\pi} W(\theta, \phi) ~ \sin\theta d\theta = \frac{\pi}{2j{+}1} \sum_{l=-j}^{+j} W(\theta_l, \phi) K_{\lfloor j \rfloor}(\theta_l)$$

$$K_n(\tilde{\theta}) = \frac{4}{\pi} \sum_{\lambda=0}^{n} \frac{\cos(2\lambda ~ \tilde{\theta})}{1-4\lambda^2} - \frac{2}{\pi} = \frac{2}{\pi} \sum_{\lambda=-n}^{n} \frac{e^{\iota 2\lambda\tilde{\theta}}}{1-4\lambda^2}$$

$$\lim_{n\to\infty} K_n(\tilde{\theta}) = \sin\tilde{\theta}$$

### Overall tomography

$$W(\theta, \phi) = \sum_{\mu=-2j}^{+2j} \sum_{\lambda=0}^{2j} W_{\mu,\lambda} ~ \cos(\lambda\theta) ~ e^{\iota \mu \phi}$$

$$W_{\mu,\lambda} = \frac{2{-}\delta_{\lambda,0}}{(2j{+}1)^2} ~ \sum_{m=-j}^{+j} \sum_{l=-j}^{+j} W(\theta_l, \phi_m) \cos(\lambda\theta_l) e^{-\iota \mu \phi_m}$$
