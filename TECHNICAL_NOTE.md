# Numerical and modeling notes

The observation sensitivity for parameters `(tau, phi)` is

```text
J_k = [omega0 + a * (t_k + tau), 1].
```

With known independent noise standard deviation `sigma`, physical parameter
scales `D = diag(0.1 s, 0.1 rad)`, and `p = D u`, the dimensionless whitened
Jacobian is `A = J D / sigma`. Singular vectors refer to `u` coordinates. The
numerical-rank threshold is `s_i > 1e-10 * s_max`; constrained directions below
`1e-2 * s_max` are also labeled weak. Both are reporting policies. The supplied
scales matter to conditioning and these thresholds. Changing the unit of tau
from seconds to milliseconds, together with its scale, preserves `A`.

## Covariance without squared conditioning

An early calculation inverted `J.T @ J`. At acceleration `a = 1e-8`, that route
became unreliable while the scaled Jacobian still passed the numerical-rank
test. Forming the normal matrix squares the condition number; checking rank
with the Jacobian and then trusting a normal-matrix inverse was inconsistent.

Both fitting and segment selection now use the same SVD calculation:

```text
A = U diag(s) V.T
B = D V diag(1/s)
Cov(p) = B B.T
log det(A.T A) = 2 sum(log(s)).
```

The covariance helper refuses numerically deficient input. It does not return a
pseudoinverse as a finite individual-parameter covariance. Full numerical rank
still permits very poor precision: at `a = 1e-4 rad/s²`, timing standard
deviation is 14.2566 s despite rank 2. That check uses noiseless observations to
evaluate the information at the injected parameters, while retaining the known
measurement-noise standard deviation of 0.03 rad in the covariance calculation.

For `a != 0` and nonzero sample-time spread, timing variance has an independent
closed form:

```text
Var(tau) = sigma² / (a² * sum((t - mean(t))²)).
```

This formula checks the small-acceleration result and every candidate subset.
It also explains why maximizing the information determinant at fixed sample
count is equivalent to maximizing time spread in this particular model.

## What is exact and what is local

Subtracting the known nominal reference gives

```text
z - theta(t) = (a * tau) * t +
              (omega0 * tau + 0.5 * a * tau² + phi) + epsilon.
```

For nonzero acceleration, ordinary slope/intercept regression yields tau and
phi through this explicit transformation. The implementation cross-checks it
against nonlinear least squares. Timing is linear in the estimated slope, so
its variance formula above is exact under the stated Gaussian model. The full
`(tau, phi)` covariance from the Jacobian is a local approximation, since phi is
a nonlinear function of the fitted slope. Large timing uncertainty should not
be interpreted as a tight joint Gaussian approximation in those parameters.

At constant rotation, only `omega0 * tau + phi` is estimable. Its MLE is the mean
of `z - theta(t)`, with standard deviation `sigma / sqrt(n)`. The displayed ridge
uses this MLE. The injected combination differs because of the fixed noise draw.

The figure's contour levels are ΔNLL = 1, 3, 6 relative to the minimum in each
panel. They are not labeled as confidence percentages. Reusing the noise draw
across panels makes the change in motion explicit without changing the sampled
measurement errors.
