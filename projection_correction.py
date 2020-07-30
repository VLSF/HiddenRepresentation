import numpy as np
from UQ import one_sweep_probabilistic_iterative_method, abs_cos, probabilistic_iterative_method_with_error_estimation

def projection_correction(x0, A, b, Sigma_n_spectral):
  '''
  Straightforward implementation of low dimensional space given by `Sigma_n_spectral[1]`.

  Parameters
  ----------
  x0 : array (N, )
    Current estimation of solution.
  A : array (N, N)
    The matrix we try to invert, i.e. we are solving Ax = b.
  b : array (N, )
    The right hand side.
  Sigma_n_spectral : list
    Spectral representation of the low-rank covariance matrix.
    `Sigma_n_spectral[0]` -- eigenvalues (shape `(M, )`), `Sigma_n_spectral[1]` eigenvectors (shape `(N, M)`)

  Returns
  -------
  x : array (N, )
    Hopefully improved estimation of solution.
  '''
  B = Sigma_n_spectral[1].T @ A @ Sigma_n_spectral[1]
  r = b - A @ x0
  r_ = Sigma_n_spectral[1].T @ r
  x = x0 + Sigma_n_spectral[1] @ np.linalg.inv(B) @ r_
  return x

def corrected_random_static_UQ(A, b, L, N_it, M, x_exact, scale=None, start='zero', N_correction=50):
  '''
  Compute `N_it` of iterative method with projection correctio along with the
  covariance matrix. Projection correction is realised via standard Petrov-Galerkin
  condition. Where the subspace is chosen from the eigenvectors of covariance matrix.
  The covariance matrix in this case has rank `M` and initialised by `M` first
  residual vectors.

  Parameters
  ----------
  A : array (N, N)
    Matrix that represents linear operator
  b : array (N, 1)
    Right-hand side.
  L : callable
    An iterative method of the form `x <- Mx + Nb`, that should be able process thin
    matrices `x` and `b`.
  N_it : int
    Number of iterations to perform.
  M : int
    A rank of covariance matrix.
  x_exact : array (N, 1)
    Exact solution equations `A^{-1}b`
  scale : (optional), None or float64
    If None, no scale is applied to the covariance matrix, if float64 the covariance
    matrix (see the article for details) multiplied by this number.
  start : (optional), `'zero'` or `'random'`
    How to choose an initial guess. For `'zero'` the starting point is zero vector,
    for `'random'` the starting point is sampled from normal distribution.
  N_corrections : int
    Projection correction is applied each `N_corrections` iteration.

  Returns
  -------
  mu_n_ : array (N, )
    Mean value after the sweep.
  Sigma_n_spectral_ : list
    Spectral representation of the low-rank covariance matrix after the sweep.
  '''
  error, residual, cos, trace = np.zeros(N_it), np.zeros(N_it), np.zeros(N_it), np.zeros(N_it)
  N = len(b)
  if start == 'zero':
    x = np.zeros_like(b)
  else:
    x = np.random.randn(*b.shape)
  e0, r0 = np.linalg.norm(x - x_exact), np.linalg.norm(b - A @ x)
  for i in range(N_it):
    r = b - A @ x
    e = x - x_exact
    norm_r = np.linalg.norm(r)
    residual[i] = norm_r/r0
    error[i] = np.linalg.norm(e)/e0
    if i == 0:
      x = L(x, b)
    if i == 1:
      X = np.random.randn(N, M)
      X = X/np.linalg.norm(X, axis=0)
      if scale is None:
        norm = (np.linalg.norm(r)/r0)**2
      else:
        norm = (scale*np.linalg.norm(r)/r0)**2
      Sigma_n_spectral = [np.ones(M)*norm/M, X]
    if i>=1:
      cos[i] = abs_cos(e, Sigma_n_spectral[1])
      trace[i] = np.sqrt(np.sum(Sigma_n_spectral[0]))
      if i%N_correction == 0:
        x = projection_correction(x, A, b, Sigma_n_spectral)
      else:
        x, Sigma_n_spectral = one_sweep_probabilistic_iterative_method(x, Sigma_n_spectral, L, A, b, M)
  return [error, residual, cos, trace], [x, Sigma_n_spectral]

def corrected_residual_UQ(A, b, L, N_it, M, x_exact, scale=None, start='zero', N_correction=20):
  '''
  Compute `N_it` of iterative method with projection correctio along with the
  covariance matrix. Projection correction is realised via standard Petrov-Galerkin
  condition. Where the subspace is chosen from the eigenvectors of covariance matrix.
  The covariance matrix in this case has rank `M` and initialised by `M` first
  residual vectors.

  Parameters
  ----------
  A : array (N, N)
    Matrix that represents linear operator
  b : array (N, 1)
    Right-hand side.
  L : callable
    An iterative method of the form `x <- Mx + Nb`, that should be able process thin
    matrices `x` and `b`.
  N_it : int
    Number of iterations to perform.
  M : int
    A rank of covariance matrix.
  x_exact : array (N, 1)
    Exact solution equations `A^{-1}b`
  scale : (optional), None or float64
    If None, no scale is applied to the covariance matrix, if float64 the covariance
    matrix (see the article for details) multiplied by this number.
  start : (optional), `'zero'` or `'random'`
    How to choose an initial guess. For `'zero'` the starting point is zero vector,
    for `'random'` the starting point is sampled from normal distribution.
  N_corrections : int
    Projection correction is applied each `N_corrections` iteration.

  Returns
  -------
  mu_n_ : array (N, )
    Mean value after the sweep.
  Sigma_n_spectral_ : list
    Spectral representation of the low-rank covariance matrix after the sweep.
  '''
  error, residual, cos, trace = np.zeros(N_it), np.zeros(N_it), np.zeros(N_it), np.zeros(N_it)
  N = len(b)
  if start == 'zero':
    x = np.zeros_like(b)
  else:
    x = np.random.randn(*b.shape)
  e0, r0 = np.linalg.norm(x - x_exact), np.linalg.norm(b - A @ x)
  for i in range(N_it):
    r = b - A @ x
    e = x - x_exact
    norm_r = np.linalg.norm(r)
    residual[i] = norm_r/r0
    error[i] = np.linalg.norm(e)/e0
    if i == 0:
      if scale is None:
        norm = (norm_r/r0)**2
      else:
        norm = (scale*norm_r/r0)**2
      Sigma_n_spectral = [np.ones(1)*norm, r.reshape(-1, 1)/norm_r]
    if i >= 1:
      cos[i] = abs_cos(e, Sigma_n_spectral[1])
      trace[i] = np.sqrt(np.sum(Sigma_n_spectral[0]))
    if i < M:
      wave_Sigma_n_spectral = [np.ones(1), r.reshape(-1, 1)/norm_r]
    else:
      wave_Sigma_n_spectral = [np.zeros(1), 0*r.reshape(-1, 1)/norm_r]
    if i%N_correction == 0:
      x = projection_correction(x, A, b, Sigma_n_spectral)
    else:
      x, Sigma_n_spectral = probabilistic_iterative_method_with_error_estimation(x, Sigma_n_spectral, L, A, b, M, wave_Sigma_n_spectral, r0, scale=scale)
  return [error, residual, cos, trace], [x, Sigma_n_spectral]
