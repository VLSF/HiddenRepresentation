import numpy as np
from scipy.stats import expon, ortho_group

def low_rank_update(A_spectral, B_spectral, K):
  '''
  This function computes best rank M approximation of the sum two symmetric matrices of rank M, from their spectral decomposition.

  Parameters
  ----------
  A_spectral : list
    `A_spectral[0]` is the array of shape `(M1, )` that contains eigenvalues of the first matrix.
    `A_spectral[1]` is the array of shape `(N, M1)` that contains eigenvectors of the first matrix.
    Hence, the matrix itself can be represented as a sum of rank-1 matrices `A_spectral[0][i]*np.outer(A_spectral[1][:, i], A_spectral[1][:, i])`.
  B_spectral : list
    The second matrix in the same format. The shape can be `(N, M2)`
  K : int
    The rank of final matrix is `min(K, M1 + M2)`.

  Returns
  -------
  C_spectral : list
    Best rank-K approximation to the sum A+B in the same format as the input.
  '''
  A_val, A_vec = A_spectral
  B_val, B_vec = B_spectral
  AB = np.hstack([A_vec, B_vec])
  Q, R = np.linalg.qr(AB)
  N, M1 = A_vec.shape
  _, M2 = B_vec.shape
  RDRT = R @ np.block([[np.diag(A_val), np.zeros((M1, M2))], [np.zeros((M2, M1)), np.diag(B_val)]]) @ R.T
  val, vec = np.linalg.eigh(RDRT)
  return [val[-np.min([K, M1+M2]):], Q @ vec[:, -np.min([K, M1+M2]):]]

def probabilistic_iterative_method(mu_n, Sigma_n_spectral, L, A, b, wave_Sigma_n_spectral, wave_Psi_n_half):
  '''
  The function provides a single sweep of probabilistic iterative method with
  `wave_Sigma_n_spectral` and `wave_Psi_n_half` of general form.

  Parameters
  ----------
  mu_n : array (N, )
    Mean value of a resulting distribution.
  Sigma_n_spectral : list
    Spectral representation of the low-rank covariance matrix.
    `Sigma_n_spectral[0]` -- eigenvalues (shape `(M, )`), `Sigma_n_spectral[1]` eigenvectors (shape `(N, M)`)
  L : callable
    An iterative method of the form `x <- Mx + Nb`, that should be able process thin
    matrices `x` and `b`.
  A : array (N, N)
    The matrix we try to invert, i.e. we are solving Ax = b.
  b : array (N, )
    The right hand side.
  wave_Sigma_n_spectral : list
    Spectral representation of the first low-rank covariance matrix. The format is the same as for `Sigma_n_spectral`.
  wave_Psi_n_half : array (N, N)
    The second covariance matrix Psi_n = `wave_Psi_n_half.T @ wave_Psi_n_half`.

  Returns
  -------
  mu_n_ : array (N, )
    Mean value after the sweep.
  Sigma_n_spectral_ : list
    Spectral representation of the low-rank covariance matrix after the sweep.
  '''
  r_n = b - A @ mu_n
  mu_n_ = L(mu_n, b)
  C = [Sigma_n_spectral[0], L(Sigma_n_spectral[1], Sigma_n_spectral[1]*0)]
  beta_n = np.sum(Sigma_n_spectral[0]*(np.linalg.norm((wave_Psi_n_half @ A @ Sigma_n_spectral[1]), axis=0)**2))
  beta_n += np.linalg.norm(wave_Psi_n_half @ r_n)**2
  B = [wave_Sigma_n_spectral[0]*beta_n, wave_Sigma_n_spectral[1]]
  Sigma_n_spectral_ = low_rank_update(C, B, len(Sigma_n_spectral[0]))
  return mu_n_, Sigma_n_spectral_

def one_sweep_probabilistic_iterative_method(mu_n, Sigma_n_spectral, L, A, b, M):
  '''
  The function provides a single sweep of probabilistic iterative method with
  `wave_Sigma_n_spectral` and `wave_Psi_n_half` being zero.

  Parameters
  ----------
  mu_n : array (N, )
    Mean value of a resulting distribution.
  Sigma_n_spectral : list
    Spectral representation of the low-rank covariance matrix.
    `Sigma_n_spectral[0]` -- eigenvalues (shape `(M, )`), `Sigma_n_spectral[1]` eigenvectors (shape `(N, M)`)
  L : callable
    An iterative method of the form `x <- Mx + Nb`, that should be able process thin
    matrices `x` and `b`.
  A : array (N, N)
    The matrix we try to invert, i.e. we are solving Ax = b.
  b : array (N, )
    The right hand side.
  M : int
    A rank of update.

  Returns
  -------
  mu_n_ : array (N, )
    Mean value after the sweep.
  Sigma_n_spectral_ : list
    Spectral representation of the low-rank covariance matrix after the sweep.
  '''
  r_n = b - A @ mu_n
  mu_n_ = L(mu_n, b)
  C = [Sigma_n_spectral[0], L(Sigma_n_spectral[1], Sigma_n_spectral[1]*0)]
  Sigma_n_spectral_ = low_rank_update(C, [np.zeros(1), np.zeros_like(b).reshape(-1, 1)], M)
  return mu_n_, Sigma_n_spectral_

def abs_cos(error, subspace):
  '''
  Compute cosine of acute angle between a single vector `error` and a set of
  vectors defined by thin matrix `subspace`.

  Parameters
  ----------
  error : array(N, 1)
    First vector.
  subspace : array (N, M)
    Orthogonal vectors that span a subspace

  Returns
  -------
  cos : float64
    Cosine of the acute angle.
  '''
  error = error / np.linalg.norm(error)
  projected_error = subspace @ subspace.T @ error
  projected_error = projected_error / np.linalg.norm(projected_error)
  return abs(sum(error.reshape(-1, )*projected_error.reshape(-1, )))

def cosines(A_spectral, v):
  '''
  Function returns cosines of acute angles between vectors in `A_spectral[1]` and `v`.

  Parameters
  ----------
  A_spectral : list
    Spectral representation of matrix A.
    `A_spectral[0]` -- eigenvalues (shape `(M, )`), `A_spectral[1]` eigenvectors (shape `(N, M)`)
  v : array (N, )
    Vector for which we want to compute cosines.

  Returns
  -------
  Cos : array (M, )
    Cosines of specified angles.
  '''
  _, A_vec = A_spectral
  _, N = A_vec.shape
  Cos = []
  for i in range(N):
    cos = abs(np.sum(A_vec[:, i]*v))
    Cos.append(cos)
  return np.array(Cos)

def probabilistic_iterative_method_with_error_estimation(mu_n, Sigma_n_spectral, L, A, b, M, wave_Sigma_n_spectral, r0, scale=None):
  '''
  The function provides a single sweep of probabilistic iterative method with
  arbitrary `wave_Sigma_n_spectral` and `wave_Psi_n_half` taken to be `A^{-1}`.
  This choice introduces actual eror in the iteration process. To get rid of it
  we use estimation `|e|~(k(A)|x|/|b|)|r|`. In addition to that some vectors can
  be rejected.

  Parameters
  ----------
  mu_n : array (N, )
    Mean value of a resulting distribution.
  Sigma_n_spectral : list
    Spectral representation of the low-rank covariance matrix.
    `Sigma_n_spectral[0]` -- eigenvalues (shape `(M, )`), `Sigma_n_spectral[1]` eigenvectors (shape `(N, M)`)
  L : callable
    An iterative method of the form `x <- Mx + Nb`, that should be able process thin
    matrices `x` and `b`.
  A : array (N, N)
    The matrix we try to invert, i.e. we are solving Ax = b.
  b : array (N, )
    The right hand side.
  M : int
    A rank of update.
  wave_Sigma_n_spectral : list
    Spectral representation of the first low-rank covariance matrix. The format is the same as for `Sigma_n_spectral`.
  r0 : float
    L2 norm of residual `|b - Ax_0|`.
  reject : (optional), bool
    If `True` given `wave_Sigma_n_spectral` is ignored.
  scale : (optional), float or `None`
    In case the parameter is `None` it is ignored, in case it is a float function uses it to rescale new residual
    before correction of variance.

  Returns
  -------
  mu_n_ : array (N, )
    Mean value after the sweep.
  Sigma_n_spectral_ : list
    Spectral representation of the low-rank covariance matrix after the sweep.
  '''
  r_n = b - A @ mu_n
  mu_n_ = L(mu_n, b)
  C = [Sigma_n_spectral[0], L(Sigma_n_spectral[1], Sigma_n_spectral[1]*0)]
  e_n = np.linalg.norm(r_n)/r0
  if scale is not None:
    e_n *= scale
  beta_n = (e_n**2 + np.sum(Sigma_n_spectral[0]))/(len(Sigma_n_spectral[0]) + 1)
  B = [wave_Sigma_n_spectral[0]*beta_n, wave_Sigma_n_spectral[1]]
  Sigma_n_spectral_ = low_rank_update(C, B, M)
  return mu_n_, Sigma_n_spectral_

def random_static_UQ(A, b, L, N_it, M, x_exact, scale=None, start='zero'):
  '''
  Compute `N_it` of iterative method of choice along with the covariance matrix.
  The covariance matrix in this case has rank `M` and initialised by `M` random
  vectors sampled from standard normal distribution.

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
      x, Sigma_n_spectral = one_sweep_probabilistic_iterative_method(x, Sigma_n_spectral, L, A, b, M)
  return [error, residual, cos, trace], [x, Sigma_n_spectral]

def residual_UQ(A, b, L, N_it, M, x_exact, scale=None, start='zero'):
  '''
  Compute `N_it` of iterative method of choice along with the covariance matrix.
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
    x, Sigma_n_spectral = probabilistic_iterative_method_with_error_estimation(x, Sigma_n_spectral, L, A, b, M, wave_Sigma_n_spectral, r0, scale=scale)
  return [error, residual, cos, trace], [x, Sigma_n_spectral]

def residual_UQ_with_adaptive_rank(A, b, L, N_it, M, x_exact, scale=None, start='zero', alpha=0.1):
  '''
  Compute `N_it` of iterative method of choice along with the covariance matrix.
  The covariance matrix in this case has rank `M` and initialised by at most `M` first
  residual vectors. The number of vectors is determined adaptively based on the
  value of cosine of acute angle between new residual and eigenvectors of covariance
  matrix. (This strategy is not described in the article.)

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
  alpha : float64
    In case cos of acute angle between new residual and the space of eigenvectors of
    covariance matrix <= alpha, the vector is accepted, otherwise it is rejected.

  Returns
  -------
  mu_n_ : array (N, )
    Mean value after the sweep.
  Sigma_n_spectral_ : list
    Spectral representation of the low-rank covariance matrix after the sweep.
  '''
  error, residual, cos, trace, rank = np.zeros(N_it), np.zeros(N_it), np.zeros(N_it), np.zeros(N_it), np.zeros(N_it)
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
      rank[i] = len(Sigma_n_spectral[0])
    Cos = cosines(Sigma_n_spectral, r.reshape(-1)/norm_r)
    if sum(Cos<=alpha) >= 1 and len(Sigma_n_spectral[0])<=M:
      wave_Sigma_n_spectral = [np.ones(1), r.reshape(-1, 1)/norm_r]
    else:
      wave_Sigma_n_spectral = [np.zeros(1), 0*r.reshape(-1, 1)/norm_r]
    x, Sigma_n_spectral = probabilistic_iterative_method_with_error_estimation(x, Sigma_n_spectral, L, A, b, M, wave_Sigma_n_spectral, r0, scale=scale)
  return [error, residual, cos, trace, rank], [x, Sigma_n_spectral]

def random_matrix_expon(size, scale=1):
  '''
  Random matrix UDU^{T}, where U~O(N), p(d_{ii})~exp(-\alpha d_{ii})

  Parameters
  ----------
  size : int
    Number of variables.
  scale : float64
    The scale of exponential distribution.
  Returns
  -------
  A : array (N, N)
    Matrix that defines linear equation.
  b : array (N, 1)
    Right-hand side.
  x_exact : array (N, 1)
    Exact solution.

  '''
  val, vec, x_exact = expon(scale=scale).rvs(size=size), ortho_group.rvs(size), np.random.randn(size, 1)
  cond = np.max(val)/np.min(val)
  A = vec @ np.diag(val) @ vec.T
  b = A @ x_exact
  return A, b, x_exact
