"""
SSW a Stochastic Steepest Weights para OtimizaAAo Vetorial
============================================================

ImplementaAAo do algoritmo SSW proposto por SchAffler, Schultz e
Weinzierl (JOTA, 2002), discretizado pelo esquema de EuleraMaruyama.

A classe herda de ``pymoo.core.algorithm.Algorithm`` e pode ser
utilizada com a interface funcional ``minimize`` ou com o padrAo
ask/tell do pymoo.

ReferAancias
-----------
[1] SchAffler, S., Schultz, R., Weinzierl, K. (2002).
    Stochastic method for the solution of unconstrained vector
    optimization problems. JOTA, 114(1), 209a222.
[2] Blank, J. and Deb, K. (2020). pymoo: Multi-objective
    Optimization in Python. IEEE Access, 8, 89497a89509.
[3] Santos, T. and Xavier, S. (2025). guiPymoo: A Graphical
    Framework for Reproducible Multi-Objective Optimization
    Experiments. arXiv preprint (under review).
"""

from util.array_backend import xp as np

from core.algorithm import Algorithm
from core.population import Population
from core.individual import Individual
from util.display.multi import MultiObjectiveOutput
from util.optimum import filter_optimum

ALGORITHM_FLAGS = {
    "SSW": {"multi", "many"},
}


def _force_cpu_backend(algorithm, reason: str) -> None:
    if not bool(getattr(algorithm, "use_gpu", False)):
        return
    algorithm.use_gpu = False
    algorithm.array_backend_effective = "numpy"
    state = dict(getattr(algorithm, "backend_state", {}) or {})
    state["forced_cpu_reason"] = str(reason)
    algorithm.backend_state = state


# ---------------------------------------------------------------------------
# CAlculo da direAAo de descida q(x)
# ---------------------------------------------------------------------------

def _project_to_simplex(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=float)
    n = x.size
    if n == 1:
        return np.array([1.0], dtype=float)

    u = np.sort(x)[::-1]
    cssv = np.cumsum(u) - 1.0
    idx = np.arange(1, n + 1, dtype=float)
    cond = u - cssv / idx > 0.0
    if not np.any(cond):
        return np.full(n, 1.0 / n, dtype=float)

    rho = int(np.where(cond)[0][-1])
    theta = cssv[rho] / float(rho + 1)
    w = np.maximum(x - theta, 0.0)
    s = float(np.sum(w))
    if s <= 1e-16:
        return np.full(n, 1.0 / n, dtype=float)
    return w / s


def _solve_simplex_qp(H: np.ndarray, max_iter: int = 250, tol: float = 1e-10) -> np.ndarray:
    m = H.shape[0]
    if m == 1:
        return np.array([1.0], dtype=float)

    alpha = np.full(m, 1.0 / m, dtype=float)
    lipschitz = float(np.linalg.norm(H, ord=np.inf))
    step = 1.0 / max(lipschitz, 1e-12)

    for _ in range(max_iter):
        prev = alpha.copy()
        g = H @ alpha
        alpha = _project_to_simplex(alpha - step * g)
        if float(np.linalg.norm(alpha - prev)) <= tol:
            break
    return alpha

def _compute_q(jac: np.ndarray) -> np.ndarray:
    """Resolve o sub-problema quadrAtico no simplex para obter q(x).

    Dado o Jacobiano J a a^{mAn} (linhas = af_i(x)^T), resolve

        min_{I a I^m}  a I I_i af_i(x) aA2  =  I^T (J Ja) I

    e retorna q(x) = Ja I*.

    ParAmetros
    ----------
    jac : np.ndarray, forma (m, n)
        Jacobiano avaliado no ponto corrente.

    Retorna
    -------
    q : np.ndarray, forma (n,)
        DireAAo de descida comum a todos os objetivos.
    """
    m, n = jac.shape
    H = jac @ jac.T                         # Gramiano mAm

    alpha_star = _solve_simplex_qp(H)
    q = jac.T @ alpha_star                  # q(x) = I I*_i af_i(x)
    return q


# ---------------------------------------------------------------------------
# Jacobiano por diferenAas finitas
# ---------------------------------------------------------------------------

def _finite_diff_jacobian(func, x, h=1e-7):
    """Jacobiano via diferenAas finitas centradas.

    ParAmetros
    ----------
    func : callable
        FunAAo vetorial f: a^n a a^m.
    x : np.ndarray, forma (n,)
        Ponto de avaliaAAo.
    h : float
        Tamanho do passo.

    Retorna
    -------
    J : np.ndarray, forma (m, n)
    """
    n = x.shape[0]
    f0 = np.asarray(func(x))
    m = f0.shape[0]
    J = np.empty((m, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = h
        J[:, j] = (np.asarray(func(x + e)) - np.asarray(func(x - e))) / (2.0 * h)
    return J


# ---------------------------------------------------------------------------
# Classe principal: SSW para pymoo
# ---------------------------------------------------------------------------

class SSW(Algorithm):
    """Algoritmo SSW (Stochastic Steepest Weights) para pymoo.

    Implementa a dinAmica de EuleraMaruyama:

        x_{j+1} = x_j a I q(x_j) + I asI  I_j,   I_j ~ N(0, I_n)

    onde q(x) A a direAAo de descida comum obtida pela projeAAo
    no simplex do espaAo de gradientes.

    ParAmetros
    ----------
    n_points : int
        NAomero de trajetA3rias paralelas (tamanho da populaAAo).
    step_size : float
        Tamanho de passo I do esquema de EuleraMaruyama.
    epsilon : float
        Intensidade de ruAdo I da difusAo.
    jac : callable ou None
        Jacobiano analAtico J(x) a a^{mAn}. Se ``None``, usa
        diferenAas finitas centradas via SciPy.
    """
    ALGO_FLAGS = {"multi", "many"}
    OBJECTIVE_SCOPE = "many"

    def __init__(self,
                 n_points: int = 100,
                 step_size: float = 0.01,
                 epsilon: float = 0.15,
                 jac=None,
                 output=MultiObjectiveOutput(),
                 array_backend: str = "auto",
                 gpu_dtype: str = "float32",
                 use_gpu: bool = False,
                 **kwargs):
        super().__init__(
            output=output,
            use_gpu=use_gpu,
            array_backend=array_backend,
            gpu_dtype=gpu_dtype,
            **kwargs,
        )
        _force_cpu_backend(
            self,
            "SSW finite-difference/Jacobian path is currently CPU-only; GPU backend is pending migration.",
        )
        self.n_points = n_points
        self.step_size = step_size
        self.epsilon = epsilon
        self.jac = jac

        # VariAvel interna para a funAAo objetivo vetorial
        self._func = None

    # ----- Setup -----------------------------------------------------------

    # ----- Setup -----------------------------------------------------------

    def _setup(self, problem, **kwargs):
        """Armazena metadados do problema."""
        # Nenhuma closure necessAria para versAo vetorizada
        pass

    # ----- InicializaAAo ---------------------------------------------------

    def _initialize_infill(self):
        """Gera pontos iniciais aleatA3rios no espaAo de decisAo."""
        xl = self.problem.xl
        xu = self.problem.xu
        n = self.problem.n_var

        X = np.empty((self.n_points, n))
        for i in range(self.n_points):
            X[i] = xl + self.random_state.random(n) * (xu - xl)

        pop = Population.new("X", X)
        return pop

    def _initialize_advance(self, infills=None, **kwargs):
        """Armazena a populaAAo avaliada apA3s a inicializaAAo."""
        self.pop = infills

    # ----- IteraAAo (EuleraMaruyama) --------------------------------------

    def _infill(self):
        """Aplica um passo de EuleraMaruyama em cada trajetA3ria."""
        X = self.pop.get("X")                           # (n_points, n_var)
        N_pop, n = X.shape
        m = self.problem.n_obj
        sigma = self.step_size
        eps = self.epsilon
        h = 1e-7  # Passo de diferenciaAAo finita

        # --- CAlculo dos Jacobianos ---
        if self.jac is not None:
             # Jacobiano analAtico (assumimos vetorizado ou loop python)
             try:
                 J_batch = self.jac(X) # Tenta chamada vetorizada (N, m, n)
             except TypeError:
                 J_batch = np.array([self.jac(x) for x in X])
        else:
             # DiferenAas finitas vetorizadas
             # Queremos J_i = [af_k / ax_j] de tamanho (m, n)
             # Expandimos X para adicionar perturbaAAes em cada dimensAo j
             # X_base: (N, n, n) -> repetimos X em dim 1
             X_rep = np.tile(X[:, np.newaxis, :], (1, n, 1))

             # Matriz identidade escalada: (n, n)
             Eye = np.eye(n) * h

             # X_plus e X_minus: (N, n, n)
             # Para cada indivAduo i, e cada dimensAo j (eixo 1), somamos h*e_j
             X_plus = X_rep + Eye[np.newaxis, :, :]
             X_minus = X_rep - Eye[np.newaxis, :, :]

             # Achatar para avaliaAAo em lote: (N*n, n)
             X_plus_flat = X_plus.reshape(-1, n)
             X_minus_flat = X_minus.reshape(-1, n)

             # Avaliar f(X + h) e f(X - h)
             # Nota: pymoo espera retorno em dicionArio
             out_plus, out_minus = {}, {}
             self.problem._evaluate(X_plus_flat, out_plus)
             self.problem._evaluate(X_minus_flat, out_minus)

             # Reshape F: (N, n, m) -> F_plus[i, j, :] = f(x_i + h*e_j)
             F_plus = out_plus["F"].reshape(N_pop, n, m)
             F_minus = out_minus["F"].reshape(N_pop, n, m)

             # DiferenAa centrada
             # J_{k, j} = (f_k^+ - f_k^-) / 2h
             # Atualmente F_diff[i, j, k] -> derivada de f_k referente a x_j
             J_raw = (F_plus - F_minus) / (2 * h)

             # Transpor para (N, m, n): J[i, k, j]
             J_batch = J_raw.transpose(0, 2, 1)

        # --- AtualizaAAo (Euler-Maruyama) ---
        X_new = np.empty_like(X)
        
        # O QP A resolvido individualmente (SLSQP A sequencial)
        # Mas A rApido pois m A pequeno.
        for i in range(N_pop):
            q = _compute_q(J_batch[i])
            eta = self.random_state.standard_normal(n)
            X_new[i] = X[i] - sigma * q + eps * np.sqrt(sigma) * eta

        # ProjeAAo nos limites (clipping)
        if self.problem.has_bounds():
            X_new = np.clip(X_new, self.problem.xl, self.problem.xu)

        pop = Population.new("X", X_new)
        return pop

    def _advance(self, infills=None, **kwargs):
        """Atualiza a populaAAo com as soluAAes avaliadas.

        Mescla a populaAAo corrente com os novos pontos e retAm
        apenas os nAo-dominados, limitando o tamanho a n_points
        para evitar crescimento descontrolado da memA3ria.
        """
        merged = Population.merge(self.pop, infills)

        # Filtra o conjunto nAo-dominado
        opt = filter_optimum(merged, least_infeasible=True)

        # Se o conjunto nAo-dominado exceder n_points, trunca
        if len(opt) > self.n_points:
            # MantAm os n_points com menor crowding (aleatA3rio aqui)
            idx = self.random_state.choice(
                len(opt), size=self.n_points, replace=False)
            self.pop = opt[idx]
        else:
            self.pop = opt

    def _set_optimum(self):
        """Identifica o conjunto nAo-dominado."""
        self.opt = filter_optimum(self.pop, least_infeasible=True)
