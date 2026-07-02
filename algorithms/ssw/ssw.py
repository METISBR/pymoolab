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
                 delta: float = 0.1,
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
        self.delta = delta
        self.jac = jac

        self.sigma = np.full(n_points, step_size, dtype=float)

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

    def _compute_jacobian(self, X):
        """Calcula o Jacobiano para um batch de pontos X (N, n).

        As avaliações de diferenças finitas passam pelo evaluator e são
        contabilizadas em ``evaluator.n_eval``, entrando no orçamento
        total de avaliações do algoritmo.
        """
        # Strict abort para previnir overshoot de avaliações
        if hasattr(self, "termination") and hasattr(self.termination, "n_max_evals"):
            if self.evaluator.n_eval >= self.termination.n_max_evals:
                return np.zeros((X.shape[0], self.problem.n_obj, X.shape[1]))

        N_pop, n = X.shape
        m = self.problem.n_obj
        h = 1e-7

        if self.jac is not None:
             try:
                 J_batch = self.jac(X)
             except TypeError:
                 J_batch = np.array([self.jac(x) for x in X])
             return J_batch

        X_rep = np.tile(X[:, np.newaxis, :], (1, n, 1))
        Eye = np.eye(n) * h
        X_plus = X_rep + Eye[np.newaxis, :, :]
        X_minus = X_rep - Eye[np.newaxis, :, :]

        X_plus_flat = X_plus.reshape(-1, n)
        X_minus_flat = X_minus.reshape(-1, n)

        pop_plus = Population.new("X", X_plus_flat)
        pop_minus = Population.new("X", X_minus_flat)

        self.evaluator.eval(self.problem, pop_plus)
        self.evaluator.eval(self.problem, pop_minus)

        F_plus = pop_plus.get("F").reshape(N_pop, n, m)
        F_minus = pop_minus.get("F").reshape(N_pop, n, m)

        J_raw = (F_plus - F_minus) / (2 * h)
        J_batch = J_raw.transpose(0, 2, 1)
        return J_batch

    def _infill(self):
        """Aplica um passo de Euler–Maruyama com controle de erro (stochbas).

        O loop de halving é limitado a ``max_halvings=3`` para conter o
        custo de Jacobianos adicionais por diferenças finitas.
        """
        X = self.pop.get("X")                           # (n_points, n_var)
        N_pop, n = X.shape
        eps = self.epsilon

        mask_done = np.zeros(N_pop, dtype=bool)
        X_new = np.empty_like(X)

        # q(x) inicial
        J_batch = self._compute_jacobian(X)
        Q_x = np.empty((N_pop, n))
        for i in range(N_pop):
            Q_x[i] = _compute_q(J_batch[i])

        halvings = np.zeros(N_pop, dtype=int)
        max_halvings = 3

        while not np.all(mask_done):
            # Strict abort para evitar excesso de avaliações nas subdivisões de passo
            if hasattr(self, "termination") and hasattr(self.termination, "n_max_evals"):
                if self.evaluator.n_eval >= self.termination.n_max_evals:
                    X_new[~mask_done] = X[~mask_done]
                    break

            idx = np.where(~mask_done)[0]
            n_act = len(idx)

            X_act = X[idx]
            Q_act = Q_x[idx]
            sigma_act = self.sigma[idx, np.newaxis]
            sqsig_act = np.sqrt(sigma_act / 2.0)

            n1 = self.random_state.standard_normal((n_act, n))
            n2 = self.random_state.standard_normal((n_act, n))
            n3 = n1 + n2

            # Passo 1 (Inteiro) e Meio Passo (1)
            x1_act = X_act - sigma_act * Q_act - eps * n3 * sqsig_act
            xi_act = X_act - 0.5 * sigma_act * Q_act - eps * n1 * sqsig_act

            # Jacobiano e q(x) no meio passo
            J_xi = self._compute_jacobian(xi_act)
            Q_xi = np.empty_like(xi_act)
            for i in range(n_act):
                Q_xi[i] = _compute_q(J_xi[i])

            # Meio Passo (2)
            x2_act = xi_act - 0.5 * sigma_act * Q_xi - eps * n2 * sqsig_act

            # Critério de aceite
            diff = x1_act - x2_act
            dist = np.sqrt(np.sum(diff**2, axis=1))
            accept = dist < self.delta

            # Atualiza trajetórias aceitas
            acc_idx = idx[accept]
            if len(acc_idx) > 0:
                X_new[acc_idx] = x2_act[accept]
                mask_done[acc_idx] = True

            # Reduz sigma para trajetórias rejeitadas
            rej_idx = idx[~accept]
            if len(rej_idx) > 0:
                self.sigma[rej_idx] /= 2.0
                halvings[rej_idx] += 1

                # Desiste após max_halvings para evitar loop infinito
                stuck = rej_idx[halvings[rej_idx] >= max_halvings]
                if len(stuck) > 0:
                    stuck_mask = halvings[rej_idx] >= max_halvings
                    X_new[stuck] = x2_act[~accept][stuck_mask]
                    mask_done[stuck] = True

        # Projeção nos limites (clipping)
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
