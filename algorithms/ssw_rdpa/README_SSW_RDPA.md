# SSW-RDPA
**Schaeffler-SDE-Weighted · Reference Directions · PBI · Adaptive**

Arquivo:

- `guiPymoo/algorithms/ssw_rdpa/ssw_rdpa.py`

Classe:

- `SSW_RDPA`

Objetivo:

- Algoritmo evolucionário com foco many-objective (`m > 3`) usando Schaeffler + SDE, decomposição por vetores de referência, niching por PBI-score com θ adaptativo, SDR para pressão de seleção elevada e two-archive (diversidade + convergência), com regime estabilizado para low-objective (`m <= 3`).

---

## Acrônimo

| Componente | Significado |
|---|---|
| **S** | **S**chaeffler QP — direção de descida local via solução do QP simplex |
| **S** | **S**DE — Stochastic Differential Evolution (busca local estocástica) |
| **W** | **W**eighted — ponderação adaptativa SBX / RVX / SDE por estado de busca |
| **R** | **R**eference **D**irections — decomposição estrutural por vetores Das-Dennis (base NSGA-III) |
| **D** | (parte de **RD**) |
| **P** | **P**BI-score — Penalty-Based Boundary Intersection para niching por vetor de referência |
| **A** | **A**daptive — θ adaptativo, ε dinâmico, Q-learning para SBX η, CMA para path/cov |

---

## Compatibilidade com pymoo

O algoritmo é **100% compatível com pymoo** e **não depende de nenhuma classe interna do NSGA-III**.

| Componente pymoo | Módulo |
|---|---|
| `Algorithm` (base) | `pymoo.core.algorithm` |
| `Population` | `pymoo.core.population` |
| `DefaultDuplicateElimination` | `pymoo.core.duplicate` |
| `cv_and_dom_tournament` | `pymoo.algorithms.moo.sms` |
| `Mating`, `SBX`, `PolynomialMutation`, `TournamentSelection` | `pymoo.core.mating` / `pymoo.operators.*` |
| `NonDominatedSorting`, `Dominator` | `pymoo.util.*` |

As funções `HyperplaneNormalization`, `associate_to_niches` e `calc_niche_count` do `pymoo.algorithms.moo.nsga3` foram **completamente removidas** e substituídas por implementações próprias (ver mecanismo 12 abaixo). Todos os novos métodos usam **apenas numpy** — zero dependências além do core do pymoo.

---

## Mecanismos implementados

### 1. Epsilon dinâmico com caminho cumulativo (CSA)

`_adapt_epsilon` — controla `epsilon_state` por cobertura de nichos, taxa de sucesso SDE, estagnação e gap SBX vs SDE. O caminho cumulativo `eps_path` (estilo CSA do CMA-ES) adapta a amplitude estocástica. Limites dinâmicos `eps_min_dyn`/`eps_max_dyn` escalam com progresso e escassez de cobertura.

### 2. Auto-calibração CMA de hiperparâmetros

`_configure_cma_hyperparams` — recalcula `c_path`, `c_cov`, `c_sigma`, `sigma_damp` a partir de `n_var` e `pop_size` usando as fórmulas canônicas do CMA-ES (Hansen 2016). Ativado por `use_cma_auto=True`.

### 3. Ramo global RVX — DE current-to-best/1

`_create_rvx_offspring` — mutação `x_i + F*(x_best − x_i) + F*(x_r1 − x_r2)` com crossover binomial. `F` e `CR` adaptados por histórico de sucesso.

**Novidade v2.1:** Implementa **Neighborhood Mating Restriction** (restrição de vizinhança) para reduzir híbridos degradados em alta dimensão.

**Atualização v2.2:** A restrição de vizinhança foi parametrizada por `prob_neighbor_mating` (default `0.70`) e passa por auto-calibração por regime (`m<=3` reduz acoplamento local para priorizar exploração global).

### 4. Seleção com foco em região esparsa

`_parent_weights` — combina rank de Pareto, raridade de nicho, convergência normalizada e ângulo com o vetor de referência. Boost extra para pais em nichos esparsos (quantil `sparse_quantile`). `_elite_refinement` — preenche nichos vazios e retém elite por convergência.

### 5. State-switch por diversidade/convergência

`_infill` — a cada geração classifica o estado em `diversity`, `balanced` ou `convergence` por entropia de nichos, cobertura e CV de ocupação. Cada modo ajusta pesos de niching, limites SDE e fração RVX.

**Novidade v2.1:** Métrica de convergência interna alterada de `sum(f)` para **distância Euclidiana média à origem** (`mean(norm(f))`). Isso remove o viés de seleção de "cantos" em frentes côncavas (ex: DTLZ2/4), garantindo pressão de convergência uniforme em qualquer geometria.

### 6. Q-learning para intensidade de cruzamento (SBX η)

`_update_q_policy` / `_apply_policy_action` — Q-learning discreto (3 estados × 3 ações) seleciona `SBX eta` ∈ {15, 20, 28} online. Recompensa baseada em ganho de convergência e diversidade entre gerações.

### 7. Inicialização por Latin Hypercube Sampling

`_initialize_infill` + `_latin_hypercube_sample` — quando `use_lhs=True`, amostra o espaço de decisão em `n_var` estratos uniformes com permutação independente por dimensão. Elimina clusters aleatórios e melhora cobertura inicial do arquivo não-dominado.

**Referência:** McKay, M.D., Beckman, R.J., Conover, W.J. (1979). A Comparison of Three Methods for Selecting Values of Input Variables in the Analysis of Output from a Computer Code. *Technometrics*, 21(2):239–245.

### 8. SDR — Strengthened Dominance Relation

`_sdr_fronts` + `_sdr_dominates` — quando `use_sdr=True` e `n_obj > 3`, o ranking em frontes usa SDR em vez de Pareto pura. `x` SDR-domina `y` se `x` melhora em ≥ 1 objetivo por margem > `sdr_sigma × span` sem piorar além de `sdr_sigma` em nenhum outro. Eleva a pressão de seleção para m ≥ 4 sem comprometer diversidade.

**Referências:**
- Tian, Y., Cheng, R., Zhang, X., Su, Y., Jin, Y. (2018). An Indicator-Based Multiobjective Evolutionary Algorithm With Reference Point Adaptation for Better Versatility. *IEEE TEVC*, 22(4):609–622. DOI: [10.1109/TEVC.2017.2749619](https://doi.org/10.1109/TEVC.2017.2749619).
- Wang, Q., Gu, Q., Zhou, Q., Xiong, N.N., Liu, D. (2025). Indicator selection and adaptive angle estimation in many-objective optimization. *Information Sciences*. DOI: [10.1016/j.ins.2024.121608](https://doi.org/10.1016/j.ins.2024.121608).

### 9. Niching por PBI-score com θ adaptativo (θ-dominância)

`_niching` + `_pbi_score` — o desempate no niching usa PBI(x, w) = d₁ + θ·d₂, onde d₁ é a componente paralela ao vetor de referência (convergência) e d₂ é a componente perpendicular (diversidade). Scores menores = melhor qualidade no subproblema do nicho.

**Novidade v2.1:** `_theta_current` agora escala automaticamente com a dimensionalidade. Base `5.0 + 1.2 × (m - 3)`, adaptando a penalidade angular para evitar deriva em espaços de muitos objetivos (`m > 3`). Com `theta_adapt=True`, esse valor decai linearmente para aumentar a tolerância no final da busca.

**Atualização v2.2:** O peso efetivo do PBI no score de niching foi suavizado para `m<=3`, reduzindo viés de convergência prematura e preservando cobertura em ZDT.

**Referências:**
- Yuan, Y., Xu, H., Wang, B., Yao, X. (2016). A New Dominance Relation-Based Evolutionary Algorithm for Many-Objective Optimization. *IEEE TEVC*, 20(1):16–37. DOI: [10.1109/TEVC.2015.2420112](https://doi.org/10.1109/TEVC.2015.2420112).
- Liang, P., Chen, Y., Sun, Y., Huang, Y., Li, W. (2024). An information entropy-driven evolutionary algorithm based on reinforcement learning for many-objective optimization. *Expert Systems with Applications*, 238(E):122164. DOI: [10.1016/j.eswa.2023.122164](https://doi.org/10.1016/j.eswa.2023.122164).

### 10. Two-archive: convergência + diversidade

`_build_conv_archive` + `_advance` — dois arquivos independentes:
- `nd_archive` — frente de Pareto (diversidade), comportamento original.
- `conv_archive` — `conv_archive_frac × pop_size` indivíduos com menor ASF de Tchebycheff por nicho (convergência por subproblema).
- injeção de ancoragem do `conv_archive` no pareamento POCEA com taxa `archive_injection_rate` (gated por modo/progresso para evitar sobreexploração em `m<=3`).

`self.opt` é a união não-dominada dos dois arquivos. Garante que cada vetor de referência tenha ao menos um representante bem convergido.

**Referências:**
- Wang, H., Jiao, L., Yao, X. (2015). Two_Arch2: An Improved Two-Archive Algorithm for Many-Objective Optimization. *IEEE TEVC*, 19(4):524–541. DOI: [10.1109/TEVC.2014.2350987](https://doi.org/10.1109/TEVC.2014.2350987).
- Wang, J., Zheng, Y., Zhang, Z., Peng, H., Wang, H. (2025). A novel multi-state reinforcement learning-based multi-objective evolutionary algorithm. *Information Sciences*, 688:121397. DOI: [10.1016/j.ins.2024.121397](https://doi.org/10.1016/j.ins.2024.121397).
- Zhang, W., Liu, J., Liu, Y., Liu, J., Tan, S. (2024). A many-objective evolutionary algorithm under diversity-first selection based framework. *Expert Systems with Applications*, 250:123949. DOI: [10.1016/j.eswa.2024.123949](https://doi.org/10.1016/j.eswa.2024.123949).

### 11. Blending Schaeffler + Tchebycheff na direção SDE

`_create_sde_offspring` — para `n_obj > 4`, a direção de descida `q` é:

```
q = (1 − α) × q_Schaeffler  +  α × q_Tchebycheff
α = clip(0.10 + 0.60 × progress, 0.10, 0.65)
```

A direção de Tchebycheff aponta para o objetivo com maior resíduo relativo ao vetor de referência do nicho — mais estável que o QP de Schaeffler quando o Jacobian é escasso.

**Referências:**
- Xing, L., Li, J., Cai, Z., Hou, F. (2023). A Two-State Dynamic Decomposition-Based Evolutionary Algorithm for Handling Many-Objective Optimization Problems. *Mathematics*, 11(3):493. DOI: [10.3390/math11030493](https://doi.org/10.3390/math11030493).
- Wang, Q., Gu, Q., Zhou, Q., Xiong, N.N., Liu, D. (2025). Indicator selection and adaptive angle estimation in many-objective optimization. *Information Sciences*. DOI: [10.1016/j.ins.2024.121608](https://doi.org/10.1016/j.ins.2024.121608).

---

### 12. Normalização e associação de nichos nativas (sem NSGA-III)

**Arquivos:** `_IdealNadirTracker`, `_associate_to_niches_angular`, `_calc_niche_count`

Substitui integralmente `HyperplaneNormalization`, `associate_to_niches` e `calc_niche_count` do `pymoo.algorithms.moo.nsga3`.

**`_IdealNadirTracker` — normalização adaptativa por EMA do nadir:**

O NSGA-III normaliza pelo hiperplano de intercepção dos pontos extremos, o que é instável para m > 5 (os pontos extremos são sensíveis a outliers e ausentes em alguns objetivos). O tracker próprio:
- **Ideal:** mínimo acumulado elemento-a-elemento sobre *todas* as soluções avaliadas — monótono não-crescente, nunca esquece um bom valor.
- **Nadir:** EMA (decaimento `nadir_ema=0.20`) do máximo por geração na frente não-dominada — estável, sem fit de hiperplano, robusto a outliers.

Resultado: normalização mais uniforme em todas as direções de referência, especialmente no interior do simplex onde o hiperplano do NSGA-III tende a distorcer distâncias.

**`_associate_to_niches_angular` — associação por menor ângulo:**

Em vez de distância perpendicular ao hiperplano (NSGA-III), usa o **ângulo entre o vetor objetivo normalizado e cada vetor de referência** — equivalente a maximizar o cosseno entre eles. Isso é:

```
niche(x) = argmax_j  cos(f_norm(x), w_j)
d_perp(x) = ||f_norm(x)|| × sin(angle)
```

Vantagens para diversidade em MaOP:
1. **Invariância de escala** — o ângulo não depende da magnitude do vetor objetivo, apenas de sua direção. Soluções no interior do simplex são associadas ao nicho geometricamente correto mesmo com normalização imperfeita.
2. **Sem dependência de pontos extremos** — o NSGA-III precisa de um ponto extremo por objetivo para calcular o hiperplano; em problemas com frentes degeneradas ou côncavas, isso falha. A associação angular não tem essa dependência.
3. **Consistência com RVEA e θ-DEA** — os algoritmos mais competitivos pós-2015 em MaOP usam associação angular (RVEA, Cheng, Jin, Olhofer e Sendhoff 2016; θ-DEA, Yuan, Xu, Wang e Yao 2016).

**Referências:**
- Cheng, R., Jin, Y., Olhofer, M., Sendhoff, B. (2016). A Reference Vector Guided Evolutionary Algorithm for Many-Objective Optimization. *IEEE TEVC*, 20(5):773–791. DOI: [10.1109/TEVC.2016.2519378](https://doi.org/10.1109/TEVC.2016.2519378). *(RVEA — associação angular)*
- Yuan, Y., Xu, H., Wang, B., Yao, X. (2016). A New Dominance Relation-Based Evolutionary Algorithm for Many-Objective Optimization. *IEEE TEVC*, 20(1):16–37. DOI: [10.1109/TEVC.2015.2420112](https://doi.org/10.1109/TEVC.2015.2420112). *(θ-DEA — associação angular + PBI)*

### 13. Regime low-objective estabilizado (`m <= 3`)

**Arquivos:** `_auto_calibrate_runtime_controls`, `_infill`, `_update_operator_credit`, `_create_sbx_offspring`, `_niching`

Principais ajustes aplicados:
- `low_obj_mode` habilita um ramo dedicado de calibração com `ratio_sde` menor e limites mais conservadores.
- `use_sdr=False` para `m<=3`; `use_two_archive` é reduzido (desligado em `m<=2`).
- redução automática de `prob_neighbor_mating` e `archive_injection_rate` em low-objective.
- amortecimento explícito de SDE quando `success_sde` fica abaixo de SBX.
- faixa de `theta_pbi` reduzida para 2D/3D e niching menos agressivo em PBI.

Efeito esperado: reduzir colapso por exploração local excessiva em frentes de baixa dimensionalidade, mantendo robustez no regime many-objective.

---

## Parâmetros

### Principais

| Parâmetro | Default | Descrição |
|---|---|---|
| `ref_dirs` | — | Vetores de referência (Das-Dennis ou equivalente) |
| `pop_size` | `len(ref_dirs)` | Tamanho da população |
| `epsilon` | `0.10` | Amplitude base da perturbação estocástica SDE |
| `ratio_sde` | `0.14` | Fração inicial da população para offspring SDE |
| `use_cma_auto` | `True` | Auto-calibração CMA de hiperparâmetros |
| `rvx_share_base` | `0.40` | Fração base do ramo global RVX |

### Novos parâmetros (v2.2)

| Parâmetro | Default | Faixa | Descrição |
|---|---|---|---|
| `use_lhs` | `True` | bool | LHS na inicialização |
| `use_sdr` | `True` | bool | SDR em vez de Pareto puro para m > 3 |
| `sdr_sigma` | `0.02` | `[0, 0.10]` | Tolerância SDR como fração do span normalizado |
| `theta_pbi` | `5.0` | `[0.5, 20]` | θ inicial do PBI-score no niching |
| `theta_adapt` | `True` | bool | Decrescer θ de `theta_pbi` → 1.5 com o progresso |
| `use_two_archive` | `True` | bool | Arquivo de convergência explícito separado |
| `conv_archive_frac` | `0.20` | `[0.05, 0.40]` | Fração de `pop_size` no arquivo de convergência |
| `prob_neighbor_mating` | `0.70` | `[0.0, 1.0]` | Probabilidade base de mating por vizinhança (auto-ajustada por regime) |
| `archive_injection_rate` | `0.12` | `[0.0, 0.50]` | Taxa base de injeção de âncoras do `conv_archive` no pareamento |

### Parâmetros avançados de controle

| Parâmetro | Default | Descrição |
|---|---|---|
| `sparse_quantile` | `0.30` | Quantil para detectar nichos esparsos |
| `elite_keep_frac` | `0.12` | Fração de elite retida em modo convergência |
| `angle_adapt_gain` | `0.30` | Peso do ângulo candidato-referência no niching |
| `qlearn_alpha` | `0.22` | Taxa de aprendizado Q-learning |
| `qlearn_gamma` | `0.90` | Fator de desconto Q-learning |
| `qlearn_eps` | `0.08` | Epsilon-greedy de exploração Q-learning |
| `rvx_mu_f` | `0.55` | Média inicial de F para RVX |
| `rvx_mu_cr` | `0.85` | Média inicial de CR para RVX |

---

## Telemetria

`alg.get_telemetry()` retorna lista de dicts por geração:

| Campo | Descrição |
|---|---|
| `gen`, `n_eval`, `progress` | Geração, avaliações e progresso normalizado [0,1] |
| `coverage` | Fração de vetores de referência ocupados |
| `ratio_sde` | Fração corrente de offspring SDE |
| `epsilon_state` | Amplitude ε corrente |
| `success_sde`, `success_sbx` | Taxas de sucesso por operador |
| `archive_improved` | Se o arquivo melhorou nesta geração |
| `stagnation_streak` | Gerações consecutivas sem melhoria |
| `search_mode` | `diversity` / `balanced` / `convergence` |
| `entropy`, `niche_cv` | Métricas de distribuição de nichos |
| `sigma_mean`, `sde_step_norm` | Estatísticas do modelo local |
| `c_path`, `c_cov`, `c_sigma` | Hiperparâmetros CMA correntes |
| `sbx_eta`, `q_state` | Estado Q-learning e η do SBX |
| `rvx_mu_f`, `rvx_mu_cr` | Parâmetros adaptativos do RVX |
| `theta_pbi_current` | θ corrente do PBI (adaptativo) |
| `conv_archive_size` | Tamanho do arquivo de convergência |
| `low_obj_mode` | Flag de regime low-objective (`m<=3`) |
| `prob_neighbor_mating` | Probabilidade corrente de mating por vizinhança |
| `archive_injection_rate` | Taxa corrente de injeção de `conv_archive` no mating |

---

## Uso rápido

```python
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from guiPymoo.algorithms.ssw_rdpa.ssw_rdpa import SSW_RDPA

ref_dirs = get_reference_directions("das-dennis", 5, n_partitions=5)

alg = SSW_RDPA(
    ref_dirs=ref_dirs,
    pop_size=len(ref_dirs),
    epsilon=0.10,
    ratio_sde=0.14,
    use_cma_auto=True,
    rvx_share_base=0.40,
    rvx_mu_f=0.55,
    rvx_mu_cr=0.85,
    sparse_quantile=0.30,
    elite_keep_frac=0.12,
    angle_adapt_gain=0.30,
    qlearn_alpha=0.22,
    qlearn_gamma=0.90,
    qlearn_eps=0.08,
    use_lhs=True,
    use_sdr=True,
    sdr_sigma=0.02,
    theta_pbi=5.0,
    theta_adapt=True,
    use_two_archive=True,
    conv_archive_frac=0.20,
    prob_neighbor_mating=0.70,
    archive_injection_rate=0.12,
    seed=1,
)

res = minimize(problem, alg, ("n_eval", 30000), seed=1, verbose=False)
telemetry = alg.get_telemetry()
```

---

## Critério de sucesso vs NSGA-III

1. **Qualidade:** média de `Delta_p` (ou IGD+) do `SSW_RDPA` menor que `NSGA-III` no agregado. Vantagem relativa mínima recomendada: `>= 1.0%`.
2. **Consistência:** vitória em pelo menos `60%` das funções testadas no mesmo valor de `m`. Repetir para `m=5` e `m=10`.
3. **Significância:** teste de Wilcoxon pareado com `p < 0.05` nas funções em que houver ganho.
4. **Custo:** reportar tempo médio por run. Se `SSW_RDPA` for mais lento, explicitar o trade-off qualidade vs tempo.

---

## Em caso de vitória, documentar

**Regra:** registrar neste README no mesmo dia do experimento. Não considerar vitória isolada por função — registrar o agregado e o recorte por função.

```
- Data:
- Script:
- Pasta de resultados:
- Configuração: runs | n_evals | m | pop_size | backend | workers
- Resultado agregado:
    Delta_p NSGA-III: | Delta_p SSW_RDPA: | ganho (%):
    funções vencidas: | perdidas: | empates:
- Significância Wilcoxon: vitórias p<0.05 | derrotas p<0.05
- Tempo: NSGA-III (s) | SSW_RDPA (s) | razão
- Observação técnica: quais mecanismos foram responsáveis pelo ganho.
```

---

## Log de benchmark

### SSW-RDPA postpatch-final (2026-02-19) — ZDT (`m=2`)

- **Script:** `guiPymoo/tests/zdt_ssw_rdpa_vs_nsga3_benchmark.py`
- **Pasta:** `guiPymoo/artigo/results/zdt_ssw_rdpa_vs_nsga3_postpatch_final/`
- **Config:** `runs=5`, `n_evals=25000`, `pop_size=100`, `backend=process`, `workers=8`
- **Seeds:** `seed_plan.csv` com `seed_nsga3` e `seed_ssw_rdpa` aleatórios por run (SystemRandom), independentes e diferentes entre algoritmos.
- **Resultado por função:**

| Função | Vencedor | Margem do vencedor sobre o perdedor |
|---|---|---|
| `zdt1` | `NSGA-III` | `67.39%` |
| `zdt2` | `NSGA-III` | `73.03%` |
| `zdt3` | `SSW-RDPA` | `49.11%` |
| `zdt4` | `NSGA-III` | `87.20%` |
| `zdt6` | `SSW-RDPA` | `85.00%` |

- **Resumo:** `SSW-RDPA` venceu `2/5` funções (com ganhos fortes em `zdt3` e `zdt6`), confirmando competitividade parcial em low-objective.

### SSW-RDPA postpatch-final (2026-02-19) — DTLZ (`m=5`)

- **Script:** `guiPymoo/tests/dtlz_ssw_rdpa_vs_nsga3_benchmark.py`
- **Pasta:** `guiPymoo/artigo/results/dtlz_ssw_rdpa_vs_nsga3_postpatch_final/`
- **Config:** `runs=5`, `n_evals=25000`, `pop_size=50` (default do script), `backend=process`, `workers=8`
- **Seeds:** `seed_plan.csv` com seeds aleatórios por run; avaliação pareada usa a mesma seed para os dois algoritmos em cada run.
- **Resultado por função:**

| Função | Vencedor | Margem do vencedor sobre o perdedor |
|---|---|---|
| `dtlz1` | `NSGA-III` | `81.13%` |
| `dtlz2` | `NSGA-III` | `90.84%` |
| `dtlz3` | `SSW-RDPA` | `9.59%` |
| `dtlz4` | `NSGA-III` | `91.02%` |
| `dtlz5` | `SSW-RDPA` | `70.81%` |
| `dtlz6` | `SSW-RDPA` | `69.74%` |
| `dtlz7` | `SSW-RDPA` | `40.20%` |

- **Resumo:** `SSW-RDPA` venceu `4/7` funções. Média simples de `Delta_p` por função: `NSGA-III=1.5295`, `SSW-RDPA=1.3504` (vantagem agregada `11.71%` para `SSW-RDPA`).

### SSW_RD_N tuned2 (2026-02-19) — linha de base histórica

- **Pasta:** `guiPymoo/artigo/results/dtlz_ssw_vs_nsga3_m5_r3_e25k_pop100_tuned2/`
- **Config:** `m=5`, `runs=3`, `n_evals=25000`, `pop_size=100`
- **Resultado:** 4 vitórias vs 3 NSGA-III | Δ_p: NSGA-III=1.7215, SSW=0.9823 | margem **42.94%**
- **Custo:** 4.19× mais lento que NSGA-III

### SSW_RD_N autocal-v2 (2026-02-19) — linha de base histórica

- **Pasta:** `guiPymoo/artigo/results/dtlz_ssw_vs_nsga3_m5_r3_e25k_pop100_autocal_v2/`
- **Config:** `m=5`, `runs=3`, `n_evals=25000`, `pop_size=100`
- **Resultado:** 4 vitórias vs 3 NSGA-III | Δ_p: NSGA-III=2.0686, SSW=0.9106 | margem **55.98%**
- **Custo:** 7.63× mais lento que NSGA-III

> **Próximo benchmark sugerido:** repetir DTLZ em `m=10` com a mesma metodologia e registrar no mesmo formato. Nota de desempenho: `_sdr_fronts` é O(n²·m) — para `pop_size > 200`, avaliar `use_sdr=False` ou `sdr_sigma=0.0` (degrada graciosamente ao NDS padrão do pymoo).

---

## Referências

- Wang, J., Zheng, Y., Zhang, Z., Peng, H., Wang, H. (2025). A novel multi-state reinforcement learning-based multi-objective evolutionary algorithm (MRL-MOEA). *Information Sciences*, 688:121397. DOI: [10.1016/j.ins.2024.121397](https://doi.org/10.1016/j.ins.2024.121397).
- Liang, P., Chen, Y., Sun, Y., Huang, Y., Li, W. (2024). An information entropy-driven evolutionary algorithm based on reinforcement learning for many-objective optimization (RL-RVEA). *Expert Systems with Applications*, 238(E):122164. DOI: [10.1016/j.eswa.2023.122164](https://doi.org/10.1016/j.eswa.2023.122164).
- Wang, Q., Gu, Q., Zhou, Q., Xiong, N.N., Liu, D. (2025). A many-objective evolutionary algorithm based on indicator selection and adaptive angle estimation (MaOEA-ISAE). *Information Sciences*. DOI: [10.1016/j.ins.2024.121608](https://doi.org/10.1016/j.ins.2024.121608).
- Xing, L., Li, J., Cai, Z., Hou, F. (2023). A Two-State Dynamic Decomposition-Based Evolutionary Algorithm for Handling Many-Objective Optimization Problems. *Mathematics*, 11(3):493. DOI: [10.3390/math11030493](https://doi.org/10.3390/math11030493).
- Zhang, W., Liu, J., Liu, Y., Liu, J., Tan, S. (2024). A many-objective evolutionary algorithm under diversity-first selection based framework. *Expert Systems with Applications*, 250:123949. DOI: [10.1016/j.eswa.2024.123949](https://doi.org/10.1016/j.eswa.2024.123949).
- Yuan, Y., Xu, H., Wang, B., Yao, X. (2016). A New Dominance Relation-Based Evolutionary Algorithm for Many-Objective Optimization. *IEEE TEVC*, 20(1):16–37. DOI: [10.1109/TEVC.2015.2420112](https://doi.org/10.1109/TEVC.2015.2420112).
- Wang, H., Jiao, L., Yao, X. (2015). Two_Arch2: An Improved Two-Archive Algorithm for Many-Objective Optimization. *IEEE TEVC*, 19(4):524–541. DOI: [10.1109/TEVC.2014.2350987](https://doi.org/10.1109/TEVC.2014.2350987).
- Tian, Y., Cheng, R., Zhang, X., Cheng, F., Jin, Y. (2018). An Indicator-Based Multiobjective Evolutionary Algorithm With Reference Point Adaptation for Better Versatility. *IEEE TEVC*, 22(4):609–622. DOI: [10.1109/TEVC.2017.2749619](https://doi.org/10.1109/TEVC.2017.2749619).
- Cheng, R., Jin, Y., Olhofer, M., Sendhoff, B. (2016). A Reference Vector Guided Evolutionary Algorithm for Many-Objective Optimization (RVEA). *IEEE TEVC*, 20(5):773–791. DOI: [10.1109/TEVC.2016.2519378](https://doi.org/10.1109/TEVC.2016.2519378).
- McKay, M.D., Beckman, R.J., Conover, W.J. (1979). A Comparison of Three Methods for Selecting Values of Input Variables in the Analysis of Output from a Computer Code. *Technometrics*, 21(2):239–245.
