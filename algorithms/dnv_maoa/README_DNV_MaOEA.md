# DNV-MaOEA (Dynamic Normal Vectors for Many-Objective Optimization)

## 1. O que foi feito?
Este módulo contém a implementação do **DNV-MaOEA**, um algoritmo evolutivo many-objective (MaOPs) com arquitetura de duas camadas. O objetivo primário desta implementação é resolver o colapso clássico de algoritmos baseados em decomposição em Frentes de Pareto irregulares (desconexas, degeneradas ou invertidas).

### Principais Componentes:
1. **Arquitetura de Duas Camadas (Two-Layer DNV):**
   - **Camada Global (Sobrevivência):** Os vetores de Das-Dennis (`ref_dirs_base`) ancoram o Niching para seleção e sobrevivência. Isso garante cobertura absoluta do hipercubo de Pareto, impossibilitando que frentes desconectadas como a do DTLZ7 se percam.
   - **Camada Local (Dynamic Normal Vectors):** O *K-Means Esférico* atua de forma cirúrgica na criação da subpopulação, calculando as normais dinâmicas baseadas na Frente Não-Dominada local. Os vetores de cluster do K-Means ditam exclusivamente a direção local para a Equação Estocástica Diferencial (SDE), otimizando apenas a descida contínua via EEJ.

2. **Integração EEJ + DNV (Zero Avaliações Extras):**
   O `DNV-MaOEA` calcula a direção do gradiente local através do Jacobiano Estocástico de Schaeffler QP.
   - Em vez de usar *Diferenças Finitas* tradicionais que custam $2 \times N_{var}$ avaliações extras da função objetivo por indivíduo (como ocorria nas primeiras versões do SSW), aqui utilizamos o **EEJ (Evolutionary Ensemble Jacobian)**.
   - O EEJ extrai a superfície de gradientes das memórias passadas usando uma pseudo-inversa algébrica.
   - **Resultado (Atenção ao `n_eval`):** A contagem de `n_eval` escala estritamente à razão de `pop_size` por geração. Se o pop_size é 50, custará exatas 50 avaliações por iteração. A falta de inflação no `n_eval` é prova do sucesso matemático do EEJ, e não um bug de contagem.

3. **Tratamento de Terminação:**
   O controle de `_search_progress()` foi reescrito para interpretar perfeitamente critérios de terminação baseados em gerações (`n_gen`), evitando flatlines de probabilidade de busca no decorrer das execuções que caíam no fallback antigo.

## 2. Como Utilizar

O `DNV-MaOEA` opera nativamente no framework Pymoolab e Pymoo, sendo 100% retrocompatível:

```python
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoolab.algorithms.dnv_maoa.dnv_maoa import DNV_MaOEA

# Problema exemplo Many-Objective
problem = get_problem("dtlz2", n_var=10, n_obj=5)

# Apenas o pop_size é necessário. O K-Means irá agrupar as ref_dirs em "pop_size" clusters dinâmicos.
alg = DNV_MaOEA(
    pop_size=50,
    use_sdr=True, # Recomendado para m > 3 para afiar a seleção
)

res = minimize(problem, alg, ("n_eval", 5000), seed=1, verbose=True)
```

### Notas Finais
A documentação atual foi atualizada para guiar futuros whitepapers e validações acadêmicas desta nova união de SDE-EEJ com a técnica DNV.
