# Proposta de Artigo: SSW-DNV

**Título Escolhido:** A Two-Layer Diversity Architecture using Dynamic Normal Vectors for Many-Objective Optimization on Irregular Pareto Fronts

## 1. Contextualização e Motivação
Algoritmos Evolucionários de Muitos Objetivos (MaOEAs) baseados em decomposição (como MOEA/D e NSGA-III) dependem de vetores de referência predefinidos para manter a diversidade. Contudo, em problemas do mundo real, a Frente de Pareto verdadeira frequentemente possui geometrias complexas: irregulares, desconexas, degeneradas ou invertidas (como na família de problemas DTLZ, especialmente DTLZ7).
Vetores estáticos (como Das-Dennis) falham nessas topologias pois alocam poder computacional em regiões inviáveis do espaço de objetivos, resultando no desperdício de avaliações e no colapso de diversidade, onde as soluções convergem para sub-regiões restritas do *manifold*.

## 2. A Lacuna na Literatura (Literature Gap)
Pesquisas recentes (2020+) focaram fortemente na adaptação dinâmica de vetores de referência (Dynamic Reference Vectors). No entanto, abordagens que substituem **todos** os vetores estáticos por vetores dinâmicos frequentemente sofrem de perda de pressão global durante as gerações iniciais (efeito funil), perdendo nichos marginais antes que a fronteira se estabilize.
A lacuna atual é: *Como adaptar vetores dinamicamente para frentes irregulares sem sacrificar a robustez e a convergência global oferecida pelas âncoras estáticas?*

## 3. A Proposta (The Proposed Approach)
Propomos o **Stochastic Steepest Weights with Dynamic Normal Vectors (SSW-DNV)**, uma nova arquitetura híbrida de Duas Camadas (Two-Layer Architecture) que combina métodos estocásticos com adaptação topológica:

1. **Camada Global (Sobrevivência e Niching Estático):** Utiliza vetores Das-Dennis fixos para ancorar o processo de seleção (Niching e Sobrevivência). Isso blinda o algoritmo contra o colapso prematuro de diversidade e garante uma exploração robusta do hipercubo, evitando o aprisionamento em frentes parciais.
2. **Camada Local (Busca Local Guiada por Topologia):** Introduz o conceito de *Dynamic Normal Vectors* através de um algoritmo de agrupamento dinâmico (**Spherical K-Means**) aplicado sobre o *Non-Dominated Archive*. Esses vetores dinâmicos são injetados **exclusivamente** na geração de descendentes via Equações Diferenciais Estocásticas (SDE / Método Schaeffler).
3. **Sinergia:** O EEJ (Evolutionary Ensemble Jacobian) calcula as normais perfeitas do *manifold* local, permitindo que os operadores de mutação "deslizem" continuamente pelas bordas irregulares sem gastar avaliações extras da função objetivo.

## 4. Estrutura do Artigo (Outline)

- **Abstract:** Resumo do colapso em frentes irregulares e a introdução da arquitetura Two-Layer.
- **1. Introduction:** Evolução dos MaOEAs, o problema dos vetores estáticos, as tentativas recentes de vetores dinâmicos e suas falhas no início da busca, e a proposta do SSW-DNV.
- **2. Related Work:**
    - MaOEAs baseados em vetores dinâmicos (2020-2024).
    - Métodos de Gradiente em algoritmos evolucionários e Jacobian-based descents.
- **3. Proposed Method (SSW-DNV):**
    - 3.1. Visão Geral da Arquitetura Híbrida.
    - 3.2. A Camada Global: Seleção guiada por âncoras estáticas.
    - 3.3. A Camada Local: Identificação Topológica via Spherical K-Means.
    - 3.4. SDE-DNV: Integração dos Vetores Dinâmicos com Equações Diferenciais Estocásticas.
- **4. Experimental Setup:**
    - Benchmarks: Suíte DTLZ (foco no DTLZ7) e WFG.
    - Baselines: NSGA-III, MOEA/D-AWA, A-NSGA-III, e o precursor SSW-RDPA.
    - Métricas: Hypervolume (HV) e Delta-p.
- **5. Results and Discussion:**
    - Validação da não-regressão em frentes regulares (DTLZ2).
    - Superioridade estatística e estabilidade topológica em frentes desconexas (DTLZ7).
    - Análise de tempo computacional (EEJ vs Avaliações Clássicas).
- **6. Conclusion:** Reflexões finais e futuras expansões para restrições dinâmicas.

## 5. Artigos Base e Referências-Chave (Para Buscar/Citar)
Para a escrita do *Related Work*, devemos buscar (via OpenAlex/Scopus) artigos a partir de 2020 que abordem:
1. *Dynamic Reference Vector Adaptation in MaOEAs.*
2. *Handling Irregular/Disconnected Pareto Fronts in Decomposition-based Evolutionary Algorithms.*
3. *Clustering Methods (K-Means/Hierarchical) for Reference Vector Generation.*
4. *Continuous Local Search in Evolutionary Multi-objective Optimization.*
