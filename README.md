# 04 — Experimentos com LLMs

> **Avaliação de IA** · Status: **Planejado · standby**  
> Parte do portfólio [Beatriz Helena — Engenharia de IA](https://beatrizorlandi.github.io/projetos/laboratorio-llms.html)

Design experimental para comparar prompts e modelos por qualidade, latência, consumo e análise de erros.

**Stack principal:** `Python` · `databricks-sdk` · `Pydantic / jsonschema` · `scikit-learn` · `MLflow Tracking / Tracing`

---

## O Problema

Uma resposta convincente isolada não mostra se uma mudança de prompt melhora uma tarefa de maneira consistente.

### Escopo Entregue

Planejamento de requisitos, dataset, rubricas, contratos e experimentos. Nenhum comparativo de geração LLM foi executado nesta trilha.

---

## Arquitetura e Fluxo

**1. Dataset congelado** → **2. Prompt versionado** → **3. Endpoint permitido** → **4. Validar schema** → **5. Avaliar por caso** → **6. MLflow + erros**

### Decisões Técnicas

- Separar 60 casos sintéticos em desenvolvimento, validação e teste com grupos de paráfrases no mesmo split.
- Mudar um fator por comparação e registrar chamadas com falha no denominador.
- Usar métricas determinísticas e rubrica humana; um LLM juiz é um sinal complementar.

---

## Validação e Critérios de Aceite

Critério de aceite futuro: executar as variantes nos mesmos casos, justificar a escolha na validação e abrir o teste final depois de fechar a configuração. Quantidades e métricas do plano são metas, não resultados.


---

## Limites Conhecidos e Próximos Passos

### Limites Conhecidos
- Execução aguarda seleção e validação do endpoint e de seus limites.
- Os exemplos de prompts são propostas; não têm desempenho medido.

### Próximos Incrementos
- Congelar dataset e rubrica com hash e versão.
- Comparar prompt direto, estruturado e few-shot sem consultar o teste durante o ajuste.

---

## Estrutura do Repositório

- `informacoes_tecnicas/PLANEJAMENTO.md`: Planejamento técnico detalhado (requisitos, contratos, arquitetura, MLflow e validação).
- `00_PLANEJAMENTO.py`: Versão em notebook Databricks do planejamento técnico.
- Notebooks e módulos em Python/PySpark (quando aplicável ao estágio de implementação do estudo).
- `evidencias/` (quando aplicável): Resumo estático das execuções validadas no Databricks.

## Links

- **Portfólio Completo:** [https://beatrizorlandi.github.io](https://beatrizorlandi.github.io)
- **Estudo de Caso no Site:** [https://beatrizorlandi.github.io/projetos/laboratorio-llms.html](https://beatrizorlandi.github.io/projetos/laboratorio-llms.html)
- **Planejamento Técnico no Site:** [https://beatrizorlandi.github.io/documentacao/laboratorio-llms.html](https://beatrizorlandi.github.io/documentacao/laboratorio-llms.html)

## Execução Validada no Databricks (05/11/2026)

- **Notebook executado:** `01_experimento_prompts_llms.py`
- **Tabela de resultados:** `workspace.estudos_ai.aurora_llm_resultados` (60 gerações: 20 casos × 3 variantes)
- **Resultados medidos via MLflow:**
  - `V1_direto_llama70b` (`databricks-meta-llama-3-3-70b-instruct`): 85% de acerto semântico, 0% no contrato JSON estrito sem markdown (`p50 = 890,9 ms`).
  - `V2_estruturado_llama70b` (`databricks-meta-llama-3-3-70b-instruct`): **100% de acerto** (20/20) e **100% JSON válido** (`p50 = 1.588,4 ms`).
  - `V3_estruturado_qwen80b` (`databricks-qwen3-next-80b-a3b-instruct`): **100% de acerto** (20/20) e **100% JSON válido** (`p50 = 528,5 ms`).
