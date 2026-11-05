# Databricks notebook source

# MAGIC %md
# MAGIC # 04 — Bancada de prompts e experimentos com LLMs
# MAGIC
# MAGIC **Estado:** planejado. **Competência:** desenhar experimentos, comparar configurações e justificar uma escolha. **Dependências:** endpoint permitido ou modo replay explícito; corpus do projeto 03 para tarefas RAG. Padrões gerais se aplicam.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Objetivo e requisitos
# MAGIC
# MAGIC Medir se um prompt, modelo ou fluxo resolve melhor a tarefa, com qualidade, latência e consumo rastreáveis.
# MAGIC
# MAGIC | ID | Requisito | Aceite futuro |
# MAGIC | --- | --- | --- |
# MAGIC | RF01 | Congelar um dataset com splits | IDs, gabaritos, grupos de intenção e hash preservados |
# MAGIC | RF02 | Versionar prompts completos | Texto, variáveis, exemplos e versão recuperáveis |
# MAGIC | RF03 | Executar configurações comparáveis | Mesmos casos, critérios e orçamento por comparação |
# MAGIC | RF04 | Validar saídas sem depender só de outro LLM | Schema e métricas determinísticas presentes |
# MAGIC | RF05 | Calibrar julgamentos subjetivos | Rubrica e amostra anotada por pessoa |
# MAGIC | RF06 | Registrar erros e execuções incompletas | Falhas entram no relatório e no denominador |
# MAGIC | RNF01 | Evitar overfitting no teste | Teste final consultado após fechar a configuração |
# MAGIC | RNF02 | Limitar custo experimental | Quantidade máxima de chamadas e timeout definidos |
# MAGIC
# MAGIC Tarefas iniciais: classificação de dúvidas, extração de campos e resposta RAG. Não misturar os escores dessas tarefas em uma média sem pesos/justificativa.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dados e contratos
# MAGIC
# MAGIC 60 casos sintéticos, distribuídos entre as três tarefas e incluindo ambiguidades e erros; 30 dev, 15 val, 15 teste. Reportar métricas por tarefa e avisar que a amostra por tarefa é pequena. Para comparação publicável mais estável, planejar expansão para pelo menos 60 casos por tarefa após fechar o processo.
# MAGIC
# MAGIC Contrato: `case_id`, `task`, `inputs`, `expected_output`, `expected_status`, `group_id`, `split`, `dataset_version`. Exemplos few-shot vêm apenas do desenvolvimento; o mesmo caso e suas paráfrases não atravessam splits.
# MAGIC
# MAGIC Resultado por caso: `experiment_id`, `run_id`, `case_id`, `variant_id`, `response`, `schema_valid`, `scores`, `error_type`, `latency_ms`, `tokens`, `retry_count`. Tabela proposta: `workspace.estudos_ai.aurora_llm_resultados`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Design experimental e fluxo
# MAGIC
# MAGIC ```mermaid
# MAGIC flowchart LR
# MAGIC     D[Dataset congelado] --> V[Selecionar variante]
# MAGIC     P[Prompt e config] --> V
# MAGIC     V --> C[Invocar endpoint ou replay]
# MAGIC     C --> S[Validar schema e conteúdo]
# MAGIC     S --> A[Avaliar por caso]
# MAGIC     A --> M[MLflow e tabela de resultados]
# MAGIC     M --> H[Revisão humana de erros]
# MAGIC     H --> F[Selecionar na validação e testar uma vez]
# MAGIC ```
# MAGIC
# MAGIC Escolher primeiro dois endpoints realmente disponíveis na conta, identificados por nome e configuração observados; não prescrever um modelo inexistente no workspace. Manter o endpoint fixo na comparação de prompts. Se o provedor não expuser uma revisão imutável, guardar nome, data e metadados acessíveis e declarar a limitação de reprodução.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stack
# MAGIC
# MAGIC | Lib/componente | Função |
# MAGIC | --- | --- |
# MAGIC | `databricks-sdk` | Integração com o endpoint disponível |
# MAGIC | `pydantic` / `jsonschema` | Contratos e validação estrutural |
# MAGIC | `scikit-learn`, `numpy`, `pandas` | Métricas e análises no conjunto pequeno |
# MAGIC | MLflow Tracking | Runs, configurações e artefatos |
# MAGIC | MLflow GenAI / Tracing, quando compatível | Avaliação/traces; fallback em resultados Delta e métricas convencionais |
# MAGIC | `matplotlib` | Gráficos de qualidade × latência e erros |
# MAGIC
# MAGIC Não instalar SDK de provedor que o endpoint não exige. O avaliador BEM do projeto 08 é um comparador adicional para subset em inglês; não substitui a rubrica de segurança, citação ou autorização.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Experimentos e MLflow
# MAGIC
# MAGIC Experimentos propostos: `.../04_laboratorio_llms/experimentos/classificacao`, `/extracao` e `/rag`. Uma run por variante, com resultados por caso como artefato/tabela referenciada. Traces devem carregar `case_id` e `variant_id`.
# MAGIC
# MAGIC | ID | Comparação | O que permanece fixo |
# MAGIC | --- | --- | --- |
# MAGIC | L01 | Prompt direto × estruturado | Modelo, casos e saída esperada |
# MAGIC | L02 | Zero-shot × 3 exemplos | Exemplos só do dev; mesmo teste e endpoint |
# MAGIC | L03 | Endpoint A × endpoint B | Prompt fechado e dataset iguais |
# MAGIC | L04 | Um chamado × geração + revisão | Mesmo orçamento documentado e critérios; contar ambas as chamadas |
# MAGIC | L05 | Juiz LLM × revisão humana | Mesmas respostas e rubrica; medir discordâncias |
# MAGIC
# MAGIC Não executar todo o produto cartesiano. Primeira rodada: até 4 variantes × 15 casos de validação = 60 gerações; segunda rodada, no máximo mais 60 para repetições ou tarefa adicional, aprovadas pelo orçamento disponível. Uma avaliação por LLM também consome chamada e entra nesse total. Três repetições em um subset de 10 casos medem variabilidade quando houver aleatoriedade, substituindo parte do orçamento.
# MAGIC
# MAGIC Parâmetros/tags: prompt hash, dataset/split, modelo/endpoint, temperatura e limite de tokens se suportados, política de retry, rubrica e release. Artefatos: prompts, manifesto, respostas, confusão, erros e ficha de decisão.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Métricas, seleção e validação
# MAGIC
# MAGIC - Classificação: macro-F1, confusão, acerto por classe e status de esclarecimento.
# MAGIC - Extração: JSON válido, precisão/recall por campo e exatidão dos campos críticos.
# MAGIC - RAG: suporte, correção, completude, validade das citações e abstenção.
# MAGIC - Operação: taxa de timeout/erro, latência p50/p95, tokens conhecidos e consumo estimado com fórmula/data quando necessário. Latência p95 em amostra pequena é apenas descritiva.
# MAGIC
# MAGIC Selecionar na validação: primeiro passar checks obrigatórios (schema, autorização e citações); entre variantes elegíveis, priorizar qualidade da tarefa e depois consumo/latência. Definir o ganho mínimo de interesse antes de executar. Reportar resultados por caso e bootstrap por grupo de intenção se houver amostra suficiente; não anunciar superioridade estatística com poucos exemplos.
# MAGIC
# MAGIC Casos obrigatórios: JSON malformado, campo ausente, recusa indevida, instrução de desvio na pergunta, erro de ferramenta/endpoint, referência inventada e resposta truncada. Limitar a 2 retries para falha transitória e registrar cada tentativa. Não repetir por uma nota baixa até obter uma resposta boa.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Plano de entrega e demonstração
# MAGIC
# MAGIC Etapas: dataset/rubrica → prompts → executor limitado → checks → MLflow → análise de erro → escolha na validação → teste final. Notebooks planejados: `01_dataset`, `02_variantes`, `03_execucao`, `04_avaliacao`, `05_decisao`.
# MAGIC
# MAGIC Entregáveis: tabela de ablação, gráficos, amostra anotada, ficha do prompt escolhido e limitações. **Demonstração:** mostrar duas respostas convincentes, revelar a avaliação por caso e explicar por que a configuração escolhida vence ou por que a evidência ainda é insuficiente.
# MAGIC
# MAGIC Riscos: vazamento do teste, juiz enviesado, revisão móvel do endpoint e custo oculto de retries/judges. As regras de split, calibração e contagem de chamadas devem aparecer no relatório final.
# MAGIC
# MAGIC Referências: [avaliação de LLMs e agentes](https://mlflow.org/docs/latest/genai/eval-monitor/), [Tracing](https://mlflow.org/docs/latest/genai/tracing/) e [Tracking](https://mlflow.org/docs/latest/ml/tracking/).
