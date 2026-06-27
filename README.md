# 3rd IMDC - XGBSillas

Pipeline final para previsão probabilística semanal de dengue e chikungunya no
3rd Infodengue-Mosqlimate Dengue Challenge.

## Equipe

Time: XGBSillas

Integrante principal: Sillas Rocha da Costa, EMAp/FGV.

## Estrutura

```text
src/config.py                desafios, rodadas, caminhos, quantis e modelos finais
src/download.py              baixa os arquivos oficiais do FTP
src/validate_raw_data.py     valida colunas mínimas dos dados brutos
src/mapbiomas.py             baixa/processa MapBiomas anual
src/spatial.py               pesos e vizinhança espacial de UFs
src/build_dataset.py         monta os painéis semanais por desafio
src/features.py              cria a matriz supervisionada sem vazamento temporal
src/models.py                baseline histórico e modelos XGBoost quantílicos
src/metrics.py               WIS, MAE, cobertura e ordenação de quantis
src/backtest.py              validações retrospectivas 2022-2025
src/forecast.py              previsão final 2026-2027
src/submission.py            CSVs por localidade/rodada no formato do sprint
src/plots.py                 gráficos simples gerados pelo backtest

scripts/plot_final_validation_models.py
                            figuras do relatório e comparação entre modelos
scripts/fetch_external_predictions.py
                            opcional: baixa previsões externas para comparação

submission_workflow.ipynb    notebook operacional para validação, plots e envio
final_report.md              relatório técnico e científico
data/submissions/            único diretório de dados versionável
```

Arquivos gerados pela pipeline ficam em `data/raw`, `data/interim`,
`data/processed`, `data/results` e `data/submissions`. Apenas
`data/submissions` entra no Git. As demais pastas são cache local e podem
ser apagadas e regeneradas.

## O que é Necessário

Para a submissão, o caminho necessário é:

```text
src.build_dataset -> src.backtest -> src.forecast -> src.submission
```

Esses módulos puxam internamente os auxiliares realmente usados:

- `config.py`: fonte única dos desafios, rodadas, caminhos e modelos finais;
- `download.py`, `validate_raw_data.py`, `mapbiomas.py`, `spatial.py`: usados por
  `build_dataset.py`;
- `features.py`, `models.py`, `metrics.py`, `plots.py`: usados por backtest,
  forecast e validação dos CSVs;
- `utils.py`: logging, quantis e helpers pequenos.

Arquivos opcionais:

- `scripts/fetch_external_predictions.py`: baixa previsões de outro modelo para
  comparação visual. Não é usado para gerar a submissão.
- `scripts/plot_final_validation_models.py`: gera figuras agregadas e análises
  do relatório. O `backtest.py` chama esse script ao final.
- `src/diagnostics.py`: diagnósticos exploratórios de zeros, covariáveis,
  extremos e espaço. Pode ser removido sem quebrar a pipeline de submissão se o
  objetivo for manter apenas o código final.
- `submission_workflow.ipynb`: notebook de operação e envio; útil para revisar e
  submeter, mas não é dependência dos módulos `src`.

Arquivos gerados e ignorados:

- `data/raw`, `data/interim`, `data/processed`, `data/results`;
- `plot_submissions/`;
- caches Python/Jupyter e logs locais.

## Dados e Variáveis

O projeto usa os dados oficiais do FTP do IMDC: casos prováveis de dengue e
chikungunya, clima observado, população Datasus, oscilações oceânicas, regiões
de saúde e malhas territoriais. Também usa MapBiomas como fonte aberta de
variáveis anuais de uso e cobertura da terra.

O processamento está em `src/build_dataset.py`. As variáveis finais são criadas
em `src/features.py`: defasagens de casos, médias móveis, quantis históricos por
semana epidemiológica, indicadores sazonais, pressão regional, população e
variáveis ambientais MapBiomas.

## Dependências

O ambiente é definido em `pyproject.toml` e `uv.lock`. As bibliotecas principais
são: `pandas`, `numpy`, `pyarrow`, `epiweeks`, `xgboost`, `scikit-learn`,
`geopandas`, `libpysal`, `esda`, `matplotlib`, `rasterio`, `statsmodels`,
`mosqlient` e `python-dotenv`.

## Modelos e Incerteza

O backtest treina quatro componentes: `baseline`, `xgb_quantile`,
`xgb_quantile_log1p` e `xgb_residual`. A submissão usa um modelo final por
desafio, definido em `src/config.py`:

- dengue UF: `xgb_residual`;
- dengue cidade: `xgb_residual`;
- chikungunya UF: `xgb_quantile_log1p`;
- chikungunya cidade: `xgb_quantile`.

Os modelos XGBoost estimam diretamente os quantis 0.025, 0.05, 0.10, 0.25,
0.50, 0.75, 0.90, 0.95 e 0.975. Esses quantis geram a mediana e os intervalos
centrais de 50%, 80%, 90% e 95%. A ordem dos quantis e valores não negativos
são garantidos antes da escrita dos CSVs.

## Restrição Temporal

Cada rodada usa apenas dados disponíveis até a semana epidemiológica 25 do ano
de origem. A temporada prevista começa na EW41 e termina na EW40 do ano
seguinte. Essa regra fica centralizada em `src/features.py` e é usada tanto no
backtest quanto na previsão final.

## Rodar Pipeline Final

Pipeline mínima para gerar os CSVs finais:

```bash
uv sync
uv run python -m src.build_dataset
MODEL_DEVICE=cuda uv run python -m src.backtest
MODEL_DEVICE=cuda uv run python -m src.forecast
uv run python -m src.submission
```

Para rodar sem GPU:

```bash
MODEL_DEVICE=cpu uv run python -m src.backtest
MODEL_DEVICE=cpu uv run python -m src.forecast
```

Para gerar comparações com previsões externas no relatório:

```bash
export MOSQLIMATE_API_KEY="sua-chave"
uv run python scripts/fetch_external_predictions.py
MODEL_DEVICE=cuda uv run python -m src.backtest
```

## Saídas Principais

```text
data/results/predictions/
data/results/metrics/
data/results/models/
data/results/figures/model_comparison/
data/submissions/
```

Os CSVs finais para envio ficam em `data/submissions/{challenge}/{model}/{round}`.
Cada arquivo tem as colunas exigidas pelo sprint: `date`, `pred`, `lower_50`,
`upper_50`, `lower_80`, `upper_80`, `lower_90`, `upper_90`, `lower_95`,
`upper_95`.

A estrutura versionável de submissão fica limitada aos quatro modelos finais:

```text
data/submissions/dengue_uf/xgb_residual/{validation_1..4,final}/
data/submissions/dengue_city/xgb_residual/{validation_1..4,final}/
data/submissions/chikungunya_uf/xgb_quantile_log1p/{validation_1..4,final}/
data/submissions/chikungunya_city/xgb_quantile/{validation_1..4,final}/
```

## Referências

O método combina regressão quantílica com XGBoost e baselines sazonais
históricos. Detalhes, fórmulas, diagnósticos e resultados de validação estão em
`final_report.md`.
