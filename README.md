# ILLM-MVP4TS

This repository provides a paper-oriented implementation of **A Knowledge-Guided Multi-Agent Large Language Model Framework for Interpretable Multivariate Time Series Forecasting in Complex Process Industries.** The framework combines knowledge-guided relationship discovery, multi-view prompting, parameter-efficient forecasting, and retrieval-augmented explanations.

Regarding code availability, we have provided selected parts of the implementation for reference. The complete source code will be publicly released upon publication of the paper.

## Experiment Steps

1. **Loading the Pre-trained Model.** The pre-trained language model and its tokenizer are loaded as the forecasting backbone. The default model is DeepSeek-R1-Distill-Qwen-1.5B, which provides the language representations used to integrate textual process knowledge with numerical time-series inputs.

2. **Preparing the Dataset.** Multivariate process measurements are arranged chronologically and divided into training, validation, and test sets with a 70%/10%/20% split. Each input window contains all process variables, while the output corresponds to a designated target variable. The default setting uses 96 historical observations to predict the next time step. Variable descriptions and domain documents provide the accompanying process knowledge.

3. **Constructing the Knowledge-Guided Relationship Graph.** Granger tests identify an initial directed dependency graph using the training data. The DA-Agent retrieves and summarizes domain evidence about the process and its variables. The CC-Agent evaluates candidate relationships drawn from the statistical graph and evidence-supported domain relations. Its confidence-based decisions guide graph refinement, producing the variable relationships used in the forecasting prompts.

4. **Building Multi-View Prompts.** Each historical window is represented through four complementary views: domain knowledge, variable relationships, temporal trends, and statistical characteristics. These descriptions are combined into a structured text prompt. Numerical inputs undergo reversible instance normalization, temporal projection, and attention across variables, followed by cross-attention with pre-trained word embeddings to align the numerical and language representations.

5. **Fine-tuning the Forecasting Model.** The text prompt and aligned numerical representations are jointly processed by the forecasting model. The original backbone parameters remain frozen, while LoRA adapters and forecasting-specific layers are trained using mean squared error. The default configuration uses a LoRA rank of 16, a learning rate of 0.001, a batch size of 128, and 100 epochs. The checkpoint with the lowest validation error is selected for testing.

6. **Evaluating Forecasting Performance.** The selected model generates predictions on the held-out test set. Predictions in the target variable's original units are compared with ground-truth values using mean absolute error (MAE), root mean squared error (RMSE), the coefficient of determination (R²), and hit rate (HR). HR is the percentage of predictions with an absolute error below 0.04.

7. **Generating Retrieval-Augmented Explanations.** Domain documents are divided into overlapping passages and encoded as dense embeddings. Retrieval queries are generated from the historical context, forecast, and operator question. Relevant passages are retrieved, merged, and deduplicated to support an explanation with evidence references. Future ground-truth values are excluded from the explanation context.

8. **Evaluating Explanation Quality.** Retrieval and explanation quality are evaluated using mean reciprocal rank (MRR), ROUGE-L, question coverage, and consistency with validated evidence, where the required annotations are available. Explanation accuracy and professionalism are assessed through explicit human annotations and expert ratings.

## Layout

| Directory | Purpose |
| --- | --- |
| `illm_mvp4ts/` | Relationship graph construction, multi-view prompts, multivariate forecasting, retrieval-augmented explanations, and evaluation |
| `Graph-Multi-Agent/` | Original graph implementation, agent prompts, and Granger tests |
| `forecasting/` | Original forecasting implementation, LoRA modules, and prompt-fusion components |
| `configs/` | Default training configuration and variable metadata template |
| `tools/` | Source release packaging utilities |

See [LICENSES.md](LICENSES.md) for license information and third-party notices.
