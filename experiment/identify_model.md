# Identifying the production embedding model

Nuxeo query vector: 1024-d, unit norm, float16 values.

| model | query format | own vectors · mean (min / median) | **Nuxeo vector · mean** | verdict |
|---|---|---|---|---|
| intfloat/multilingual-e5-large | query:  | 0.829 / 0.885 | **0.266** | no |
| intfloat/multilingual-e5-large | no prefix | 0.830 / 0.899 | **0.261** | no |
| intfloat/multilingual-e5-large-instruct | instruct | 0.861 / 0.893 | **0.218** | no |
| intfloat/multilingual-e5-large-instruct | no prefix | 0.878 / 0.908 | **0.217** | no |
| BAAI/bge-m3 | no prefix | 0.530 / 0.626 | **0.515** | **match** |
| Snowflake/snowflake-arctic-embed-l-v2.0 | query:  | 0.235 / 0.412 | **0.167** | no |
| Snowflake/snowflake-arctic-embed-l-v2.0 | no prefix | 0.401 / 0.524 | **0.124** | no |
| Qwen/Qwen3-Embedding-0.6B | query prompt | 0.245 / 0.402 | **-0.026** | no |
| Qwen/Qwen3-Embedding-0.6B | no prefix | 0.509 / 0.641 | **-0.025** | no |
