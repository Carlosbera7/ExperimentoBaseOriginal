# Experimento: Classificação de Discurso de Ódio em Português com LSTM e XGBoost

Este repositório contém a implementação de um experimento descrito no artigo (Fortuna, Paula, et al., 2019), que combina LSTM e XGBoost para a classificação de discurso de ódio em português. O método utiliza embeddings pré-treinados GloVe, uma LSTM para extração de representações e um modelo XGBoost para classificação final.

## Descrição do Experimento
O experimento segue as etapas descritas no artigo:

1. **Divisão de Dados**:
   - Utiliza validação cruzada com 10 folds combinada com validação holdout.
   - Parte dos dados é usada para validação cruzada e ajuste de parâmetros; a outra parte é reservada para testes.

2. **Pré-processamento de Texto**:
   - Remoção de stopwords usando NLTK.
   - Remoção de pontuação e transformação para minúsculas.

3. **Extração de Features**:
   - Uso de embeddings pré-treinados GloVe com 300 dimensões para português.
   - Tokenização e padding das sequências de entrada.

4. **Classificação**:
   - Construção de um modelo LSTM com:
     - Camada de embedding usando os pesos dos embeddings GloVe.
     - Uma camada LSTM com 50 unidades e dropout.
     - Uma camada densa final com ativação sigmoidal para classificação binária.
   - Extração das representações da penúltima camada da LSTM como entrada para o XGBoost.

5. **XGBoost**:
   - Treinamento do XGBoost com os seguintes parâmetros ajustáveis via grid search:
     - `eta`: [0, 0.3, 1]
     - `gamma`: [0.1, 1, 10]

## Implementação
O experimento foi implementado em Python 3.6 utilizando as bibliotecas:
- TensorFlow
- NLTK
- Gensim
- Scikit-learn
- XGBoost

O script principal executa as seguintes etapas:
1. Carregamento e pré-processamento dos dados.
2. Tokenização e padding das sequências de texto.
3. Carregamento dos embeddings GloVe.
4. Construção e treinamento do modelo LSTM.
5. Extração das representações intermediárias.
6. Treinamento e avaliação do XGBoost.
7. Busca de hiperparâmetros com validação cruzada.

## Resultados
Os resultados incluem:
- **Relatórios de métricas**: Precision, recall, f1-score e accuracy.
- **Melhores parâmetros do XGBoost** obtidos via grid search.

Exemplo de saída:
```

Melhores parâmetros: {'eta': 0.3, 'gamma': 0.1}
Melhor f1-score: 0.7851
              precision    recall  f1-score   support

           0       0.78      1.00      0.88       444
           1       0.50      0.01      0.02       123

    accuracy                           0.78       567
   macro avg       0.64      0.50      0.45       567
weighted avg       0.72      0.78      0.69       567
```
![Figure_1](https://github.com/user-attachments/assets/895e3251-73d5-452d-b8ab-5b4f0aaf18f6)

## Estrutura do Repositório
-  [`Scripts/SalvaParticoes.py`](https://github.com/Carlosbera7/ExperimentoBaseOriginal/blob/main/Script/ClassificadorOriginal.py): Script principal para executar o experimento.
- [`Data/`](https://github.com/Carlosbera7/ExperimentoBaseOriginal/tree/main/Data): Pasta contendo o conjunto de dados e o Embeddings GloVe pré-treinados (necessário para execução).


