# Importa bibliotecas essenciais para manipulação de dados, processamento de texto e aprendizado de máquina.
import pandas as pd  # Para manipulação de dados estruturados.
import numpy as np  # Para operações numéricas.
import re  # Para trabalhar com expressões regulares.
import tensorflow as tf  # Para construir e treinar modelos de deep learning.
import nltk  # Para tarefas de processamento de linguagem natural.
from sklearn.model_selection import GridSearchCV  # Para avaliar melhores parâmetros.
from sklearn.model_selection import train_test_split  # Para divisão de dados em conjuntos de treino e teste.
from xgboost import XGBClassifier  # Importa o classificador XGBoost.
from sklearn.metrics import classification_report  # Para gerar relatórios de classificação.

# Baixa o pacote de stopwords da biblioteca NLTK.
nltk.download('stopwords')
from nltk.corpus import stopwords  # Importa stopwords para remoção de palavras irrelevantes.

def load_data(filepath):
    return pd.read_csv(filepath)

def clean_text(text):
    text = re.sub(r'[\^\w\s]', '', str(text).lower())  # Remove pontuação e converte o texto para minúsculas.
    stop_words = set(stopwords.words('portuguese'))  # Define a lista de stopwords em português.
    words = [word for word in text.split() if word not in stop_words]  # Remove stopwords do texto.
    return ' '.join(words)  # Retorna o texto processado como uma única string.

def preprocess_data(data):
    data['cleaned_text'] = data['text'].apply(clean_text)
    return train_test_split(
        data['cleaned_text'],  # Texto processado.
        data['Hate.speech'],  # Rótulos binários (se é hate speech ou não).
        test_size=0.1,  # Define 10% dos dados como teste.
        random_state=42,  # Garante que a divisão seja reprodutível.
        stratify=data['Hate.speech']  # Garante que a divisão seja estratificada.
    )

def tokenize_and_pad_sequences(X_train, X_test):
    tokenizer = tf.keras.preprocessing.text.Tokenizer()  # Cria um tokenizador para converter palavras em números.
    tokenizer.fit_on_texts(X_train)  # Ajusta o tokenizador aos dados de treino.
    X_train_seq = tokenizer.texts_to_sequences(X_train)  # Converte o texto em sequências numéricas.
    X_test_seq = tokenizer.texts_to_sequences(X_test)  # Faz o mesmo para os dados de teste.
    X_train_pad = tf.keras.preprocessing.sequence.pad_sequences(X_train_seq, maxlen=100)  # Padroniza as sequências de treino para 100 tokens.
    X_test_pad = tf.keras.preprocessing.sequence.pad_sequences(X_test_seq, maxlen=100)  # Faz o mesmo para as sequências de teste.
    return X_train_pad, X_test_pad, tokenizer

def load_embeddings(filepath, tokenizer):
    embeddings_index = {}
    with open(filepath, encoding='utf-8') as f:  # Abre o arquivo de embeddings GloVe.
        for line in f:  # Itera sobre as linhas do arquivo.
            values = line.split()  # Divide cada linha em uma palavra e seus vetores.
            word = values[0]  # A primeira palavra é a palavra do vocabulário.
            try:
                coefs = np.asarray([float(val.replace(',', '.')) for val in values[1:]], dtype='float32')  # Converte os valores em floats.
                embeddings_index[word] = coefs  # Adiciona ao índice de embeddings.
            except ValueError:
                print(f"Ignoring line: {line.strip()}")  # Ignora linhas que não seguem o formato esperado.
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, 300))  # Inicializa uma matriz de zeros.
    for word, i in tokenizer.word_index.items():  # Itera sobre o vocabulário do tokenizador.
        embedding_vector = embeddings_index.get(word)  # Busca o vetor de embedding para a palavra.
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector  # Adiciona o vetor à matriz na posição correspondente.
    return embedding_matrix

def build_lstm_model(embedding_matrix, input_length):
    model = tf.keras.models.Sequential([  # Cria um modelo sequencial.
        tf.keras.layers.Embedding(input_dim=len(embedding_matrix),  # Tamanho do vocabulário.
                                  output_dim=300,  # Dimensão dos embeddings.
                                  weights=[embedding_matrix],  # Usa os embeddings pré-treinados.
                                  input_length=input_length,  # Comprimento das sequências de entrada.
                                  trainable=False),  # Embeddings não serão ajustados durante o treinamento.
        tf.keras.layers.LSTM(50, return_sequences=False),  # Adiciona uma camada LSTM com 50 unidades.
        tf.keras.layers.Dropout(0.5),  # Adiciona dropout para evitar overfitting.
        tf.keras.layers.Dense(1, activation='sigmoid')  # Camada densa com ativação sigmoidal para classificação binária.
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Compila o modelo com otimizador Adam e função de perda binária.
    return model

def train_lstm_model(model, X_train_pad, y_train):
    model.fit(X_train_pad, y_train, validation_split=0.1, epochs=10, batch_size=128)  # Treina o modelo.
    return model

def extract_intermediate_features(model, X_train_pad, X_test_pad):
    model.predict(X_train_pad[:1])  # Faz uma previsão inicial para garantir que os pesos estão carregados.
    intermediate_layer_model = tf.keras.models.Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)  # Cria um modelo que extrai a penúltima camada.
    X_train_features = intermediate_layer_model.predict(X_train_pad)  # Extrai representações para os dados de treino.
    X_test_features = intermediate_layer_model.predict(X_test_pad)  # Faz o mesmo para os dados de teste.
    return X_train_features, X_test_features

def train_xgb_model(X_train_features, y_train):
    xgb_model = XGBClassifier(eta=0.3, gamma=1, eval_metric='logloss')  # Inicializa o classificador XGBoost.
    xgb_model.fit(X_train_features, y_train)  # Treina o modelo XGBoost com os dados processados.
    return xgb_model

def evaluate_model(model, X_test_features, y_test):
    y_pred = model.predict(X_test_features)  # Faz previsões com o modelo.
    print(classification_report(y_test, y_pred))  # Gera um relatório de classificação detalhado.

def perform_grid_search(X_train_features, y_train):
    param_grid = {
        'eta': [0, 0.3, 1],  # Taxa de aprendizado
        'gamma': [0.1, 1, 10]  # Parâmetro de regularização
    }
    xgb_model = XGBClassifier(eval_metric='logloss')  # Cria o modelo base.
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='accuracy',  # Métrica de avaliação (pode ser alterada para 'f1', 'roc_auc', etc.)
        cv=10,  # Número de folds para validação cruzada.
        verbose=2,  # Nível de detalhamento do log.
        n_jobs=-1  # Utiliza todos os núcleos disponíveis.
    )
    grid_search.fit(X_train_features, y_train)  # Realiza o Grid Search nos dados de treino.
    print("Melhores parâmetros:", grid_search.best_params_)  # Exibe os melhores hiperparâmetros encontrados.
    print("Melhor f1-score:", grid_search.best_score_)  # Exibe a melhor pontuação obtida.
    return grid_search.best_estimator_

def main():
    data = load_data('2019-05-28_portuguese_hate_speech_hierarchical_classification.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    X_train_pad, X_test_pad, tokenizer = tokenize_and_pad_sequences(X_train, X_test)
    embedding_matrix = load_embeddings('glove_s300.txt', tokenizer)
    lstm_model = build_lstm_model(embedding_matrix, input_length=100)
    lstm_model = train_lstm_model(lstm_model, X_train_pad, y_train)
    X_train_features, X_test_features = extract_intermediate_features(lstm_model, X_train_pad, X_test_pad)
    xgb_model = train_xgb_model(X_train_features, y_train)
    evaluate_model(xgb_model, X_test_features, y_test)
    best_model = perform_grid_search(X_train_features, y_train)
    evaluate_model(best_model, X_test_features, y_test)

if __name__ == "__main__":
    main()