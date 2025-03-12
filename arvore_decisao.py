#Manipulação dos dados
import pandas as pd
# Divide o conjunto de dados em duas partes: uma para treinamento e outra para teste.
from sklearn.model_selection import train_test_split
# Implementa a Arvore de decisão.
# export_graphviz # Exportação da arvore de decisão para o formato .dot, para visualização futura
from sklearn.tree import DecisionTreeClassifier, export_graphviz
# confusion_matrix: Calcula a matriz de confusão, que mostra o desempenho de um classificador, 
# comparando as previsões com os valores reais.
#Avaliar o desempenho de modelos de classificação (ex.: True Positives, False Positives, etc.).
# acuracu_score: Calcula a acurácia do modelo, ou seja, a porcentagem de previsões corretas.
from sklearn.metrics import confusion_matrix, accuracy_score
#Converte rótulos categóricos (como texto) em valores numéricos
# para que possam ser usados em modelos de machine learning.
from sklearn.preprocessing import LabelEncoder
#Um visualizador da matriz de confusão, tornando os resultados mais fáceis de interpretar, 
# exibindo-os graficamente.
from yellowbrick.classifier import ConfusionMatrix


#Ler os Dados
dados = pd.read_csv('Credit.csv')
print(dados.head())

#Formato da Matriz
previsores = dados.iloc[:,0:20].values
classe = dados.iloc[:,20].values


# Transformação dos atributos categóricos em atributos numéricos, 
# passando o índice de cada coluna categórica. Precisamos criar um objeto para cada atributo categórico, 
# pois na sequência vamos executar o processo de encoding novamente para o registro de teste
# Se forem utilizados objetos diferentes, o número atribuído a cada valor poderá ser diferente,
# o que deixará o teste inconsistente.
# Codificação de variáveis categóricas para variáveis numéricas.
labelencoder1 = LabelEncoder()

previsores[:,0] = labelencoder1.fit_transform(previsores[:,0])

labelencoder2 = LabelEncoder()
previsores[:,2] = labelencoder2.fit_transform(previsores[:,2])

labelencoder3 = LabelEncoder()
previsores[:, 3] = labelencoder3.fit_transform(previsores[:, 3])

labelencoder4 = LabelEncoder()
previsores[:, 5] = labelencoder4.fit_transform(previsores[:, 5])

labelencoder5 = LabelEncoder()
previsores[:, 6] = labelencoder5.fit_transform(previsores[:, 6])

labelencoder6 = LabelEncoder()
previsores[:, 8] = labelencoder6.fit_transform(previsores[:, 8])

labelencoder7 = LabelEncoder()
previsores[:, 9] = labelencoder7.fit_transform(previsores[:, 9])

labelencoder8 = LabelEncoder()
previsores[:, 11] = labelencoder8.fit_transform(previsores[:, 11])

labelencoder9 = LabelEncoder()
previsores[:, 13] = labelencoder9.fit_transform(previsores[:, 13])

labelencoder10 = LabelEncoder()
previsores[:, 14] = labelencoder10.fit_transform(previsores[:, 14])

labelencoder11 = LabelEncoder()
previsores[:, 16] = labelencoder11.fit_transform(previsores[:, 16])

labelencoder12 = LabelEncoder()
previsores[:, 18] = labelencoder12.fit_transform(previsores[:, 18])

labelencoder13 = LabelEncoder()
previsores[:, 19] = labelencoder13.fit_transform(previsores[:, 19])

# Divisão da base de dados entre treinamento e teste (30% para testar e 70% para treinar)
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(previsores,
                                                                  classe,
                                                                  test_size = 0.3,
                                                                  random_state = 0)
print(x_teste)  

#Metodo arvore de decisao
arvore = DecisionTreeClassifier()
#Treina o modelo
arvore.fit(x_treinamento, y_treinamento)

# Exportação da arvore de decisão para o formato .dot, para visualização futura
export_graphviz(arvore, out_file='tree.dot')

# Obtenção de previsões
previsoes = arvore.predict(x_teste)
print(previsoes)


#geração da matriz de confusão
#A matriz de confusão é uma ferramenta essencial para avaliar a performance de um modelo de classificação, 
# pois mostra não apenas os acertos do modelo (TP e TN), mas também os erros (FP e FN). 
# Isso ajuda a entender onde o modelo está errando e pode fornecer informações valiosas para ajustar o
# modelo ou o processo de treinamento

#gera a matriz de confusão, que compara os valores reais de y_teste com os valores previstos 
# pelo modelo (previsoes).Ela mostra quantas classificações o modelo acertou e quantas errou, 
# separando os erros em falsos positivos (FP) e falsos negativos (FN). Isso ajuda a entender 
# onde o modelo está performando bem e onde precisa ser ajustado.
confusao = confusion_matrix(y_teste, previsoes)
print(confusao)
# Visualização da matriz de confusão
v = ConfusionMatrix(DecisionTreeClassifier())
# v.fit(x_treinamento, y_treinamento) está treinando o classificador (DecisionTreeClassifier) 
# com os dados de treinamento x_treinamento (variáveis independentes) e y_treinamento (valores reais de saída).
v.fit(x_treinamento, y_treinamento)
#v.score(x_teste, y_teste) está avaliando o modelo treinado com os dados de teste x_teste 
# e comparando as previsões com os valores reais y_teste.
v.score(x_teste, y_teste)
# usada para gerar visualizações.
v.poof()


# calcula a taxa de acerto e a taxa de erro do modelo
taxa_acerto = accuracy_score(y_teste, previsoes)
taxa_erro = 1 - taxa_acerto
#Taxa de acerto: 0.6833333333333334, logo 68,3% de acertos
#Taxa de erro:0.31666666666666664, logo 31,6% de erro
print(f'Taxa de acerto: {taxa_acerto}\nTaxa de erro:{taxa_erro}')

