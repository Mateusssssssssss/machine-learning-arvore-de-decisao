# machine learning arvore de decisao
# Previsão de Pagadores - Árvore de Decisão

Este projeto utiliza a técnica de **árvore de decisão** para prever se um cliente é **bom** ou **mau pagador**, com base em um conjunto de dados contendo diversas informações financeiras.

## 📌 Tecnologias Utilizadas
- Python
- Pandas
- Scikit-learn
- Yellowbrick

## 📂 Estrutura do Projeto
- `arvore_decisao.py`: Script principal com todo o fluxo de tratamento de dados e modelagem.
- `Credit.csv`: Base de dados usada no treinamento do modelo.
- `tree.dot`: Arquivo gerado com a exportação da árvore de decisão.
- `README.md`: Documentação do projeto.

## 📊 Fluxo de Trabalho
1. **Leitura e exploração dos dados** com `pandas`.
2. **Pré-processamento**: Conversão de variáveis categóricas para numéricas com `LabelEncoder`.
3. **Divisão dos dados**: 70% para treinamento e 30% para teste com `train_test_split`.
4. **Criação do modelo** de árvore de decisão usando `DecisionTreeClassifier`.
5. **Treinamento do modelo**.
6. **Exportação da árvore de decisão** para um arquivo `.dot`.
7. **Predição dos dados de teste**.
8. **Avaliação do modelo**:
   - Matriz de confusão
   - Taxa de acerto e erro

## Resultados Obtidos
- **Taxa de acerto**: 68,3%
- **Taxa de erro**: 31,6%



## 🔗 Referências
- [Scikit-learn Documentation](https://scikit-learn.org/stable/user_guide.html)
- [Yellowbrick Documentation](https://www.scikit-yb.org/en/latest/)

## 📜 Licença
Este projeto é de livre uso para fins educacionais.


