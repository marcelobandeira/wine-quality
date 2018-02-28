
# coding: utf-8

# ## importando o dataset

# In[1]:


import pandas as pd
train = pd.read_csv('winequality-white.csv')


# In[2]:


train.head(5)


# ### Altera a coluna "quality" transformando-a em binaria. Se a nota eh maior ou igual a 7, "quality" recebe 1, senao recebe 0. Pular este passo para testar o resultado com multiplos valores para a coluna "quality"

# In[3]:


for dataset in [train]:
    dataset.loc[(dataset.quality < 7), 'quality'] = 0
    dataset.loc[(dataset.quality >= 7), 'quality'] = 1
    


# ### histograma da coluna "quality"

# In[4]:


# print train[(train['quality'>=8])]
print len(train[train.quality == 1])
train['quality'].plot.hist()


# In[5]:


import numpy as np
from collections import Counter
from sklearn.cross_validation import cross_val_score

X_df = train[['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'chlorides', 'pH', 'sulphates', 'alcohol']]
Y_df = train['quality']
#Xdummies_df = pd.get_dummies(X_df)
# X = Xdummies_df.values
X = X_df.values
Y = Y_df.values

porcentagem_de_treino = 0.9

tamanho_de_treino = int(porcentagem_de_treino * len(Y))

treino_dados = X[0:tamanho_de_treino]
treino_marcacoes = Y[0:tamanho_de_treino]

validacao_dados = X[tamanho_de_treino:]
validacao_marcacoes = Y[tamanho_de_treino:]

result_teste = []

def fit_and_predict_kf(nome, modelo, dados, marcacoes):
    k = 10
    scores = cross_val_score(modelo, dados, marcacoes, cv = k)
    
    taxa_de_acerto = np.mean(scores)
    result_teste.append({nome: taxa_de_acerto})

    print("Acerto %s: %.2f" % (nome, taxa_de_acerto*100))
    
result_validacao = []


# In[6]:


from sklearn.naive_bayes import GaussianNB
modeloGaussian = GaussianNB()
fit_and_predict_kf("GaussianNB", modeloGaussian, treino_dados, treino_marcacoes)


# In[7]:


from sklearn.naive_bayes import MultinomialNB
modeloMultinomial = MultinomialNB()
fit_and_predict_kf("kf MultinomialNB", modeloMultinomial, treino_dados, treino_marcacoes)


# In[8]:


from sklearn.ensemble import AdaBoostClassifier
modeloAdaBoost = AdaBoostClassifier()
fit_and_predict_kf("AdaBoostClassifier", modeloAdaBoost, treino_dados, treino_marcacoes)


# In[9]:


from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
modeloOneVsRest = OneVsRestClassifier(LinearSVC(random_state = 0))
fit_and_predict_kf("OneVsRest", modeloOneVsRest, treino_dados, treino_marcacoes)


# In[10]:


from sklearn.multiclass import OneVsOneClassifier
modeloOneVsOne = OneVsOneClassifier(LinearSVC(random_state = 0))
fit_and_predict_kf("OneVsOne", modeloOneVsOne, treino_dados, treino_marcacoes)


# In[11]:


from sklearn.ensemble import RandomForestClassifier
modeloRandomForest = RandomForestClassifier()
fit_and_predict_kf("RandomForest", modeloRandomForest, treino_dados, treino_marcacoes)


# In[12]:


print(result_teste)

