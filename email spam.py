#Bibliotecas necessárias
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Importando o arquivo do database
df = pd.read_csv('mail_data.csv')

data = df.where((pd.notnull(df)), '')
data.loc[data['Category'] == 'spam', 'Category',] = 0
data.loc[data['Category'] == 'ham', 'Category', ] = 1

X = data['Message']
Y = data['Category']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 3)

featureExtraction = TfidfVectorizer(min_df = 1, stop_words = 'english', lowercase = True)

X_train_features = featureExtraction.fit_transform(X_train)
X_test_features = featureExtraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

model = LogisticRegression()
model.fit(X_train_features, Y_train)

predictionOnTrainingData = model.predict(X_train_features)
predictionOnTestData = model.predict(X_test_features)

#Inserir email para teste de SPAM
verifySpam = ["Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"]
inputDataFeatures = featureExtraction.transform(verifySpam)

prediction = model.predict(inputDataFeatures)

if(prediction[0] == 1):
  print('NOT SPAM!')
else:
  print('SPAM!')

#Exibir tabela de emails / Quantidade de emails a ser exibida
data.head(100)