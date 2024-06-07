import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import re
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import keras
from keras.layers import Dense, Dropout, Input
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier

def preprocess(df):
    df = df.copy()
   
    def extract_title(x):
        pat = r",\s([^ .]+)\.?\s+"
        title_match = re.search(pat, x)
        if title_match:
            return title_match.group(1)
         
    def normalize_name(x):
        x = re.sub(r",\s([^ .]+)\.?\s+", ", ", x)
        return " ".join([v.strip(",()[].\"'") for v in x.split(" ")])
    
    def ticket_number(x):
        return x.split(" ")[-1]
        
    def ticket_item(x):
        items = x.split(" ")
        if len(items) == 1:
            return "NONE"
        return "_".join(items[0:-1])
    
    df["Title"] = df["Name"].apply(extract_title)
    df["Name"] = df["Name"].apply(normalize_name)
    df["Ticket_number"] = df["Ticket"].apply(ticket_number)
    df["Ticket_item"] = df["Ticket"].apply(ticket_item)   

    return df

def create_model(optimizer='adam', dropout_rate=0.0):
    model = Sequential()
    model.add(Dense(units=16, input_shape=(X_train_scaled.shape[1],), activation='relu'))
    model.add(Dense(units=32, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=64, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=32, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

#Wczytywanie danych
guess = pd.read_csv('gender_submission.csv', header=0)
train_df = pd.read_csv('train.csv', header=0)
test_df = pd.read_csv('test.csv', header=0)
print(train_df.head(10))

#Przygotowanie danych
preprocessed_train_df = preprocess(train_df)
preprocessed_test_df = preprocess(test_df)

train_df["Title"] = preprocessed_train_df["Title"]
train_df["Ticket_number"] = preprocessed_train_df["Ticket_number"]
train_df["Ticket_item"] = preprocessed_train_df["Ticket_item"]
test_df["Title"] = preprocessed_test_df["Title"]
test_df["Ticket_number"] = preprocessed_test_df["Ticket_number"]
test_df["Ticket_item"] = preprocessed_test_df["Ticket_item"]

print(preprocessed_train_df.head(5))

# Kategoryzacja danych
title_mapping = {
    "Mr": "Mr",
    "Miss": "Miss",
    "Mrs": "Mrs",
    "Master": "Master",
    "Dr": "Rare",
    "Rev": "Rare",
    "Col": "Rare",
    "Major": "Rare",
    "Mlle": "Miss",
    "Countess": "Rare",
    "Ms": "Miss",
    "Lady": "Rare",
    "Jonkheer": "Rare",
    "Don": "Rare",
    "Mme": "Mrs",
    "Capt": "Rare",
    "Sir": "Rare",
    "Dona": "Rare"
}

train_df['Title'] = train_df['Title'].map(title_mapping)
test_df['Title'] = test_df['Title'].map(title_mapping)

# Sprawdzenie wyników
print(train_df['Title'].value_counts())
print(test_df['Title'].value_counts())

# Podstawowe informacje o danych
print("Podstawowe informacje o danych:")
print(train_df.info())

# Statystyki podsumowujące dla danych numerycznych
print("\nStatystyki podsumowujące dla danych numerycznych:")
print(train_df.describe())

# Rozkłady cech kategorycznych
print("\nRozkłady cech kategorycznych:")
print(train_df['Sex'].value_counts())
print(train_df['Pclass'].value_counts())
print(train_df['Embarked'].value_counts())

# Wizualizacja rozkładów cech
plt.figure(figsize=(16, 8))

plt.subplot(2, 3, 1)
train_df['Age'].hist(bins=20)
plt.title('Rozkład wieku')
plt.xlabel('Wiek')
plt.ylabel('Liczba pasażerów')

plt.subplot(2, 3, 2)
train_df['Fare'].hist(bins=20)
plt.title('Rozkład opłat')
plt.xlabel('Opłata')
plt.ylabel('Liczba pasażerów')

plt.subplot(2, 3, 3)
train_df['Pclass'].value_counts().plot(kind='bar')
plt.title('Rozkład klasy')
plt.xlabel('Klasa')
plt.ylabel('Liczba pasażerów')

plt.subplot(2, 3, 4)
train_df['Embarked'].value_counts().plot(kind='bar')
plt.title('Rozkład portu')
plt.xlabel('Port')
plt.ylabel('Liczba pasażerów')

plt.subplot(2, 3, 5)
train_df['Sex'].value_counts().plot(kind='bar')
plt.title('Rozkład płci')
plt.xlabel('Płeć')
plt.ylabel('Liczba pasażerów')

plt.tight_layout()
plt.show()

# Analiza brakujących wartości
print("\nAnaliza brakujących wartości:")
print(train_df.isnull().sum())

# Korelacja między cechami a przetrwaniem
columns_to_exclude = ['PassengerId','Name', 'Sex', 'Title', 'Ticket','Ticket_number', 'Ticket_item', 'Cabin', 'Embarked']
train_df_dropped = train_df.drop(columns_to_exclude, axis=1)
print(train_df_dropped.corr()['Survived'])

# Korelacja między pclass a survived
plt.figure(figsize=(8, 6))
sns.barplot(x='Pclass', y='Survived', data=train_df)
plt.title('Zależność między klasą a przetrwaniem')
plt.show()

# Korelacja między sex a survived
plt.figure(figsize=(8, 6))
sns.barplot(x='Sex', y='Survived', data=train_df)
plt.title('Zależność między płcią a przetrwaniem')
plt.show()

print("Najsilniejsza korelacja dodatnia występuję pomiędzy ceną biletu a przetrwaniem, natomiast najsilniejsza korelacja ujemna występuję pomiędzy klasą a przetrwaniem.")
print("Korelacja między wiekiem a przetrwaniem jest niewielka, ale przetrwanie jest bardziej prawdopodobne dla młodszych osób.")
print("Korelacja między liczbą rodzeństwa/małżonków a przetrwaniem jest niewielka, ale przetrwanie jest bardziej prawdopodobne dla osób podróżujących z niewielką rodziną.")
print("Istnieje bardzo silna zależność między płcią a przetrwaniem. Przetrwanie jest dużo bardziej prawdopodobne dla kobiet.")

# Mapa korelacji
correlation = train_df_dropped.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Mapa cieplna korelacji')
plt.show()

# Korelacja tytułu z przetrwaniem
sns.countplot(data=train_df, x='Title', hue='Survived')
plt.title('Korelacja tytułu z przetrwaniem')
plt.show()

# Test chi-kwadrat dla zmiennych kategorycznych
contingency_table = pd.crosstab(train_df['Pclass'], train_df['Survived'])
chi2, p, _, _ = chi2_contingency(contingency_table)
print(f"P-value dla zależności między Pclass a Survived: {p}")

contingency_table = pd.crosstab(train_df['Sex'], train_df['Survived'])
chi2, p, _, _ = chi2_contingency(contingency_table)
print(f"P-value dla zależności między Sex a Survived: {p}")

# Wykresy pomocnicze do uzupełnienia brakujących wartości
# #średnia i mediana dla wieku 
# mean_age = train_df['Age'].mean()
# median_age = train_df['Age'].median()

# print(f"Średnia wieku: {mean_age}")
# print(f"Mediana wieku: {median_age}")

# # Wykres zależności między opłatą (Fare) a portem zaokrętowania (Embarked)
# plt.subplot(1, 2, 1)
# sns.barplot(x='Embarked', y='Fare', data=train_df)
# plt.title('Zależność między opłatą a portem zaokrętowania')

# # Wykres zależności między klasą (Pclass) a portem zaokrętowania (Embarked)
# plt.subplot(1, 2, 2)
# sns.barplot(x='Embarked', y='Pclass', data=train_df)
# plt.title('Zależność między klasą a portem zaokrętowania')

# plt.tight_layout()
# plt.show()

# Uzupełnienie brakujących wartości
# Uzupełnienie brakujących wartości dla kolumny "Age" medianą
median_age = train_df['Age'].median()
train_df['Age'] = train_df['Age'].fillna(median_age)
median_age = test_df['Age'].median()
test_df['Age'] = test_df['Age'].fillna(median_age)

# Uzupełnienie brakujących danych w kolumnie "Title" najczęściej występującą wartością w danej klasie podróżnej ("Pclass")
most_common_title = train_df.groupby('Pclass')['Title'].apply(lambda x: x.mode().iloc[0])
train_df['Title'] = train_df.apply(lambda row: most_common_title[row['Pclass']] if pd.isnull(row['Title']) else row['Title'], axis=1)
most_common_title = test_df.groupby('Pclass')['Title'].apply(lambda x: x.mode().iloc[0])
test_df['Title'] = test_df.apply(lambda row: most_common_title[row['Pclass']] if pd.isnull(row['Title']) else row['Title'], axis=1)

# Uzupełnienie brakujących wartości dla kolumny "Embarked" najczęściej występującą wartością w danej klasie podróżnej ("Pclass")
most_common_embarked = train_df.groupby('Pclass')['Embarked'].apply(lambda x: x.mode().iloc[0])
train_df['Embarked'] = train_df.apply(lambda row: most_common_embarked[row['Pclass']] if pd.isnull(row['Embarked']) else row['Embarked'], axis=1)
most_common_embarked = test_df.groupby('Pclass')['Embarked'].apply(lambda x: x.mode().iloc[0])
test_df['Embarked'] = test_df.apply(lambda row: most_common_embarked[row['Pclass']] if pd.isnull(row['Embarked']) else row['Embarked'], axis=1)

print("\nSprawdzenie brakujących wartości:")
print(train_df.isnull().sum())

# Zdefiniowanie niepotrzebnych kolumn
columns_to_drop = ['Ticket', 'PassengerId', 'Ticket_number', 'Ticket_item', 'Cabin', 'Name']

# Przekształcanie danych kategorycznych
categorical_columns = ['Sex', 'Title', 'Embarked']
train_df = pd.get_dummies(train_df, columns=categorical_columns)
test_df = pd.get_dummies(test_df, columns=categorical_columns)

# Podział danych na zbiór treningowy i testowy

X_train = train_df.drop('Survived', axis=1)
Y_train = train_df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X_train,Y_train,test_size=0.2,random_state=42)

# Standaryzacja danych
X_train = X_train.drop(columns_to_drop, axis=1)
X_test = X_test.drop(columns_to_drop, axis=1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# Tworzenie modelu
model = Sequential()
model.add(Dense(units=16, input_shape=(X_train_scaled.shape[1],), activation='relu'))
model.add(Dense(units=32, activation='relu', kernel_initializer='glorot_uniform'))
model.add(tf.keras.layers.BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(units=64, activation='relu', kernel_initializer='glorot_uniform'))
model.add(Dropout(0.2))
model.add(Dense(units=32, activation='relu', kernel_initializer='glorot_uniform'))
model.add(Dropout(0.3))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss = tf.keras.losses.binary_crossentropy, optimizer = tf.keras.optimizers.Adam(), metrics = ['acc'])

# Trenowanie modelu
model.fit(X_train_scaled, y_train, batch_size = 32, verbose = 2, epochs = 50)

# Przewidywanie
predict = model.predict(X_test_scaled)
predict = (predict > 0.5).astype(int).ravel()

# Ocena modelu
Y_pred_rand = (model.predict(X_train) > 0.5).astype(int)
print('Precision : ', np.round(metrics.precision_score(y_train, Y_pred_rand)*100,2))
print('Accuracy : ', np.round(metrics.accuracy_score(y_train, Y_pred_rand)*100,2))

# #Optymalizacja i dostrojenie modelu
# param_grid = {
#     'batch_size': [16, 32, 64],
#     'epochs': [20, 30, 50],
#     'optimizer': ['adam', 'rmsprop'],
# }

# model = KerasClassifier(build_fn=create_model)

# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
# grid_result = grid.fit(X_train_scaled, y_train)

# print("Najlepsze wyniki: ", grid_result.best_score_)
# print("Najlepsze parametry: ", grid_result.best_params_)

#Trenowanie modelu z optymalnymi parametrami
model.compile(loss = tf.keras.losses.binary_crossentropy, optimizer = tf.keras.optimizers.RMSprop(), metrics = ['acc'])
model.fit(X_train_scaled, y_train, batch_size = 32, verbose = 2, epochs = 30)

# Przewidywanie
predict = model.predict(X_test_scaled)
predict = (predict > 0.5).astype(int).ravel()

# Ocena modelu
Y_pred_rand = (model.predict(X_train) > 0.5).astype(int)
print('Precision : ', np.round(metrics.precision_score(y_train, Y_pred_rand)*100,2))
print('Accuracy : ', np.round(metrics.accuracy_score(y_train, Y_pred_rand)*100,2))

print("Do dokonania optymalizacji i dostosowaniu hipermarametrów zwykorzystałam przeszukiwania siatki GridSearch.")
print("Teoretycznie model osiągnął dokładność na poziomie 85%, jednak w praktyce wyniki są gorsze i są porównywalne do modelu bez optymalizacji.")
print("Możliwe, że model jest nadmiernie dopasowany do danych treningowych, co prowadzi do przetrenowania modelu.")
print("Przykładowe wyniki:")
print("Model przed optymalizacją: Precision: 68.46 Accuracy: 70.08\nModel po optymalizacji: Precision: 53.24 Accuracy: 65.03")
print("Model przed optymalizacją: Precision: 61.98 Accuracy: 68.82\nModel po optymalizacji: Precision: 66.45 Accuracy: 69.52 Precision")
print("Aby uzyskać bardziej rzetelne oszacowania wydajności modelu można zastosować zbiór walidacyjny lub walidację krzyżową.")