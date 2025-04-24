# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# Load dataset
df = pd.read_csv('heart1.csv')  # Pastikan file CSV berada di direktori yang sama dengan script ini
print(df.head())                # Tampilkan 5 baris pertama
print(df.describe())            # Ringkasan statistik
print(df.info())                # Informasi struktur data
print(df.isnull().sum())        # Cek missing values

# Pisahkan fitur dan target
X = df.drop(['output'], axis=1)
Y = df['output']
print(X.shape, Y.shape)         # Verifikasi dimensi

# Normalisasi fitur
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
print(X.head())                 # (Opsional) tampilkan data setelah normalisasi

# Split data menjadi train dan test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Inisialisasi dan training model
model = LogisticRegression()
model.fit(X_train, y_train)

print(model.coef_)

y_pred = pd.Series(model.predict(X_test))
y_test = y_test.reset_index(drop=True)
z = pd.concat([y_test, y_pred], axis=1)
z.columns = ['True', 'Prediction']
print (z.head())  # Tampilkan 5 baris pertama dari hasil prediksi

print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))  # Hitung akurasi
print("Precision: ", metrics.precision_score(y_test, y_pred))  # Hitung presisi
print("Recall: ", metrics.recall_score(y_test, y_pred))  # Hitung recall
print("F1 Score: ", metrics.f1_score(y_test, y_pred))  # Hitung F1 score

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

labels = ['0', '1']
fig, ax = plt.subplots()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels)
plt.yticks(tick_marks, labels)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
