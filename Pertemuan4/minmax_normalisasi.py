# import libraries pandas dan sklearn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Upload data ke DataFrame
# Pastikan file CSV berada di direktori yang sama dengan script ini
data = pd.read_csv("data latihan pertemuan 3.csv")
print("Data Asli:")
print(data)

# Normalisasi dengan Min-Max Scaling dan Standardization
scaler_minmax = MinMaxScaler()
scaled_minmax = scaler_minmax.fit_transform(data[['Pengalaman']])
print("\nMin-Max Normalisasi:")
print(scaled_minmax)

# Normalisasi dengan StandardScaler (Z-Score Normalization)
scaler_standard = StandardScaler()
scaled_standard = scaler_standard.fit_transform(data[['Pengalaman']])
print("\nStandardScaler (Z-Score Normalisasi):")
print(scaled_standard)
