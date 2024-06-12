import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Membaca file .csv dengan encoding yang sesuai
try:
    data = pd.read_csv('2009.csv', encoding='utf-8')
except UnicodeDecodeError:
    data = pd.read_csv('2009.csv', encoding='ISO-8859-1')

# Menampilkan beberapa baris pertama data
print(data.head())

# Menghapus kolom yang tidak relevan dan membuat label prediksi
data = data[['KEDALAMAN', 'Mag']]  # Sesuaikan dengan kolom yang relevan

# Mengonversi kolom KEDALAMAN dan Mag ke tipe data numerik
data['KEDALAMAN'] = pd.to_numeric(data['KEDALAMAN'].str.replace(' Km', ''), errors='coerce')
data['Mag'] = pd.to_numeric(data['Mag'].str.replace(' SR', ''), errors='coerce')

# Menghapus baris dengan nilai NaN
data.dropna(inplace=True)

# Membuat label prediksi
data['Tsunami'] = (data['Mag'] >= 7.0) & (data['KEDALAMAN'] <= 50)  # Contoh label prediksi

# Membagi data menjadi fitur (X) dan label (y)
X = data[['KEDALAMAN', 'Mag']]
y = data['Tsunami']

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Melatih model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Memprediksi data uji
y_pred = model.predict(X_test)

# Mencetak laporan klasifikasi
print(classification_report(y_test, y_pred))

# Menyimpan model
joblib.dump(model, 'model_tsunami.pkl')
print("Model saved as model_tsunami.pkl")

#akurasi data catatan 