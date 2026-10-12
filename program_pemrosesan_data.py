import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# 1. LOAD DATA
df = pd.read_csv('data.csv')

# 2. PRAPROSES DATA
# Drop kolom tidak relevan (ID dan kolom kosong)
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)
if 'Unnamed: 32' in df.columns:
    df.drop('Unnamed: 32', axis=1, inplace=True)

# Penanganan Missing Values dan Duplikasi
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Kategori (Encoding target ke Numerik)
df['diagnosis'] = df['diagnosis'].str.upper() 
le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis']) # B = 0, M = 1

# Outlier Detection (IQR) untuk 'area_mean'
Q1 = df['area_mean'].quantile(0.25)
Q3 = df['area_mean'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter data untuk menghapus outlier
df = df[(df['area_mean'] >= lower_bound) & (df['area_mean'] <= upper_bound)]

# 3. MENCEGAH DATA LEAKAGE & STANDARISASI
# Split data sebelum standarisasi (mencegah data leakage)
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standarisasi menggunakan pola dari data Train saja
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Mengubah hasil scaling kembali ke DataFrame untuk visualisasi
processed_df = pd.DataFrame(X_train_scaled, columns=X.columns)

# 4. MENAMPILKAN HASIL PRAPROSES
print("\n>>> 5 Teratas Data Hasil Pemrosesan <<<")
print(processed_df.head())
print("\n>>> 5 Terbawah Data Hasil Pemrosesan <<<")
print(processed_df.tail())

# 5. KLASIFIKASI NAIVE BAYES GAUSSIAN
gnb = GaussianNB()
gnb.fit(X_train_scaled, y_train)

# Prediksi dan Evaluasi
y_pred = gnb.predict(X_test_scaled)

print("\n>> Laporan Evaluasi Model <<")
print(f"Akurasi: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=['Jinak (B)', 'Ganas (M)']))