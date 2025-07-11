# **Laporan Proyek Machine Learning: Supplement Demand Forcasting**

## **0. Domain Proyek**

Saat ini industri kesehatan menjadi bagian penting dalam kehidupan masyarakat modern, terutama pasca pandemi COVID-19. Hal ini menunjukkan adanya peningkatan kesadaran masyarakat terhadap kesehatan dan mendorong permintaan produk-produk kesehatan seperti vitamin, mineral dan herbal. Namun, di tengah pertumbuhan ini, perusahan di industri kesehatan mengalami tantangan besar dalam memprediksi kebutuhan produk-produk tersebut secara akurat. Pola ini dapat dipengaruhi oleh beberapa faktor seperti harga, diskon, platform penjualan dan lokasi geografis.

Masalah yang timbul adalah terjadinya overstock (stok berlebihan) dan stockout(kehabisan stock) yang menyebabkan kerugian finansial bagi perusahaan karena kesalahan dalam prediksi permintaan pasar. Menurut penelitian oleh Carbonneau et al. (2008) dalam European Journal of Operational Research, kesalahan prediksi permintaan dapat menyebabkan biaya tambahan sebesar 20-30% dari total biaya logistik, terutama untuk produk musiman atau berbasis kesehatan. Oleh karena itu, untuk meminimalisir kerugian finansial dari kesalahan prediksi, perlu diselesaikan dengan membangun model machine learning. Model yang dibangun model prediksi berbasis data historis agar lebih efektif dan efisien.

**Referensi:**
- Carbonneau, R., Laframboise, K., & Vahidov, R. (2008). Application of machine learning techniques for supply chain demand forecasting. European Journal of Operational Research, 184(3), 1140–1154. https://doi.org/10.1016/j.ejor.2006.12.004

---

## **1. Business Understanding**

### **1.1 Problem Statements**

Berdasarkan kondisi yang telah diuraikan sebelumnya, perusahaan akan mengembangkan sebuah sistem prediksi penjualan produk suplemen kesehatan untuk menjawab permasalahan berikut:

- Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh terhadap jumlah penjualan produk suplemen kesehatan (units sold)?

- Berapa jumlah produk suplemen yang akan terjual pada bulan tertentu dengan karakteristik atau fitur tertentu seperti harga, diskon, lokasi, dan platform penjualan?

### **1.2 Goals**

Untuk menjawab pertanyaan tersebut, Anda akan membuat predictive modelling dengan tujuan atau goals sebagai berikut:

- Mengetahui fitur-fitur yang paling berkorelasi dan berpengaruh terhadap volume penjualan produk (units sold).

- Membangun model machine learning yang dapat memprediksi jumlah unit produk yang akan terjual secara akurat berdasarkan data historis dan fitur yang tersedia.

### **1.3 Solution Statements**

Untuk mencapai tujuan tersebut, beberapa solusi yang akan dilakukan:

- Membangun model dengan algortima XGBoost yang akan dievaluasi dengan metrik RMSE dan MAPE.

- Melakukan hyperparameter tuning untuk menemukan model terbaik sebagai solusi.

---

## **2. Data Understanding**
Dataset ini memuat data penjualan mingguan dari berbagai suplemen kesehatan mulai dari Januari 2020 hingga April 2025. Dataset ini memuat produk dalam kategori seperti Protein, Vitamin, Omega dan Asam Amino serta lainnya yang dijual pada beberapa platform e-commerce seperti Amazon, Walmart dan iHerb. Dataset ini juga melihat penjualan di 3 negara yakni USA, UK dan Canada. Sumber dataset yang digunakan adalah **Supplement Sales Data** dari repositori Kaggle dengan link berikut https://www.kaggle.com/datasets/zahidmughal2343/supplement-sales-data

### **2.1 Variabel-variabel pada Supplement Salses Data dataset adalah sebagai berikut:**
- Date : tanggal penjualan mingguan (setiap hari Senin) dari Januari 2020 - April 2025.
- Product Name : Nama produk suplemen yang dijual (contoh: Whey Protein, Vitamin C, dll).
- Category : Kategori produk yang dijual (contoh:Protein, Vitamin, Omega, dll).
- Units Sold : Jumlah unit produk yang terjual pada minggu tersebut.
- Price : Harga jual produk per unit.
- Revenue : Total pendapatan yang dihasilkan (Units Sold * Price).
- Discount : Diskon yang diberikan (dalam persentase dari harga asli).
- Units Returned : Jumlah unit yang dikembalikan pada minggu tersebut.
- Location : Lokasi penjualan.
- Platform : Platform e-commerce tempat penjualan berlangsung.

### **2.2 Load Dataset**
Untuk mengetahui secara umum dataset menggunakan code sebagai berikut.
```
dataset.info()
```
Pada dataset yang digunakan terdiri dari 10 fitur dengan 4384 baris data yang terbagi menjadi fitur dengan tipe data numerik dan kategorik. Salah satu fitur yakni Date bertipe data object sehingga perlu diubah menjadi format datetime. Selain itu, pada dataset tidak ditemukan missing value dan nilai duplikat.

### **2.3 Check Unique Value**

Pada tahap ini, fitur-fitur kategorik diperiksa berdasarkan nilai unik yang dimilikinya. Nilai unik merupakan nilai yang merepresentasikan satu kategori data tertentu. Untuk itu, penulis menggunakan sebuah fungsi yang menampilkan jumlah serta jenis nilai unik yang terdapat pada masing-masing fitur kategorik. Adapun fitur kategorik yang diperiksa yakni Product Name, Category, Location dan Platform.
```
def unique_categorical(dataset, column_name):
    try:
        print(f"Number of unique {column_name}: {dataset[column_name].nunique()}")
        print(f"Unique {column_name}:")
        for value in dataset[column_name].unique():
            print(f"- {value}")
    except KeyError:
        print(f"Error: Column '{column_name}' not found in the dataset.")
```
Pada fitur Product Name memiliki 16 nilai unik diantaranya Whey Protein, Electrolyte Powder, Iron Supplement, Green Tea Extract, Biotin, Ashwagandha, Magnesium, Collagen Peptides, Melatonin, Creatine, BCAA, Pre-Workout, Multivitamin, Fish Oil, Vitamin C dan Zinc.

Pada fitur Category memiliki 10 nilai unik diantaranya Protein, Hydration, Mineral, Fat Burner, Vitamin, Herbal, Sleep Aid, Performance, Amino Acid dan Omega.

Pada fitur Location memiliki 3 nilai unik yaitu Canada, USA dan UK. Dan terakhir fitur Platform memiliki 3 nilai unik yaitu Walmart, Amazon dan iHerb.

### **2.3 Exploratory Data Analysis (EDA)**

#### **2.3.1 Descriptive Statistics**

Untuk mendapatkan ringaksan statistik deskriptif dari dataset yang digunakan, dapat menggunakan kode sebagai berikut.
```
dataset.describe()
```
Dimana hasilnya ditampilkan pada tabel berikut.
|           | Date                | Units Sold | Price     | Revenue   | Discount | Units Returned  |
|-----------|---------------------|------------|-----------|-----------|----------|-----------------|
| Count     | 4384                | 4384.000   | 4384.000  | 4384.000  | 4384.000 | 4384.000        |
| Mean      | 2022-08-18 12:00:00 | 150.200    | 34.781    | 5226.569  | 0.124    | 1.531           |
| Min       | 2020-01-06 00:00:00 | 103.000    | 10.000    | 1284.000  | 0.000    | 0.000           |
| 25%       | 2021-04-26 00:00:00 | 142.000    | 22.598    | 3349.373  | 0.060    | 1.000           |
| 50%       | 2022-08-18 12:00:00 | 150.000    | 34.720    | 5173.140  | 0.120    | 1.000           |
| 75%       | 2023-12-11 00:00:00 | 158.000    | 46.713    | 7009.960  | 0.190    | 2.000           |
| Max       | 2025-03-31 00:00:00 | 194.000    | 59.970    | 10761.850 | 0.250    | 8.000           |
| Std       | -                   | 12.396     | 14.198    | 2192.492  | 0.072    | 1.258           |

Dari statistik deskriptif yang ditampilkan, secara umum nilai pada Units Sold terdistribusi cukup normal. Berbeda dengan fitur Price dan Discount yang nilainya terdistribusi merata. Untuk Revenue dan Units Returned memiliki distribusi yang mengerah ke right-skewed namun tidak ekstrim.

#### **2.3.2 Univariat Analysis**

Pada bagian ini menampilkan time series dari Units Sold untuk melihat trend transaksinya berdasarkan mingguan. Selain itu, pada bagian ini juga menampilkan distribusi dari fitur numerik dan kategorik pada dataset.

##### **2.3.2.1 Time Series**

![Gambar dari GitHub](https://raw.githubusercontent.com/lukyg/markdown-image/main/time_series_1.png)

Berdasarkan time series Units Sold mingguan dari tahun 2020 hingga 2025, terlihat bahwa tren penjualan bersifat fluktuatif tanpa pola musiman yang konsisten. Hal ini mengindikasikan bahwa permintaan suplemen terjadi secara kontinu sepanjang tahun dan tidak terlalu dipengaruhi oleh musim atau periode tertentu.

##### **2.3.2.2 Numerical Features**

![Gambar dari GitHub](https://raw.githubusercontent.com/lukyg/markdown-image/main/numerik_dist_1.png)

Dari grafik distribusi tersebut secara umum nilai pada Units Sold terdistribusi cukup normal. Berbeda pada Price dan Discount yang nilainya distribusi merata. Untuk Revenue dan Units Returned memiliki distribusi yang mengarah ke right-skew namun tidak ekstrim.

##### **2.3.2.3 Categorical Features**

![Gambar dari GitHub](https://raw.githubusercontent.com/lukyg/markdown-image/main/cat_dist_1.png)

Secara keseluruhan setiap produk pada dataset ini memiliki jumlah yang seimbang dari setiap 16 produk yang ada dengan jumlah 274.

![Gambar dari GitHub](https://raw.githubusercontent.com/lukyg/markdown-image/main/cat_dist_2.png)

Jika dilihat distribusi berdasarkan kategori produk, Vitamin dan Mineral merupakan produk terbanyak dengan jumlah 822.

![Gambar dari GitHub](https://raw.githubusercontent.com/lukyg/markdown-image/main/cat_dist_3.png)

Dari ketiga lokasi yang ada, Canada merupakan negara dengan jumlah record terbanyak dengan 1507 data, kemudian disusul dengan UK dengan 1475 data dan USA dengan 1402 data.

![Gambar dari GitHub](https://raw.githubusercontent.com/lukyg/markdown-image/main/cat_dist_4.png)

Dari ketiga platform yang ada, iHerb merupakan platform dengan jumlah record terbanyak dengan 1499 data, kemudian disusul dengan Amazon dengan 1473 data dan Walmart dengan 1412 data.

#### **2.3.3 Bivariat Analysis**

##### **2.3.3.1 Numerical Features**

![Gambar dari GitHub](https://raw.githubusercontent.com/lukyg/markdown-image/main/bi_num_dist_1.png)

Dari scatterplot Units Sold terhadap Revenue, dapat dilihat bahwa trennya tidak dapat ditentukan dengan pasti, karena hasil ini dipengaruhi oleh faktor lain seperti diskon, produk yang diretur, dll.

##### **2.3.3.2 Categorical Features**

![Gambar dari GitHub](https://raw.githubusercontent.com/lukyg/markdown-image/main/bi_cat_dist_1.png)

Karena sebelumnya dari jumlah produk distribusinya merata, jika produk terhadap platform penjualan produk dapat dilihat bahwa beberapa produk lebih banyak terjual di platform tertentu. Misalnya Whey Protein, Iron Suplement, Biotin, Magnesium dan Melatonin yang banyak terjual di iHerb.

![Gambar dari GitHub](https://raw.githubusercontent.com/lukyg/markdown-image/main/bi_cat_dist_2.png)

Tren serupa juga antara category terhadap platform dimana kategori tertentu unggul pada platform tertentu juga. Misalnya pada Vitamin, Mineral dan Sleep Aid yang banyak dibeli pada platform iHerb.

![Gambar dari GitHub](https://raw.githubusercontent.com/lukyg/markdown-image/main/bi_cat_dist_3.png)

Pada grafik hubungan platform terhadap location menunjukkan bahwa iHerb dan Amazon pasar tertinggi di Canada, meskipun selisihnya tidak signifikan.

#### **2.3.4 Multivariat Analysis**

##### **2.3.4.1 Numerical Features**

![Gambar dari GitHub](https://raw.githubusercontent.com/lukyg/markdown-image/main/multi_num_1.png)

Dari persebasaran data antara Units Sold dengan Revenue terhadap Location, menunjukkan bahwa pola yang terbentuk tidak menentu. Namun dapat diamati bahwa data bertumpuk pada rentang nilai 125 hingga 165. Sedangkan terdapat beberapa titik data yang di luar dari tren. Asumsi dapat menjadi anomali, namun hasil ini dipengaruhi oleh fitur lain seperti diskon, dll.

![Gambar dari GitHub](https://raw.githubusercontent.com/lukyg/markdown-image/main/heatmap_1.png)

Dari heatmap tersebut menunjukkan bahwa fitur target yakni Units Sold memiliki korelasi yang rendah, baik terhadap Units Returned, Price, Revenue dan Diskon. Namun, fitur-fitur ini tetap digunakan pada proses pelatihan model mengingat fitur yang cukup terbatas.

---

## **3. Data Preparation**
Pada tahap ini, dataset perlu melalui beberapa proses preprocessing untuk memastikan data yang digunakan dalam pemodelan memiliki fitur yang kaya, bermakna, mudah dipahami oleh model, serta dapat meminimalkan bias. Beberapa teknik preprocessing yang digunakan meliputi:
- Label Encoding
- Feature Extraction
- Feature Engineering
- Split Dataset

### **3.1 Label Encoding**

```
dataset_pre['Product Name'] = le_product.fit_transform(dataset_pre['Product Name'])
dataset_pre['Category'] = le_category.fit_transform(dataset_pre['Category'])
dataset_pre['Location'] = le_location.fit_transform(dataset_pre['Location'])
dataset_pre['Platform'] = le_platform.fit_transform(dataset_pre['Platform'])
```

Pada dataset dengan fitur kategorikal, dilakukan proses Label Encoding untuk mengubah data kategorikal menjadi format numerik agar dapat digunakan dalam proses pelatihan model machine learning. Proses ini penting terutama untuk algoritma yang tidak dapat bekerja langsung dengan data dalam bentuk string.

### **3.2 Feature Extraction**

```
dataset_pre['year'] = dataset_pre['Date'].dt.year
dataset_pre['month'] = dataset_pre['Date'].dt.month
dataset_pre['Week'] = dataset_pre['Date'].dt.isocalendar().week
```

Fitur ektraksi dilakukan pada kolom Date yang terbagi menjadi tahun, bulan dan minggu untuk membantu dalam analisis musiman dan memudahkan model untuk mempelajari pola berdasarkan waktu.

Tujuan dari proses ini adalah untuk menyediakan informasi waktu dalam bentuk numerik, sehingga:

   - Dapat membantu analisis musiman seperti tren penjualan per bulan atau minggu,

   - Memudahkan model dalam mempelajari pola temporal, misalnya apakah penjualan cenderung naik saat akhir tahun atau turun di awal tahun.

### **3.3 Feature Engineering**

```
# Create lag features for Units Sold (1, 2, 4 weeks)
dataset_pre = dataset_pre.sort_values(['Product Name', 'Date'])
dataset_pre['Lag 1'] = dataset_pre.groupby('Product Name')['Units Sold'].shift(1)
dataset_pre['Lag 2'] = dataset_pre.groupby('Product Name')['Units Sold'].shift(2)
dataset_pre['Lag 4'] = dataset_pre.groupby('Product Name')['Units Sold'].shift(4)

# price after disc feature
dataset_pre['Price Discount'] =dataset_pre['Price'] * dataset_pre['Discount']
```
```
# Create rolling statistics (4-week window)
dataset_pre['Rolling Mean 4'] = dataset_pre.groupby('Product Name')['Units Sold'].shift(1).rolling(window=4).mean()
dataset_pre['Rolling Std 4'] = dataset_pre.groupby('Product Name')['Units Sold'].shift(1).rolling(window=4).std()

dataset_pre = dataset_pre.dropna()
```

Pada tahap ini, ada beberapa proses yang dilakukan untuk menambah fitur baru pada dataset dengan tujuan untuk memperkaya fitur dan membantu dalam menafsirkan pola pada dataset. Adapun uraiannya sebagai berikut.

- Lag features (Lag 1, Lag 2, Lag 4): Mengambil nilai Units Sold dari 1, 2, dan 4 periode sebelumnya untuk masing-masing Product Name, yang dapat membantu model mempelajari pola time series atau tren penjualan historis.

- Price Discount: Fitur baru yang merepresentasikan besaran diskon aktual (Price × Discount) sehingga bisa menganalisis pengaruh diskon terhadap penjualan.

- Rolling Mean dan Std (4 periode): Menghitung rata-rata bergerak (Rolling Mean) dan standar deviasi bergerak (Rolling Std) dari Units Sold selama 4 periode sebelumnya, yang memberikan informasi kontekstual mengenai fluktuasi dan stabilitas penjualan tiap produk dari waktu ke waktu.

- Setelah proses feature extraction dan feature engineering dataset menjadi 19 kolom.

Seluruh proses ini dilakukan dengan tujuan untuk memastikan model memiliki cukup informasi historis, statistik, dan kontekstual dalam mempelajari data penjualan agar model memiliki fitur yang cukup dan dapat melakukan prediksi yang akurat.

### **3.4 Split Dataset**

```
train_data = dataset_clean[dataset_clean['year'] <= 2023]
test_data = dataset_clean[dataset_clean['year'] >= 2024]
```
```
features = ['year',
            'month',
            'Week',
            'Product Name',
            'Category',
            'Price',
            'Discount',
            'Price Discount',
            'Units Returned',
            'Location',
            'Platform',
            'Lag 1',
            'Lag 2',
            'Lag 4',
            'Rolling Mean 4',
            'Rolling Std 4'
]
target = 'Units Sold'

X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]
```

Untuk pemodelan forcasting, sangat penting untuk tidak mencampur data masa depan dalam pelatihan model. Data train yang digunakan yakni dengan rentang histori tahun <= 2023, sedangkan untuk data test menggunakan histori tahun >= 2024, dimana Units Sold sebagai variabel target.

Dengan pemisahan berbasis waktu ini, model diharapkan dapat menghasilkan prediksi yang lebih general dan dapat diandalkan dalam menghadapi data penjualan di masa depan.

---

## **4. Modeling**

XGBoost (Extreme Gradient Boosting) adalah algoritma machine learning berbasis decision tree yang menggunakan teknik gradient boosting untuk meningkatkan akurasi model. XGBoost dirancang agar cepat, efisien, dan banyak digunakan dalam kompetisi serta industri. Dimana cara kerjanya sebagai berikut.

- Memulai dari model sederhana.
- Mengukur kesalahan (loss).
- Membuat model baru untuk memperbaiki kesalahan sebelumnya.
- Menggabungkan semua model menjadi prediksi akhir.
- Proses diulang sampai error minimal atau mencapai batas iterasi.

Adapun kelebihan yang dimiliki oleh model XGBoost sebagai berikut.

- **Performa tinggi:** Cepat dan akurat, sangat cocok untuk data tabular.
- **Regularisasi built-in:** Mencegah overfitting dengan L1 dan L2 regularization.
- **Menangani missing value:** Otomatis mengelola data yang hilang.
- **Support untuk berbagai tipe masalah:** Bisa digunakan untuk klasifikasi, regresi, dan ranking.
- **Parallel processing:** Memanfaatkan CPU multi-core untuk pelatihan lebih cepat.
- **Feature importance:** Memberikan informasi fitur yang paling berpengaruh.
- **Banyak opsi tuning:** Hyperparameter yang fleksibel untuk optimasi performa.

Meskipun memiliki banyak kelebihan, model ini juga memiliki kekurangan sebagai berikut.

- **Kompleksitas tinggi:** Banyak parameter yang harus dipahami dan diatur dengan baik.
- **Training time bisa lama:** Terutama pada dataset besar dengan fitur banyak.
- **Kurang cocok untuk data sekuensial:** Perlu preprocessing tambahan untuk data teks, gambar, atau time-series.
- **Interpretabilitas terbatas:** Sulit dijelaskan dibanding model linear sederhana.

Pada proyek ini, model XGBoost akan dilatih dengan 2 skema yakni, pertama, model XGBoost dilatih tanpa melakukan pengaturan pada parameter apapun (base model), kedua, model dilatih dengan melakukan hypertuning parameter. Parameter terbaik didapatkan dengan memanfaatkan metode GridSearchCV yang melatih model dalam beberapa iterasi sesuai dengan jumlah konfigurasi yang digunakan.

### **4.1 XGBoost Base Model**

```
models = []

# XGBoost
models.append(
    (
        'XGBoost',
        XGBRegressor()
    )
)
```

Untuk tahap awal, dilakukan pelatihan model menggunakan algoritma XGBoost Regressor tanpa penyetelan parameter (default hyperparameters). Model ini digunakan sebagai baseline untuk membandingkan performa dengan model yang dituning.

### **4.2 XGBoost Tuned Model**

```
xgbr = XGBRegressor(random_state=42)

# Define hyperparameters
params = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
}

# perform GridSearchCV with 5-fold cross-validation
xgbr_cv = GridSearchCV(xgbr, params, cv=5, scoring='r2', n_jobs=-1)
xgbr_cv.fit(X_train, y_train)

# Make predictions on training and test sets
y_xgbr_pred_train = xgbr_cv.predict(X_train)
y_xgbr_pred_test = xgbr_cv.predict(X_test)

# best parameters
print("Best parameters:", xgbr_cv.best_params_)
```
```
Best parameters: {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 100}
```

Setelah baseline model, dilakukan proses penyetelan (tuning) menggunakan GridSearchCV dengan 5-fold cross-validation. Dengan parameter yang dituning meliputi:
  1. max_depth: [3, 5, 7] → mengontrol kompleksitas pohon.
  2. learning_rate: [0.01, 0.1, 0.2] → menentukan seberapa besar kontribusi setiap pohon.
  3. n_estimators: [100, 200, 300] → jumlah pohon yang digunakan.

Model XGBoost yang dilatih memiliki parameter terbaik dengan learning rate 0.01, max depth 3 dan n estimators 100.

## **5. Evaluation**

Dalam proyek ini, digunakan **dua metrik evaluasi utama** untuk mengukur kinerja model regresi dalam memprediksi *units sold* Suplement Sales:

#### **1. RMSE (Root Mean Squared Error)**
- **Formula**:
![RMSE](https://latex.codecogs.com/svg.image?RMSE%20%3D%20%5Csqrt%7B%5Cfrac%7B1%7D%7Bn%7D%20%5Csum_%7Bi%3D1%7D%5En%20%28y_i%20-%20%5Chat%7By%7D_i%29%5E2%7D)
- **Penjelasan**:  
  RMSE mengukur seberapa jauh rata-rata prediksi model dari nilai aktual dalam **satuan asli** (misalnya, unit penjualan). Nilai RMSE yang **lebih kecil** menunjukkan performa model yang lebih baik. RMSE **sensitif terhadap outlier** karena menggunakan kuadrat dari selisih prediksi dan nilai aktual.

#### **2. MAPE (Mean Absolute Percentage Error)**
- **Formula**:
![MAPE](https://latex.codecogs.com/svg.image?MAPE%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20%5Csum_%7Bi%3D1%7D%5En%20%5Cleft%7C%20%5Cfrac%7By_i%20-%20%5Chat%7By%7D_i%7D%7By_i%7D%20%5Cright%7C)

- **Penjelasan**:  
  MAPE menunjukkan rata-rata kesalahan prediksi dalam bentuk **persentase terhadap nilai aktual**.  
  Metrik ini **mudah dipahami oleh bisnis** karena disajikan dalam bentuk persentase.  
  Nilai MAPE yang **lebih kecil** menunjukkan prediksi yang lebih akurat.  
  Namun, MAPE bisa **bermasalah jika terdapat nilai aktual mendekati nol**.

### **5.1 Tabel Evaluasi**

Adapun hasil train dan test dari model XGBoost base model maupun tuning model dengan GridSearchCV dilampirkan pada tabel berikut.

| Algorithm        | Train RMSE | Train MAPE | Test RMSE | Test MAPE |
|------------------|------------|------------|-----------|-----------|
| XGBoost          | 3.699408   | 0.018083   | 13.659235 | 0.071931  |
| Tuned XGBoost    | 12.213225  | 0.065535   | 12.254086 | 0.064967  |

Berdasarkan tabel tersebut, diuraikan insight sebagai berikut.
- Baseline XGBoost memiliki train RMSE dan MAPE yang sangat rendah, menunjukkan kemungkinan overfitting: model sangat bagus pada data pelatihan namun performa menurun pada data uji (test RMSE jauh lebih tinggi dari train RMSE).

- Tuned XGBoost menggunakan GridSearchCV:

    1. Memiliki train RMSE dan MAPE yang lebih tinggi dari baseline → artinya model menjadi lebih general (tidak overfit).

    2. Test RMSE dan Test MAPE lebih rendah dari baseline, menunjukkan bahwa model ini lebih stabil dan lebih baik dalam generalisasi ke data baru.

- Sehingga model Tuned XGBoost dapat digunakan untuk memprediksi *Units Sold* dan membantu dalam pengambilan keputusan stok suplemen. Berdasarkan hasil evaluasi, model ini memiliki rata-rata kesalahan prediksi sebesar **12,21 unit (RMSE)** dan kesalahan relatif rata-rata sebesar **6,5% (MAPE)**. Artinya, secara umum, model dapat memprediksi jumlah penjualan dengan deviasi sekitar 12 unit dari nilai aktual. Sebagai contoh, jika jumlah penjualan aktual adalah **150 unit**, maka prediksi model kemungkinan berada di sekitar angka **138 hingga 162 unit**. Namun, ini hanyalah gambaran umum karena **RMSE bukan batas pasti**, melainkan rata-rata dari semua kesalahan prediksi.

### **5.2 Pengujian Model**

Pengujian dilakukan dengan membandingkan hasil prediksi model terhadap nilai aktual menggunakan 10 sampel data pertama dari dataset uji. Hasil prediksi menunjukkan bahwa model mampu menghasilkan nilai yang cukup mendekati nilai aktual, sebagaimana terlihat pada tabel berikut:

| Index | Actual | Predicted   |
|-------|--------|-------------|
| 3338  | 172    | 152.888855  |
| 3354  | 137    | 150.600769  |
| 3370  | 146    | 149.770157  |
| 3386  | 154    | 150.654465  |
| 3402  | 153    | 150.495361  |
| 3418  | 152    | 149.225464  |
| 3434  | 140    | 149.824783  |
| 3450  | 160    | 149.770157  |
| 3466  | 140    | 151.114182  |
| 3482  | 145    | 149.770157  |

### **5.3 Time Series Prediction vs Aktual pada Bulan Tertentu**

![Gambar dari GitHub](https://raw.githubusercontent.com/lukyg/markdown-image/main/trend_actual_pred.png)

Dari grafik timeseries rerata aktual dengan hasil prediksi menunjukkan bahwa tren yang cukup berbeda. Dimana untuk data aktual grafik memiliki tren fluktuatif, berbeda dengan hasil prediksi yang cenderung stabil. Hasil ini menunjukkan model memberikan prediksi yang konservatif sehingga dalam implementasinya dapat mengantisipasi lonjakan permintaan dan menghindari kelebihan stok.

### **5.5 Feature Important**

![Gambar dari GitHub](https://raw.githubusercontent.com/lukyg/markdown-image/main/feature_importan.png)

Berdasarkan analisis feature importance dari model XGBoost yang telah dibangun, terdapat lima fitur utama yang paling berpengaruh terhadap prediksi Units Sold, yaitu: Units Returned, Rolling Mean 4, Lag 2, Price, dan Price Discount. Hal ini menunjukkan bahwa faktor historis penjualan (melalui fitur lag dan rolling), harga jual, serta jumlah unit yang dikembalikan pelanggan memiliki peran signifikan dalam menentukan jumlah penjualan mingguan produk suplemen. Temuan ini sejalan dengan logika bisnis, di mana performa historis dan strategi harga sangat memengaruhi permintaan pasar.

### **5.5 Dampak Terhadap Business Understanding**

Model yang dibangun menggunakan algoritma Tuned XGBoost menunjukkan performa prediksi yang cukup baik dengan nilai MAPE sebesar 6,5% dan RMSE sebesar 12,25 unit. Hal ini berarti, secara rata-rata, model mampu memprediksi jumlah penjualan mingguan dengan deviasi sekitar 12 unit dari nilai aktual.

Dalam konteks bisnis, performa model ini dapat memberikan dampak positif yang signifikan, khususnya dalam pengambilan keputusan terkait pengelolaan rantai pasok dan inventori:

- Mencegah kelebihan stok (overstock): Dengan prediksi yang lebih akurat, perusahaan dapat menghindari pembelian atau produksi berlebih yang dapat meningkatkan biaya penyimpanan dan risiko kerusakan/kedaluwarsa produk.

- Menghindari kekurangan stok (stockout): Prediksi yang handal dapat memastikan ketersediaan produk yang cukup di gudang dan marketplace, sehingga mencegah hilangnya potensi penjualan dan menurunnya kepuasan pelanggan.

- Optimalisasi alokasi distribusi: Berdasarkan prediksi yang memperhitungkan lokasi dan platform penjualan, perusahaan dapat mengalokasikan produk secara lebih efisien ke pasar yang lebih potensial.

- Efisiensi biaya logistik dan operasional: Dengan prediksi penjualan yang lebih akurat, pengadaan dan distribusi dapat direncanakan lebih tepat, sehingga meminimalkan biaya yang tidak perlu.

Dengan demikian, model ini berkontribusi langsung terhadap peningkatan efisiensi operasional dan pengambilan keputusan strategis berbasis data, sejalan dengan tujuan awal dari proyek ini.

### **5.5 Refleksi terhadap Problem Statement dan Goals**

Laporan ini secara umum telah menjawab seluruh problem statement dan memenuhi goals yang telah ditetapkan pada tahap Business Understanding:

- Problem Statement 1: "Fitur apa yang paling berpengaruh terhadap units sold?"
Berdasarkan analisis feature importance dari model XGBoost yang telah dibangun, terdapat lima fitur utama yang paling berpengaruh terhadap prediksi Units Sold, yaitu: Units Returned, Rolling Mean 4, Lag 2, Price, dan Price Discount.

- Problem Statement 2: "Berapa jumlah produk yang akan terjual berdasarkan karakteristik tertentu?"
Telah dijawab dengan pembangunan model prediktif yang mampu menghasilkan estimasi penjualan mingguan berdasarkan fitur seperti harga, diskon, waktu, lokasi, dan platform. Dimana pada time series rerata prediksi vs aktual dalam tren bulanan model melakukan prediksi yang konservatif dengan rerata nilai 150 units sold pada setiap bulannya.

- Goal: Membangun model prediksi akurat dan mengetahui fitur penting.
Hasil evaluasi menunjukkan bahwa model dengan tuning memiliki performa yang lebih general dan stabil dibanding model baseline, dengan MAPE rendah yang dapat diterima secara bisnis. Berdasarkan hasil evaluasi, model ini memiliki rata-rata kesalahan prediksi sebesar **12,21 unit (RMSE)** dan kesalahan relatif rata-rata sebesar **6,5% (MAPE)**. Artinya, secara umum, model dapat memprediksi jumlah penjualan dengan deviasi sekitar 12 unit dari nilai aktual. Sebagai contoh, jika jumlah penjualan aktual adalah **150 unit**, maka prediksi model kemungkinan berada di sekitar angka **138 hingga 162 unit**. Hasil ini cenderung stabil dan konservatif dalam menghadapi permintaan pasar yang fluktuatif.

Oleh karena itu, proyek ini tidak hanya menyelesaikan tantangan teknis pemodelan prediktif, namun juga berhasil memberikan wawasan dan solusi nyata yang berdampak terhadap proses bisnis perusahaan, terutama dalam pengambilan keputusan terkait manajemen inventori.


