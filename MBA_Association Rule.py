# -*- coding: utf-8 -*-
# **Study Case Association Rule**
## <left><h2><strong><font color="lightseagreen">Kelompok 2</font></strong></h2></left>
|Nama|NPM|Kontribusi|Tingkat Kontribusi
|---|---|---|---|
|Kayla Zahira Amadya|2206053890|Terlibat aktif diskusi, melakukan analisis MBA|100%|
|Najwa Putri Faradila|2206051355|Terlibat aktif diskusi, mencari data, melakukan preprocessing|100%|
|Fauzan Adzhima Alamsyah|1906376804|Terlibat aktif diskusi, membuat interpretasi analisis data MBA|100%|
|Adha Abdullah|2206053921|Terlibat aktif diskusi, membuat interpretasi bagian EDA|100%|

# **Informasi Data**
Data yang digunakan untuk study case ini berasal dari https://www.kaggle.com/datasets/timchant/supstore-dataset-2019-2022/data. Dataset terdiri dari catatan penjualan komprehensif dari sebuah superstore, yang berisi 9.994 entri di 19 kolom berbeda. Ini mencakup berbagai kategori seperti detail pesanan, informasi pelanggan anonim, spesifikasi produk, dan metrik keuangan. Keterangan untuk setiap kolomnya adalah sebagai berikut

1.   Order ID : Nomor unik yang mengidentifikasi setiap pesanan atau transaksi yang dilakukan.
2.   Order Date : Tanggal ketika pesanan dilakukan oleh pelanggan.
3. Ship Date : Tanggal ketika pesanan dikirim kepada pelanggan.
4. Customer: Nama atau ID unik yang mewakili pelanggan yang melakukan pesanan.
5. Segment: Segmentasi pelanggan berdasarkan kategori tertentu
6. Manufactory: Nama atau ID dari produsen atau pabrikan barang yang dipesan.
7. Product Name: Nama produk yang dipesan.
8. Category: Kategori umum dari produk yang dipesan.
9. Subcategory: Subkategori yang lebih spesifik di bawah kategori umum.
10. Region: Wilayah geografis tempat pelanggan atau toko berada.
11. City: Kota tempat pelanggan atau toko yang menerima pengiriman berada.
12. State: Negara bagian tempat pesanan dikirim atau pelanggan berada.
13. Country: Negara tempat pesanan dikirim.
14. ZIP: Kode pos untuk pelanggan atau lokasi pengiriman.
15. Sales: Total nilai penjualan untuk pesanan tersebut.
16. Quantity: Jumlah unit produk yang dipesan dalam pesanan tersebut.
17. Discount: Persentase atau jumlah diskon yang diterapkan pada pesanan.
18. Profit: Keuntungan yang dihasilkan dari penjualan pesanan tersebut.
19. Profit Margin: Persentase keuntungan bersih yang dihasilkan dari penjualan, dihitung dari rasio antara keuntungan (profit) dan penjualan (sales).

# **Preprocessing**
"""

try:
    import google.colab as gc_
    print("Running the code in Google Colab.", gc_)
    print("Installing required Module, please wait ... ")
    !pip install mlxtend
    !pip install pycaret
except:
    print("Running the code locally, make sure to run `pip install mlxtend, pycaret` in terminal")

# Commented out IPython magic to ensure Python compatibility.
# Mengimpor semua library yang dibutuhkan dalam eksplorasi data serta aturan asosiasi
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from itertools import combinations
from collections import Counter
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

import warnings
warnings.filterwarnings('ignore')

# %matplotlib inline
plt.style.use('bmh'); sns.set()

"""## Import Data

"""

# Membaca dataset
data = pd.read_csv('/content/superstore_dataset.csv')
data

"""Terlihat bahwa data berukuran 9994 baris × 19 kolom"""

data.info()

"""Data berisi 19 kolom dengan entry no-null dan terdiri atas 3 jenis kategori : objek, float dan integeer"""

# Menampilkan informasi umum dan statistik tentang dataset untuk mengecek ketidakkonsistenan data atau masalah lainnya
data.describe()

"""*   Kode pos dalam kolom `zip` idealnya harus dikelompokkan sebagai data kategoris untuk identifikasi geografis dan bukan sebagai angka biasa. Ini memungkinkan segmentasi pasar yang lebih tepat dan analisis data yang lebih mendalam.
*   Kolom seperti `customer`, `manufacturer`, `product_name`, dan `city` menampilkan beragam nilai unik, mengindikasikan kekayaan data yang cocok untuk analisis yang komprehensif.
*   Kolom `order_date` dan `ship_date` yang telah diformat sebagai datetime mencakup periode dari 1 Januari 2019 hingga 30 Desember 2022. Hal ini memfasilitasi analisis tren yang berkembang selama empat tahun, memungkinkan penilaian atas pola musiman atau tahunan.
*   Terdapat beberapa nilai margin keuntungan yang sangat rendah hingga negatif, hingga -2.75, menandakan potensi kerugian dalam beberapa transaksi.
*   Kategori dan subkategori data tampak konsisten, terdiri dari tiga kategori utama dan 17 subkategori, memudahkan pemetaan dan klasifikasi data.
*   Kolom diskon dan keuntungan menunjukkan variasi yang sangat lebar, dengan diskon maksimal hingga 80% dan keuntungan yang bisa mencapai hingga $8,399.976, menunjukkan adanya kemungkinan kesalahan pencatatan atau kebijakan diskon yang agresif
"""

# Mengubah data 'zip' menjadi data kategorik
data['zip'] = data['zip'].astype('category')

# Mengecek missing value untuk dataset
missing_data = data.isnull().sum()

# Mengecek entry yang terduplikasi
duplicate_data = data.duplicated().sum()

# mengecek semua ketidakkonsistenan seperti penjualan atau profit yang negatif
negative_sales = data[data['sales'] <= 0]
negative_profit = data[data['profit'] <= 0]

{
    "Missing Values": missing_data,
    "Duplicate Entries": duplicate_data,
    "Negative Sales": negative_sales.shape[0],
    "Negative Profit": negative_profit.shape[0]
}

"""Semua entry pada dataset tidak memiliki missing value. Terdapat 1 duplicate entries pada dataset. Sedangkan penjualan yang minus berjumlah 0 dan profit penjualan yang minus berjumlah 1936"""

# cek duplikat
duplicate_rows = data[data.duplicated(keep=False)]
duplicate_rows

# Menghilangkan data yang terduplikasi
data = data.drop_duplicates()

# Memverifikasi data terduplikasi dengan pengecekan ulang
duplicate_check = data.duplicated().sum()
duplicate_check

# Mengecek data dengan type 'order date' untuk mengonfirmasi waktu dan tanggal atau butuh convert
order_date_dtype = data['order_date'].dtype

# Jika 'order date' bukan waktu dan tanggal, convert data
if order_date_dtype != 'datetime64[ns]':
    data['order_date'] = pd.to_datetime(data['order_date'])
    added_datetime_info = True
else:
    added_datetime_info = False

# Print output untuk pengecekan convert
order_date_dtype, added_datetime_info

# Menambahkan kolom hari, bulan, dan tahun ke dataframe
data['day'] = data['order_date'].dt.day
data['month'] = data['order_date'].dt.month
data['year'] = data['order_date'].dt.year

# Mendefinisikan fungsi musim berdasarkan bulan
def get_season(month):
    if 3 <= month <= 5:
        return 'spring'
    elif 6 <= month <= 8:
        return 'summer'
    elif 9 <= month <= 11:
        return 'autumn'
    else:
        return 'winter'

# Mengaplikasikan fungsi musim ke kolom pada dataset
data['season'] = data['month'].apply(get_season)

# Ubah data_format untuk menggunakan 'subkategori' dan bukan 'nama_produk' untuk deskripsi item
formatted_data = data[['customer', 'order_date', 'subcategory', 'day', 'month', 'year', 'season']]
formatted_data.rename(columns={'customer': 'Member_name', 'order_date': 'Date', 'subcategory': 'itemDescription'}, inplace=True)

formatted_data

# Melihat frekuensi produk yang paling sering dibeli
item_counts = formatted_data['itemDescription'].value_counts()

plt.figure(figsize=(24,6))
item_counts.plot(kind='bar')
plt.title('Total Produk yang Dibeli')
plt.xticks(size=7)
plt.xlabel('Produk')
plt.ylabel('Jumlah Pembelian')
plt.show()

"""Terlihat ada plot yang ditampilkan bahwa produk paling banyak dibeli adalah Binder dengan jumlah >1400 dan produk paling jarang dibeli adalah mesin fotokopi dengan jumlah pembelian <100"""

# Melihat frekuensi 20 produk yang paling sering dibeli
plt.figure(figsize=(12,6))
item_counts.head(10).plot(kind='bar')

plt.title('10 Produk Teratas yang Paling Sering Dibeli')
plt.ylabel('Frekuensi')
plt.show()

"""Kita mempartisi 17 produk menjadi 10 produk dengan pembelian terbanyak dan dari 10 produk, pembelian paling sedikit ada pada produk label"""

# Melihat frekuensi produk yang paling sering dibeli
sells_per_season = formatted_data.groupby(['season', 'itemDescription']).size().reset_index().pivot(columns='season', index='itemDescription', values=0)

# Menentukan urutan musim yang diinginkan
season_order = ['spring', 'summer', 'autumn', 'winter']

# Menyusun ulang kolom dari dataframe
sells_per_season = sells_per_season.reindex(columns=season_order)

# Menyussun index dari jumlah tiap kolomnya
sells_per_season['total'] = sells_per_season.sum(axis=1)
sells_per_season = sells_per_season.sort_values(by='total', ascending=False)
sells_per_season = sells_per_season.drop(columns=['total']) # drop the total column, it's for sorting

# Membuat plot dengan ukuran yang diinginkan
fig = plt.figure(figsize=(30, 8))  # Mengatur ulang ukuran untuk mencegah overlap

# Membuat objek pada sumbu x
ax = fig.add_subplot(111) # 111 menciptakan satu sumbu x

# Plot data pada sumbu x
sells_per_season.plot(kind='bar', stacked=True, ax=ax) # Mengganti ukuran ke (10,6)

# Mengatur detail plot
plt.title('Total Produk yang Dibeli')
plt.xticks(size=7)
plt.xlabel('Produk')
plt.ylabel('Jumlah Pembelian')
plt.show()

"""Berikut adalah plot pembelian produk yang dibagi menjadi 4 musing sesuai urutan, yaitu 'musim semi', 'musim panas', 'musim gugur', dan 'musim dingin'"""

# Mengelompokan data sesuai musim dan mengurutkan 10 pembelian terbanyak tiap musimnya
top_items_per_season = formatted_data.groupby('season')['itemDescription'].value_counts().groupby('season').head(10)

# Membuat plot untuk 10 pembelian terbanyak tiap musim
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(24, 8))
seasons = ['spring', 'summer', 'autumn', 'winter']
for i, season in enumerate(seasons):
    row = i // 2
    col = i % 2
    top_items = top_items_per_season[season]
    top_items = top_items.sort_values()
    top_items.plot(kind='barh', ax=axes[row, col])
    axes[row, col].set_title(f'Top 20 Items in {season.capitalize()}')
    axes[row, col].set_xlabel('Items')
    axes[row, col].set_ylabel('Frequency')
    axes[row, col].tick_params(axis='x')  # Menyesuaikan ukuran label dan rotasi sesuai yang dibutuhkan

plt.tight_layout()
plt.show()

"""Berikut adalah pola temporal untuk tiap musimnya"""

# Menghitung jumlah transaksi per bulan
transactions_per_day = formatted_data.groupby(data['order_date']).size()

# Membuat diagram timeline jumlah transaksi
plt.figure(figsize=(30,6))
transactions_per_day.plot(kind='line', marker='o', color='b')
plt.title('Timeline Jumlah Transaksi Per Bulan')
plt.xlabel('Tanggal')
plt.ylabel('Jumlah Transaksi')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Menampilkan plot
plt.show()

"""Berikut adalah plot berjenis seasonal time series yang menampilkan timeline jumlah transaksi per bulan berdasarkan data tanggal yang diurutkan dari tahun 2019 hingga awal 2023. Terlihat pada plot bahwa jumlah transaksi cenderung meningkat dari waktu ke waktu, terutama setelah tahun 2021"""

plt.figure(figsize=(12,6))
data[data['day'].isin(range(0,31))].day.value_counts().sort_index().plot(kind='line', marker='o')
plt.title('Frekuensi Pembelian Berdasarkan Hari Tiap Bulannya')
plt.ylabel('Frekuensi')
plt.show()

"""Jika plot transaksi dipartisi menjadi plot pembelian berdasarkan hari tiap bulannya, maka berikut adalah plot yang dihasilkan. Terlihat bahwa kenaikan pembelian terbesar terjadi pada hari ke 16-17 dan penurunan pembelian terbesar terjadi pada hari ke 21-22 tiap bulannya"""

plt.figure(figsize=(12,6))
data['order_date'].dt.month.value_counts().sort_index().plot(kind='line', marker='o')
plt.title('Frekuensi Pembelian Berdasarkan Bulan')
plt.ylabel('Frekuensi')
plt.xlabel('Bulan')
plt.show()

"""Jika plot dipartisi berdasarkan frekuensi pembelian tiap bulannya, maka berikut adalah plot yang dihasilkan. Terlihat bahwa kenaikan frekuensi pembelian terbesar terjadi di bulan Agustus-September dan penurunan frekuensi pembelian terbesar terjadi di bulan September-Oktober"""

plt.figure(figsize=(12,6))
data['order_date'].dt.year.value_counts().sort_index().plot(kind='bar')
plt.title('Frekuensi Pembelian Berdasarkan Tahun')
plt.ylabel('Frekuensi')
plt.xlabel('Tahun')
plt.show()

"""Jika plot transaksi dipartisi berdasarkan tahun, maka terlihat dari tahun 2019 hingga 2023, jumlah total transaksi pembelian selalu mengalami tren kenaikan secara berkala

# **Market Basket Analysis** (Association rule using Apriori Algorithm)

## General Analysis
"""

formatted_data

# Mengubah format data agar sesuai untuk analisis - membuat tabel one-hot encoding
basket = formatted_data.groupby(['Member_name', 'itemDescription'])['itemDescription'].count().unstack().reset_index().fillna(0)

# Mengubah nilai menjadi 1 atau 0 (1 jika dibeli, 0 jika tidak)
basket = basket.set_index('Member_name')
basket = basket.applymap(lambda x: 1 if x > 0 else 0)

basket.head(10)

"""Tabel one-hot encoded di atas menggambarkan perilaku belanja setiap pelanggan (diwakili oleh Member_name) terhadap berbagai kategori produk (itemDescription). Contoh:
1. Aaron Bergman:
   - Membeli produk dari kategori berikut: Art, Bookcases, Chairs, Phones, dan Storage (ditunjukkan oleh angka 1 pada kolom-kolom tersebut).
   - Tidak membeli produk dari kategori seperti Accessories, Appliances, atau Binders (ditunjukkan oleh angka 0 pada kolom-kolom tersebut).
2. Aaron Hawkins:
   - Membeli produk dari banyak kategori: Accessories, Art, Binders, Chairs, Envelopes, Furnishings, Labels, Paper, Phones, dan Storage.
   - Tidak membeli produk dari kategori seperti Appliances, Fasteners, atau Copiers.
"""

frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)
frequent_itemsets.sort_values(by='support', ascending=False, na_position='last', inplace = True)
frequent_itemsets

"""**Keypoint dari output di atas**<br>
1. **Support**: Kolom ini mewakili frekuensi itemset yang terjadi dalam transaksi. Contoh:
   - Itemset {'Binders'} memiliki support sebesar 0.819672, yang artinya sekitar 81.967% transaksi terdiri dari "Binders".
   - Sama hal nya dengan {'Paper'} muncul di 77.04% transaksi, dan {'Furnishings'} muncul di 66.58% transaksi.
2. **Itemsets**: Ini adalah kombinasi item yang sering muncul bersamaan dalam transaksi. Contoh
   - {'Binders'} paling sering muncul pada transaksi tunggal.
   - Kombinasi dari {'Paper', 'Binders'} sering muncul pada 65.06% transaksi.




"""

# Menghitung rules asosiasi keseluruhan
rules = association_rules(frequent_itemsets, metric="lift")

# Melihat hasil
rules.head(10)

"""Dari hasil tabel di atas:

1. **Antecedents**: Item atau kumpulan item di sisi kiri aturan. Ini adalah barang-barang yang bila dibeli, dapat mengarah pada pembelian barang-barang di kolom consequents.

  - Misalnya, "Paper" adalah bagian Antecedents atau "jika" dari aturan tersebut.

2. **Consequents**: Item atau kumpulan item di sisi kanan aturan. Ini adalah barang-barang yang kemungkinan besar akan dibeli ketika barang-barang di pendahulunya dibeli.

  - Misalnya, "Binders" adalah konsekuensi atau bagian "kemudian" dari aturan tersebut.

3. **Antecedent support**: Proporsi transaksi yang mengandung item antecedent.

  - Misalnya support untuk (Paper) adalah 0.770492, artinya 77.04% transaksi mengandung "Paper".

4. **Consequents support**: Proporsi transaksi yang mengandung item consequent.

  - Misalnya support untuk (Binders) adalah 0.819672, artinya 81.97% transaksi mengandung "Binders".

5. **Support**: Proporsi transaksi yang mengandung anteseden dan konsekuen. Ini adalah peluang gabungan kedua barang dibeli bersamaan.

  - Misalnya, aturan (Paper) -> (Binder) memiliki dukungan 0,650694, artinya 65,07% transaksi berisi "Paper" dan "Binder".

6. **Confidence**: Kemungkinan konsekuen dibeli ketika anteseden dibeli.

  - Misalnya, confidence sebesar 0,844517 untuk aturan (Paper) -> (Binder) berarti bahwa 84,45% saat "Paper" dibeli, "Binder" juga dibeli.
<br>

$$ Confidence=
\frac{\text{Support of Antecedent and Consequent}}
{\text{Support of Antecedent}}
$$
<br>

7. **Lift**: Kekuatan aturan dibandingkan dengan kejadian bersamaan secara acak dari antecedent dan consequent, dapat dihitung sebagai berikut:

  - Nilai lift >1 menunjukkan hubungan positif (pembelian antecedent membuat pembelian consequent lebih mungkin terjadi).
  - Misalnya, aturan (Paper) -> (Binder) memiliki lift sebesar 1,030311, artinya membeli "Paper" meningkatkan kemungkinan juga membeli "Binder" sebesar 3,03%.
<br>

$$ Lift=
\frac{\text{Confidence}}
{\text{Consequent Support}}
$$
<br>

8. **Leverage**: Mengukur perbedaan antara frekuensi aturan yang diamati dan apa yang diharapkan jika anteseden dan konsekuennya independen. Nilai leverage yang positif berarti aturan tersebut lebih baik daripada kebetulan.

  - Misalnya, leverage sebesar 0,019143 untuk aturan (Paper) -> (Binder) berarti kedua item tersebut muncul bersamaan 1,91% lebih sering dibandingkan jika keduanya independen.

9. **Conviction**: Mengukur tingkat implikasi aturan, dengan nilai yang lebih tinggi menunjukkan implikasi yang lebih kuat.

  - Misalnya, conviction sebesar 1,159793 untuk (Paper) -> (Binder) menunjukkan hubungan yang relatif kuat.
<br>

$$ Conviction=
\frac{\text{1 - Consequent Support}}
{\text{1 - Confidence}}
$$
<br>

10. **zhangs_metric**: Metrik lain yang mengukur kekuatan asosiasi, khususnya berguna ketika terdapat ketidakseimbangan yang tinggi dalam dukungan antara anteseden dan konsekuen.



**Key Insights**:

 - Aturan Kuat: Aturan dengan confidence tinggi dan nilai lift lebih kuat. Misalnya:
    - Aturan (Paper) -> (Binders) memiliki confidence sebesar 0,844517 dan lift sebesar 1,030311, yang menunjukkan bahwa pelanggan yang membeli "Paper" memiliki kemungkinan 3% lebih besar untuk juga membeli "Binders".
    - Aturan (Storage) -> (Binders) memiliki confidence 0,842412 dan lift 1,027743, menunjukkan hubungan serupa antara "Storage" dan "Binders".
 - Aturan yang Lebih Lemah: Confidence atau lift yang lebih rendah mungkin mengindikasikan hubungan yang lebih lemah atau kejadian bersama yang lebih umum yang tidak menunjukkan adanya hubungan yang kuat.

  




"""

# Mengurutkan berdasarkan confidence
rules = rules.sort_values(by='confidence', ascending=False)

# Menampilkan beberapa rules teratas
rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10)

# Mengurutkan berdasarkan lift
rules = rules.sort_values(by='lift', ascending=False)

# Menampilkan beberapa rules teratas
rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10)

"""**Insight**:
<br>
Saat diurutkan berdasarkan confidence,

  - Aturan dengan confidence tinggi (seperti 1.0 dalam kasus ini) berarti setiap kali antecedent dibeli, consequent juga selalu dibeli dalam transaksi yang diamati.
  - Nilai lift lebih besar dari 1 (misalnya, 1.551859) menunjukkan hubungan positif yang kuat antara antecedent dan consequent, artinya pembelian antecedent membuat pembelian consequent lebih mungkin dibandingkan jika secara kebetulan.
<br>

Saat diurutkan berdasarkan lift,
  - Lift tinggi (seperti 8.88) menunjukkan hubungan yang sangat kuat antara antecedents dan consequents, artinya pembelian item di kolom antecedents sangat mungkin diikuti oleh pembelian item di kolom consequents.
  - Confidence yang lebih rendah, seperti 0.235294, menunjukkan bahwa meskipun lift tinggi, tidak semua pelanggan yang membeli antecedents juga membeli consequents. Hal ini berarti bahwa hubungan antar item tetap kuat, tetapi hanya untuk sebagian kecil dari transaksi.

## Season-Based Analysis

Untuk mempermudah membuat strategi penjualan, data akan dilihat berdasarkan empat musim.

### Spring
"""

# Helper Function untuk mempermudah analisis
def apriori_mba(formatted_data, min_support=0.01, use_colnames=True):
  # Mengubah format data agar sesuai untuk analisis - membuat tabel one-hot encoding
  basket = formatted_data.groupby(['Member_name', 'itemDescription'])['itemDescription'].count().unstack().reset_index().fillna(0)

  # Mengubah nilai menjadi 1 atau 0 (1 jika dibeli, 0 jika tidak)
  basket = basket.set_index('Member_name')
  basket = basket.applymap(lambda x: 1 if x > 0 else 0)

  return apriori(basket, min_support=min_support, use_colnames=use_colnames)

basket

frequent_itemsets_spring = apriori_mba(formatted_data[formatted_data['season'] == 'winter'], min_support=0.001, use_colnames=True)
frequent_itemsets_spring.sort_values(by='support', ascending=False, na_position='last', inplace = True)
frequent_itemsets_spring.head()

# Menghitung rules asosiasi
rules_spring = association_rules(frequent_itemsets_spring, metric="confidence", min_threshold=0.5)

# Melihat hasil
rules_spring.sort_values(by='lift', ascending=False).head()

rules_spring.sort_values(by='support', ascending=False).head()

"""### Summer"""

frequent_itemsets_summer = apriori_mba(formatted_data[formatted_data['season'] == 'summer'], min_support=0.001, use_colnames=True)
frequent_itemsets_summer.sort_values(by='support', ascending=False, na_position='last', inplace = True)
frequent_itemsets_summer

# Menghitung rules asosiasi untuk musim panas
rules_summer = association_rules(frequent_itemsets_summer, metric="confidence", min_threshold=0.5)

# Melihat hasil
rules_summer.sort_values(by='lift', ascending=False).head()

rules_summer.sort_values(by='support', ascending=False).head()

"""###Autumn"""

frequent_itemsets_autumn = apriori_mba(formatted_data[formatted_data['season'] == 'autumn'], min_support=0.001, use_colnames=True)
frequent_itemsets_autumn.sort_values(by='support', ascending=False, na_position='last', inplace = True)
frequent_itemsets_autumn.head()

# Menghitung rules asosiasi untuk musim gugur
rules_autumn = association_rules(frequent_itemsets_autumn, metric="confidence", min_threshold=0.5)

# Melihat hasil
rules_autumn.sort_values(by='lift', ascending=False).head()

rules_autumn.sort_values(by='support', ascending=False).head()

"""### Winter"""

frequent_itemsets_winter = apriori_mba(formatted_data[formatted_data['season'] == 'winter'], min_support=0.001, use_colnames=True)
frequent_itemsets_winter.sort_values(by='support', ascending=False, na_position='last', inplace = True)
frequent_itemsets_winter.head()

# Menghitung rules asosiasi untuk musim dingin
rules_winter = association_rules(frequent_itemsets_winter, metric="confidence", min_threshold=0.5)

# Melihat hasil
rules_winter.sort_values(by='lift', ascending=False).head()

rules_winter.sort_values(by='support', ascending=False).head()

"""### Key Insights and Strategic Recommendations

Berikut insights yang bisa diperoleh dari analisis data berdasarkan musim di atas.
<br>
1. **Spring**:
  - Antecedents seperti (Paper, Storage) atau (Binders, Storage) menunjukkan kombinasi produk yang sering dibeli bersama di musim semi.
  - Support tertinggi sebesar 0.061189 menunjukkan bahwa sekitar 6.1% dari transaksi mencakup kombinasi antecedents dan consequents tersebut, dengan confidence sekitar 52-57%.
  - Lift menunjukkan nilai yang cukup baik antara 1.37 hingga 1.51, yang berarti pembelian antecedent meningkatkan kemungkinan pembelian consequent sebesar 37% hingga 51%.

  Kesimpulan: Kombinasi produk seperti Paper dan Storage cenderung dikaitkan dengan pembelian Binders, dan Accessories dan Paper memiliki asosiasi kuat dengan Binders. Strategi cross-selling atau penempatan barang di musim semi bisa mempertimbangkan asosiasi ini.

2. **Summer**:
  - (Chairs -> Paper) dan (Appliances -> Binders) adalah aturan dengan support tertinggi pada musim panas, dengan support mencapai 10.15% untuk aturan Chairs -> Paper.
  - Confidence cukup tinggi, mencapai hingga 62.92% untuk aturan Appliances -> Binders.
  - Lift berkisar antara 1.28 hingga 1.48, menunjukkan peningkatan yang signifikan dalam kemungkinan pembelian consequent ketika antecedent dibeli.

  Kesimpulan: Musim panas menunjukkan korelasi kuat antara Chairs dan Paper, serta Appliances dengan Binders, yang bisa dipertimbangkan untuk strategi penjualan musiman.

3. **Autumn**:
  - Paper -> Binders memiliki support tertinggi, sekitar 26.17%, yang berarti aturan ini sangat umum terjadi di musim gugur.
  - Confidence mencapai 56.42% untuk aturan Paper -> Binders dan 56.52% untuk aturan Storage -> Binders walaupun memiliki support yang lebih rendah dibanding Paper -> Binders.
  - Lift lebih rendah dibandingkan musim lain, dengan rentang 1.05 hingga 1.18, menunjukkan asosiasi yang lemah namun tetap signifikan.

  Kesimpulan: Musim gugur menunjukkan aturan kuat untuk kategori produk inti seperti Paper dan Binders, yang sering dibeli bersama, menunjukkan potensi untuk promosi terkait produk.

4. **Winter**:
  - Paper, Storage -> Binders dan Binders, Storage -> Paper menunjukkan aturan dominan di musim dingin dengan support tertinggi sekitar 6.1%.
  - Confidence berkisar antara 52.2% hingga 57.37%, yang berarti lebih dari separuh waktu pembelian antecedents diikuti oleh consequents.
  - Lift bervariasi dari 1.37 hingga 1.51, yang menunjukkan hubungan yang signifikan.

  Kesimpulan: Musim dingin menunjukkan preferensi pelanggan terhadap produk seperti Binders dan Storage, serta kombinasi Accessories dan Paper. Ini bisa menjadi indikasi produk yang saling melengkapi untuk promosi bundling atau penempatan produk.


**Rekomendasi Strategis**:
  - Kombinasi Produk: Di setiap musim, aturan yang kuat muncul untuk kategori produk inti seperti Paper, Binders, Accessories, dan Storage. Produk-produk ini sering kali muncul bersama, dan ini bisa dimanfaatkan untuk kampanye promosi yang sesuai dengan musim.
  - Lift dan Confidence di setiap musim menunjukkan variasi kecil, yang bisa dimanfaatkan untuk mengoptimalkan penawaran di setiap musim sesuai dengan tren pembelian pelanggan.
"""
