# Dataset Documentation

Dataset ini berisi citra daun kentang yang diklasifikasikan ke dalam 3 kelas utama.

## Dataset Information

Deteksi penyakit daun kentang pada tahap awal cukup menantang karena adanya variasi spesies tanaman, gejala penyakit, dan faktor lingkungan. Hal ini membuat pendeteksian penyakit sejak dini menjadi sulit. Berbagai teknik machine learning telah dikembangkan, namun kebanyakan model hanya dilatih pada gambar dari wilayah tertentu sehingga kurang mampu mengenali penyakit secara umum.

Dalam penelitian ini, dikembangkan model deep learning bertingkat untuk mengenali penyakit daun kentang. Pada tahap pertama, YOLOv5 digunakan untuk mengekstraksi daun dari gambar tanaman. Pada tahap kedua, model CNN khusus dirancang untuk mendeteksi penyakit early blight dan late blight dari gambar daun kentang. Dataset yang digunakan berisi 4062 gambar dari Punjab, Pakistan.

Model yang diusulkan mencapai akurasi 99,75% dan juga dievaluasi pada dataset PlantVillage. Dibandingkan dengan metode lain, model ini menunjukkan kinerja lebih baik baik dari sisi akurasi maupun efisiensi komputasi.

source: https://www.kaggle.com/datasets/rizwan123456789/potato-disease-leaf-datasetpld/data

## Dataset Structure

```
Dataset/
├── Training/
│   ├── Early_Blight/
│   ├── Healthy/
│   └── Late_Blight/
├── Validation/
│   ├── Early_Blight/
│   ├── Healthy/
│   └── Late_Blight/
└── Testing/
    ├── Early_Blight/
    ├── Healthy/
    └── Late_Blight/
```

## Detail Data

Total Gambar: +/- 2,152 gambar.

Preprocessing:
- Resize: 224x224 pixels.
- Normalization: Menggunakan mean/std ImageNet.
- Augmentation: Random Rotation, Horizontal Flip, Color Jitter.

Notes: Dataset kami zip karena terlalu besar sizenya.