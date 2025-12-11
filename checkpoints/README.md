# Model Checkpoints

Folder ini adalah tempat penyimpanan file bobot model (model weights) yang telah dilatih (`.pth`).

## File Model

* **`resnet_model_best.pth`**: Model terbaik menggunakan arsitektur ResNet50 (Transfer Learning). Biasanya memiliki akurasi tertinggi.
* **`cnn_model_best.pth`**: Model terbaik menggunakan arsitektur Custom CNN sederhana.

## Cara Menggunakan
Model ini akan diload secara otomatis oleh `src/utils.py` atau `app/app.py` berdasarkan konfigurasi di `src/config.py`.

*Notes: File model berukuran besar mungkin tidak disertakan dalam repository Git (ada di .gitignore).*