# ShuttleSense
# ShuttleSense: Pelatih Bulu Tangkis Berbasis AI

ShuttleSense adalah platiform yang menggunakan **komputer vision** dan **machine learning** untuk membantu pengguna memeriksa postur tubuh mereka saat melakukan gerakan dalam bulu tangkis. Aplikasi ini mendeteksi pose pengguna secara real-time dan memberikan umpan balik langsung apakah postur mereka sudah benar atau masih perlu diperbaiki, berdasarkan model AI yang telah dilatih.

---

### Fitur Utama

-   **Deteksi Pose Real-Time**: Menggunakan MediaPipe untuk mendeteksi landmark tubuh secara langsung dari kamera.
-   **Umpan Balik Instan**: Memberikan notifikasi "BENAR!" atau "SALAH!" berdasarkan analisis postur.
-   **Panduan Postur**: Menampilkan petunjuk langkah demi langkah untuk postur yang benar jika deteksi menunjukkan kesalahan.
-   **UI**: Dibangun dengan Streamlit, membuatnya mudah untuk dijalankan dan diakses.

---

### Prasyarat

Sebelum menjalankan ini, pastikan Anda telah menginstal Python (disarankan versi 3.8 atau yang lebih baru).

### Instalasi

1.  **Instal pustaka yang diperlukan**:
    Aplikasi ini menggunakan beberapa pustaka Python. Anda dapat menginstalnya menggunakan command
    ```
    pip install -r requirements.txt
    ```

---

2. ### Cara Menjalankan Aplikasi

-  Pastikan Anda berada di direktori proyek.
-  Jalankan aplikasi Streamlit dari terminal:
    ```
    python -m streamlit run webapp.py
    ```

3.  Setelah perintah dijalankan, platform akan terbuka di browser web Anda.

### Konten Proyek

-   `Shuttle.py`: Kode sumber utama aplikasi.
-   `lgbm_classifier.pkl`: Model _machine learning_ yang dilatih untuk mengklasifikasikan pose.
-   `requirements.txt`: Daftar semua pustaka Python yang diperlukan.
-   `README.md`: File ini.

---
