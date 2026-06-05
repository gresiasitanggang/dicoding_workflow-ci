# Intel Image Preprocessing

Dataset asli yang terdiri dari 24.000+ gambar telah dikompresi menjadi `dataset.zip` demi efisiensi waktu eksekusi pada workflow CI. File tersebut dilacak menggunakan Git LFS (`with Git LFS tracking`). 

Proses ekstraksi dilakukan secara otomatis oleh script `modelling.py` sesaat sebelum proses pelatihan model dimulai di dalam lingkungan runner CI.