<h1 align="center">  This is a Machine Learning / AI Project </h1>

<p align="center"> 
Repository Massive III Bhaskara Chipta_AI Division
</p>

<div align="center">
    <!-- Your badges here -->
    <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54">
    <img src="https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white">
    <img src="https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white">
    <img src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white">
    <img src="https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white">
    <img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white">
    <img src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white">
    <img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white">
    <img src="https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB">
    <img src="https://img.shields.io/badge/tailwindcss-%2338B2AC.svg?style=for-the-badge&logo=tailwind-css&logoColor=white">
</div>

## Teams

- (Design Researcher)
- (Data Engineer)
- (Machine Learning Engineer)
- (Machine Learning Ops)


# Before We Start

### Overview

Repository ini akan memuat keseluruhan hasil dari apa saja yang telah kami lakukan. Pada README ini, saya akan menjelaskan lebih dalam mengenai WatsonX Assistant, sebuah layanan dari IBM Cloud yang memungkinkan pembuatan chatbot dan integrasinya yang lebih mudah ke aplikasi mobile.

Mengapa ada folder "model building"? Folder tersebut memuat dua hasil pembuatan model ML kami menggunakan TensorFlow dan Huggingface Transformer.

### Validation
![Screenshot 2024-06-18 091122](https://github.com/GufronAridho/Test1/assets/119670148/78b324a9-f44d-459f-807a-9478ac5ec598)

Pada Screnenshot diatas menampilkan penilaian kami memutuskan untuk menggunakan WatsonX Assistant untuk diintegrasikan ke aplikasi mobile project massive collab ini, penilaianya berupa berikut
- Data Preperation
- Model Building
- Model Evaluation
- Integration
- Response

# Lets Start Building

## Idea Background

### 1. Theme
Tema : Siaga Gempa Bumi

### 2. Problem
Masalah : 

Terlambatnya peringatan gempa menyebabkan kesulitan evakuasi, sementara distribusi bantuan yang tidak merata memperburuk situasi di wilayah pelosok yang terabaikan. Proses evakuasi yang lambat dan tidak terorganisir, disertai respons darurat yang tidak efektif, mengakibatkan kerugian dan kesulitan bagi korban gempa. Raihan juga mengalami kesulitan mengakses informasi terkini, menyoroti kekurangan dalam sistem informasi yang dapat diandalkan.

Dengan demikian, permasalahan terkait ketidakefisienan sistem peringatan, distribusi bantuan, evakuasi, respons darurat, dan keterbatasan informasi menjadi fokus utama dalam upaya perbaikan dan peningkatan kesiapsiagaan terhadap gempa bumi di Indonesia.

### 3. Solution
Solusi : 

Aplikasi Peringatan Gempa yang kami hadirkan merupakan landasan kokoh dalam menjawab tantangan kritis terkait keamanan dan kesigapan masyarakat di hadapan potensi bahaya gempa. Kami tidak hanya berfokus pada memberikan informasi, tetapi juga memberikan kontribusi yang positif dan berdampak nyata dalam upaya penanggulangan dampak bencana. Dengan memadukan teknologi dan pemahaman mendalam terhadap kebutuhan masyarakat, fitur-fitur seperti SOS yang dapat melakukan pemanggilan darurat dengan cepat serta memberikan notifikasi peringatan gempa kepada warga di sekitar pusat gempa. Fitur riwayat juga memberikan riwayat lengkap dan bisa mengetahui daerah yang rawan akan terjadinya gempa. Sementara itu edukasi membantu masyarakat mengetahui informasi gempa melalui artikel dan video dan fitur Mari Bertanya  dengan implementasi chatbot yang mampu melayani masyarakat tanpa batasan waktu pada pertanyaan yang ia miliki dan juga Tidak lupa donasi akan sangat membantu warga yang terkena dampak terjadinya gempa. Dalam aplikasi ini membentuk solusi komprehensif yang menggabungkan pendidikan, respons darurat, serta aksesibilitas untuk menciptakan ekosistem yang aman dan siap menghadapi risiko gempa.

## Dataset and Algorithm

### 1. Dataset
- Data Collection <br />
Kami menemukan data pertanyaan kami dari berbagai situs yang memuat pembahasan mengenai gempa, beberapa situsnya adalah sebagai berikut: <br />
https://www.usgs.gov/programs/earthquake-hazards/faqs-category <br />
https://www.earthquakescanada.nrcan.gc.ca/info-gen/faq-en.php <br />
https://scweb.cwa.gov.tw/en-us/guidance/faq <br />
https://www.earthquakes.bgs.ac.uk/education/faqs/faq_index.html <br />
https://polarisdrt.org/100-frequently-asked-questions-about-earthquakes-and-their-answers/ <br />
https://www.bmkg.go.id/ <br />
https://id.quora.com/

- Data Cleaning <br />
Data Cleaning yang kami lakukan dibuat secara mmanual, dengan point pemilihan seperti :
> "Apakah pertanyaan tersebut relate ke wilayah Indonesia ?" <br />
> "Apakah pertanyaan tersebut berguna ?"

- Data Transformation <br />
IBM WatsonX Assistant memungkinkan kita untuk mengupload data intents yang kita buat dalam format .csv untuk dijadikan actions dengan format data seperti berikut:
`<phrase>,<intent>`

Contoh 
```
Apa itu gempa bumi?,pengertian_gempa
Jelaskan gempa bumi,pengertian_gempa
Bisakah Anda menjelaskan tingkatan skala SIG BMKG?,tingkatan_skala_sig
Tingkatan skala SIG BMKG dijelaskan bagaimana?,tingkatan_skala_sig
Berikan saya definisi gempa vulkanik,gempa_vulkanik
Apa sih yang dimaksud dengan gempa vulkanik,gempa_vulkanik
```

### 2. Project FlowChart

![Watson Asisstant Project Flow](https://github.com/GufronAridho/Test1/assets/119670148/642a3d49-b943-4658-aad5-b4454624fcd3)

### 3. Algorithm

- Framework <br />
Kami menggunanakan WatsonX Assistant. Dengan versi algoritma terbaru (15-Apr-2023) dari setting watsonx assistant Yang mana versi algoritma ini menggunakan foundation model baru untuk meningkatkan deteksi niat dan pencocokan tindakan di asisten, foundation model ini dilatih dengan menggunakan arsitektur transformator.

- Pembangunan Model

Akan saya jelaskan 2 cara untuk membuat Actions skills untuk chatbot ini, yaitu:
1. Upload Actions melalui csv intents
     - Pada halaman Actions utama klik ikon upload intens
     - Pilih file intens yang Anda miliki
     - Setelah di upload WatsonX akan memvalidasi data Anda dan melatih systemnya berdasarkan data tersebut
2. Membuat Actions secara manual
     - Pada halaman Actions utama klik "New Actions"
     - Pada bagian "Add example phrase:" Isi dengan Masukkan frasa yang diketik atau diucapkan pelanggan untuk memulai percakapan tentang topik tertentu. Frasa ini menentukan tugas, masalah, atau pertanyaan yang dimiliki pelanggan Anda. Semakin banyak frasa yang Anda masukkan, semakin baik asisten Anda mengenali apa yang diinginkan pelanggan.

Sekarang kita sudah memiliki Actions, waktu nya untuk menambah response atau jawaban dari Actions atau pertanyaan yang sudah kita buat. 
1. Pada halaman Actions utama pilih Actions yang sudah dibuat
2. Pada Conversation steps Anda bisa menambahkan apa yang harus assistant katanan, isi "Assistant says" dengan jawaban dari pertanyaan tersebut
3. Kita bisa mengkomplekskan jawaban dari chatbot ini dengan menambah Options pada "Define customer response" untuk disambungkan ke Actions lainnya, contoh jika kita masuk ke Actions "jenis gempa" kita bisa menyambungkannya ke Actions lain seperti "gemppa tektonik", "gempa vulkanik", "gempa buatan" dan lain lain. Cara nya adalah sebagai berikut:
     - Buat Options pada steps nya
     - Buat steps baru dan atur "Is taken" menjadi "with condition" dan sesuaikan conditions nya dengan Options yang telah dibuat
     - Scroll kebawah dan pada bagian "And Then" ubah menjadi "Goes to a subaction" dan atur sesuai dengan Actions mana yang Anda mau

Kita sudah membuat Action yang akan dipakai oleh chatbot untuk menjawab pertanyaan pengguna nantinya, jangan lupa untuk mereview chatbot yang telah dibuat dan mempublish dengan cara pergi ke tab Publish lalu klik Publish. 

- Model Evaluation <br />

## Prototype

![Watson Asisstant Flow](https://github.com/Bhaskara-Cipta/AAI/assets/173335150/8e374138-ed78-4339-8775-dd15221cfa6b)

Flowchart ini menggambarkan alur percakapan atau dialog untuk sebuah chatbot dari Watson Assistant. Proses dimulai dari "Open" atau membuka percakapan. Pengguna kemudian diminta untuk "Input Question" atau memasukkan pertanyaan. Pertanyaan pengguna ini akan diproses oleh "Watson Assistant" yang memiliki kemampuan memahami dan mengolah bahasa alami. Setelah itu, sistem akan melakukan pencarian informasi yang relevan melalui "Searching Action Skills" yang melibatkan penelusuran pada basis pengetahuan atau sumber eksternal untuk menemukan jawaban atas pertanyaan pengguna. Sistem akan menghasilkan "Question Response" atau respons jawaban berdasarkan hasil pencarian tersebut. Setelah itu, sistem akan memeriksa apakah terdapat opsi atau tindakan lanjutan yang diperlukan. Jika tidak ada opsi ("No Option"), pengguna akan ditanya "Ask Again?" untuk mengajukan pertanyaan lain. Jika pengguna memilih tidak bertanya lagi ("Close"), maka alur percakapan akan berakhir. Namun, jika terdapat opsi ("There Are Option"), sistem akan menampilkan langkah "Pick One" di mana pengguna dapat memilih dari opsi yang tersedia. Apabila pengguna memilih opsi "Tidak Penasaran", maka alur percakapan akan berakhir dengan "End Dialog Flow". Jika pengguna memilih opsi lain, sistem akan memberikan respons yang sesuai ("Pick Response"). Setelah itu, pengguna akan ditanya apakah ingin "Repeat Pick?" atau memilih opsi lain. Jika ya, maka akan kembali ke langkah "Pick One". Jika tidak, maka alur percakapan akan berakhir.

## Deployment

Watsonx asisten adalah salah satu service yang disediakan oleh IBM Cloud untuk membuat sebuah chatbot . Di watsonx asisten kita bisa membuat, menguji, mempublish, mendeploy dan menganalisis chatbot kita dalam antarmuka yang sederhana dan interaktif, watsonx asisten membuat chatbot yang telah kita buat sudah di deploy dan bisa diakses dari mana saja dengan langkah yang simple.

Setelah kita sudah selesai membuat Actions di halaman utama Watsonx asisten kita bisa melakukan pengujian pada chatbot ini melalui tab Preview, ketika dirasa sudah cukup masuk pada tab Publish untuk mempublish chatbot ini pada environment Live agar dapat di deploy dan diintegrasikan ke aplikasi mobile melalui webview.

![1](https://github.com/Bhaskara-Cipta/AAI/assets/173335150/ab74bd4b-26df-46e7-ac8b-10dc730a4338)

Selanjutnya kita akan masuk pada tab integration lalu pada bagian webchat pilih environment Live, di environment Live ini kita mengatur kembali tampilan dari chatbot nya dan penyesuain lainya. Untuk mengintegrasikan chatbot ini yang sudah dideploy di Cloudnya IBM watsonx asisten adalah penggunaan script embed yang akan ada pada gambar berikut untuk menyambungkan pengguna dengan fitur chatbot ini.

![2](https://github.com/Bhaskara-Cipta/AAI/assets/173335150/ab133407-8149-445e-8cff-1744df6b41a4)

## Integration

Proses integrasi chatbot akan menggunakan embed code yang ada pada bagian sebelumnya, script tersebut akan dimuat pada suatu file html untuk dijadikan perantara agar Webview dapat dilakukan  untuk mengakses chatbot tersebut di aplikasi mobile dan website.

Teknis dari integrasi chatbot ini ke mobile memiliki penjelasan langkah singkat sebagai berikut:

1. Membuat file .html sederhana yang mengandung script dari embed code tersebut lalu upload ke dalam folder assets.
2. Membuat file webview component untuk memungkinan aplikasi mengakses konten yang berasal dari web ini sangat berpengaruh untuk mengakses file .html yang mengandung script embed tersebut 
![WhatsApp Image 2024-06-14 at 20 42 01](https://github.com/Bhaskara-Cipta/AAI/assets/173335150/de3d35be-c4c7-4774-8afd-2a8097ad3258)

4. Selanjutnya kita membuat javascript interface yang akan bertanggung jawab untuk menangani pengiriman pesan pengguna ke watsonx assistant  
![WhatsApp Image 2024-06-14 at 20 35 09](https://github.com/Bhaskara-Cipta/AAI/assets/173335150/cf870769-a7f0-474b-9ca3-24d1db97bba0)

6. Dan terakhir kita membuat file .kt untuk membuat screen chatbot dan memanggil fungsi dari kedua file tersebut.
![WhatsApp Image 2024-06-14 at 20 34 36](https://github.com/Bhaskara-Cipta/AAI/assets/173335150/7c4f2fa5-499b-4b61-8a1e-7855230cd282)

## Result
Kami telah mengembangkan fitur AI dalam bentuk chatbot dengan menggunakan Watsonx Assistant untuk memberikan informasi dan jawaban secara real-time mengenai gempa bumi. Chatbot AI ini didesain untuk membantu pengguna dengan pertanyaan seputar gempa, langkah-langkah keselamatan, dan pengetahuan umum tentang gempa. Dengan integrasi Watsonx Assistant, chatbot kami dapat memahami dan memproses pertanyaan dalam bahasa alami dengan efisien, memastikan pengguna mendapatkan informasi yang akurat dan tepat waktu.

![WhatsApp Image 2024-06-25 at 22 19 49](https://github.com/Bhaskara-Cipta/AAI/assets/119670148/3afffad4-bbc8-4569-8692-c7316696127b)

Selain itu, chatbot ini mampu menawarkan topik yang relevan dengan pertanyaan pengguna, memberikan solusi jawaban yang komprehensif. Dengan Watsonx Assistant, kami dapat menganalisis kepuasan pengguna dan melakukan perbaikan pada chatbot tanpa perlu mengubah kode dalam aplikasi mobile. Chatbot ini dapat diakses kapan saja dan diperbarui berdasarkan analisis yang dilakukan tanpa perlu hard coding.
   
![WhatsApp Image 2024-06-25 at 22 19 48](https://github.com/Bhaskara-Cipta/AAI/assets/119670148/1833b890-cd97-4625-ae18-45b4d7d0e405)


Dari segi fungsionalitas, pengguna tidak perlu lagi beralih antara web atau aplikasi lain untuk mendapatkan informasi seputar gempa, karena chatbot ini dirancang khusus untuk memberikan informasi terkait gempa. Fitur ini tidak hanya meningkatkan pengalaman pengguna, tetapi juga memberikan dukungan penting dan fleksibilitas untuk beradaptasi dan berkembang seiring berjalannya waktu.
   
![WhatsApp Image 2024-06-25 at 22 19 49 (1)](https://github.com/Bhaskara-Cipta/AAI/assets/119670148/209e37d5-287a-4e77-b7bf-0e246e9d2979)


## Conclusion
Indonesia sangat rawan gempa bumi karena berada di Cincin Api Pasifik, menyebabkan kerugian besar. Meskipun ada upaya mitigasi, banyak masyarakat kurang paham tentang tindakan yang harus diambil sebelum, saat, dan setelah gempa. Akses informasi cepat dan akurat menjadi kendala utama. 

Teknologi informasi dan komunikasi dapat menjadi solusi efektif, seperti pengembangan Chatbot “Mari Bertanya” adalah solusi yang kami kembangkan dalam aplikasi Siaga Gempa. Chatbot ini bertujuan untuk memuaskan keingintahuan pengguna mengenai gempa bumi di Indonesia dengan memberikan jawaban dari pertanyaan nya yang responsif, relevan dan konsisten dari waktu ke waktu. Melalui pendekatan ini, kami berharap untuk memberdayakan individu dengan pengetahuan tentang gempa bumi, memungkinkan mereka mengambil keputusan dan mengambil tindakan proaktif untuk menjamin keselamatan dan kesejahteraan mereka.

Dengan penggunaan watsonx asisten sebagai platfrom utama pembuatan, pendeployan dan analisis chatbot untuk integrasi yang bisa dilakukan di platfrom mobile maupun website. Model yang dikembangkan pada watsonx asisten mampu melakukan Natural Languange Procesing (NLP) untuk mengenali intens pengguna dari inputan pengguna, kemudian model ini akan memberikan respon berdasarkan dataset yang dibuat dan juga kelebihannya yang sudah di deploy di cloud sehingga bisa diakses kapan saja dan proses integrasi yang tidak perlu merubah banyak kode dasar dari website dan mobile
