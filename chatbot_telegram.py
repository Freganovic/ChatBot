import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Konfigurasi logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Data FAQ
faq_data = [
    {"question": "Apa jadwal kereta dari Cikarang ke Jakarta?", "answer": "Jadwal kereta dari Cikarang ke Jakarta tersedia mulai pukul 05.00 hingga 22.00 dengan interval 30 menit."},
    {"question": "Bagaimana cara memesan tiket kereta?", "answer": "Anda dapat memesan tiket melalui aplikasi KAI Access atau langsung di loket stasiun."},
    {"question": "Berapa harga tiket kereta lokal dari Cikarang?", "answer": "Harga tiket kereta lokal dari Cikarang adalah Rp5.000 untuk semua rute."},
    {"question": "Apa saja jadwal keberangkatan kereta dari Stasiun Cikarang?", "answer": "Jadwal keberangkatan kereta dari Stasiun Cikarang dapat dilihat di situs resmi KAI atau melalui aplikasi KAI Access."},
    {"question": "Apakah ada kereta api yang langsung menuju Jakarta dari Stasiun Cikarang?", "answer": "Ya, ada kereta api dari Stasiun Cikarang menuju Jakarta, seperti kereta api ekonomi dan bisnis."},
    {"question": "Apakah kereta api dari Stasiun Cikarang ada yang menuju Bandung?", "answer": "Saat ini, kereta api dari Stasiun Cikarang menuju Bandung belum tersedia. Anda bisa menggunakan kereta menuju Jakarta dan melanjutkan perjalanan ke Bandung."},
    {"question": "Apa yang harus dilakukan jika tiket kereta hilang?", "answer": "Jika tiket hilang, segera laporkan ke petugas di stasiun atau melalui aplikasi KAI Access untuk mendapatkan solusi."},
    {"question": "Apakah ada WiFi di Stasiun Cikarang?", "answer": "Stasiun Cikarang menyediakan fasilitas Wi-Fi gratis di area tertentu. Anda bisa mengaksesnya dengan mengikuti petunjuk di stasiun."},
    {"question": "Berapa lama perjalanan kereta dari Stasiun Cikarang ke Jakarta?", "answer": "Perjalanan kereta dari Stasiun Cikarang ke Jakarta biasanya memakan waktu sekitar 1 hingga 1,5 jam, tergantung jenis kereta."},
    {"question": "Apakah ada kereta api yang ramah bagi penyandang disabilitas di Stasiun Cikarang?", "answer": "Ya, beberapa kereta api dan fasilitas di Stasiun Cikarang telah dilengkapi untuk mendukung penyandang disabilitas."},
    {"question": "Apakah ada layanan parkir dan parkiran di Stasiun Cikarang?", "answer": "Stasiun Cikarang menyediakan area parkir untuk penumpang yang ingin meninggalkan kendaraan saat bepergian dengan kereta."},
    {"question": "Apakah ada kereta api pagi dari Stasiun Cikarang menuju Jakarta?", "answer": "Ya, ada beberapa kereta api pagi yang berangkat dari Stasiun Cikarang menuju Jakarta. Silakan cek jadwal di aplikasi KAI Access."},
    {"question": "Apakah Stasiun Cikarang menyediakan ruang tunggu?", "answer": "Stasiun Cikarang menyediakan ruang tunggu bagi penumpang dengan fasilitas nyaman."},
    {"question": "Bagaimana cara membatalkan tiket kereta api?", "answer": "Untuk membatalkan tiket, Anda bisa melakukannya melalui aplikasi KAI Access atau di loket stasiun dengan ketentuan yang berlaku."},
    {"question": "Berapa banyak jam sebelum keberangkatan saya harus berada di Stasiun Cikarang?", "answer": "Disarankan untuk tiba di stasiun setidaknya 30 menit sebelum keberangkatan kereta."},
    {"question": "Apakah ada fasilitas makanan di Stasiun Cikarang?", "answer": "Stasiun Cikarang memiliki berbagai kios makanan dan minuman di area keberangkatan dan kedatangan."},
    {"question": "Apakah saya bisa mengubah nama pada tiket kereta api?", "answer": "Nama pada tiket kereta api tidak dapat diubah. Jika perlu, Anda harus membatalkan tiket dan membeli tiket baru."},
    {"question": "Apakah kereta api dari Stasiun Cikarang ada yang menuju Yogyakarta?", "answer": "Kereta api dari Stasiun Cikarang menuju Yogyakarta tidak tersedia langsung, tetapi Anda bisa transit di Jakarta untuk melanjutkan perjalanan ke Yogyakarta."},
    {"question": "Apa saja fasilitas yang tersedia di kereta ekonomi dari Stasiun Cikarang?", "answer": "Kereta ekonomi menyediakan tempat duduk yang nyaman, toilet, dan penjualan makanan dan minuman di dalam kereta."},
    {"question": "Apakah ada layanan kereta api malam hari dari Stasiun Cikarang?", "answer": "Saat ini, kereta api malam dari Stasiun Cikarang tidak tersedia. Kereta umumnya beroperasi hingga malam hari, namun tidak ada keberangkatan larut malam."},
    {"question": "Bagaimana cara mengetahui jika ada perubahan jadwal kereta dari Stasiun Cikarang?Apakah saya bisa membeli tiket kereta api untuk perjalanan di masa depan?", "answer": "Untuk mengetahui perubahan jadwal, Anda bisa memeriksa aplikasi KAI Access atau menghubungi layanan pelanggan KAI.Ya, Anda bisa membeli tiket kereta api untuk perjalanan di masa depan melalui aplikasi KAI Access atau di loket stasiun."},
    {"question": "Apa yang harus dilakukan jika tiket kereta saya tidak dapat diverifikasi?", "answer": "Jika tiket Anda tidak dapat diverifikasi, segera hubungi petugas di stasiun atau menggunakan layanan pelanggan KAI untuk bantuan lebih lanjut."},
    {"question": "Apakah ada layanan khusus untuk penumpang dengan anak-anak di Stasiun Cikarang?", "answer": "Ya, Stasiun Cikarang menyediakan layanan dan fasilitas yang ramah keluarga, termasuk area tunggu yang nyaman untuk penumpang dengan anak-anak."},
    {"question": "Apakah kereta api dari Stasiun Cikarang menuju Surabaya?", "answer": "Saat ini, tidak ada kereta api langsung dari Stasiun Cikarang menuju Surabaya. Anda bisa transit di Jakarta untuk melanjutkan perjalanan ke Surabaya."},
    {"question": "Bagaimana cara memesan tiket kereta secara online?", "answer": "Anda bisa memesan tiket kereta secara online melalui aplikasi KAI Access atau melalui website resmi KAI."},
    {"question": "Apakah ada kereta api yang menyediakan kursi prioritas di Stasiun Cikarang?", "answer": "Ya, beberapa kereta api menyediakan kursi prioritas bagi lansia, ibu hamil, atau penumpang dengan disabilitas."},
    {"question": "Bagaimana cara mengetahui jika kereta api mengalami pembatalan?", "answer": "Informasi tentang pembatalan kereta api dapat dilihat di aplikasi KAI Access atau diumumkan di stasiun."},
    {"question": "Apakah ada layanan antar jemput dari Stasiun Cikarang ke hotel?", "answer": "Saat ini, Stasiun Cikarang tidak menyediakan layanan antar jemput ke hotel, tetapi Anda bisa menggunakan taksi atau layanan ojek online."},
    {"question": "Apakah ada layanan kereta api yang membawa hewan peliharaan?", "answer": "Hewan peliharaan dapat dibawa dengan syarat dan ketentuan yang berlaku di kereta api. Pastikan untuk memeriksa kebijakan KAI terkait membawa hewan peliharaan."},
    {"question": "Bagaimana cara memeriksa ketersediaan tempat duduk di kereta?", "answer": "Anda dapat memeriksa ketersediaan tempat duduk di kereta melalui aplikasi KAI Access atau di loket stasiun."},
    {"question": "Apakah ada kereta api yang bisa diakses oleh penyandang disabilitas di Stasiun Cikarang?", "answer": "Stasiun Cikarang telah dilengkapi dengan fasilitas aksesibilitas untuk penyandang disabilitas, termasuk jalur khusus dan fasilitas toilet."},
    {"question": "Apakah ada kereta api yang menuju Cirebon dari Stasiun Cikarang?", "answer": "Saat ini, tidak ada kereta api langsung dari Stasiun Cikarang menuju Cirebon, tetapi Anda dapat transit di Jakarta untuk melanjutkan perjalanan ke Cirebon."},
    {"question": "Apa saja rute yang dilayani oleh KRL dari Stasiun Cikarang?", "answer": "KRL dari Stasiun Cikarang melayani rute Cikarang - Jakarta Kota, Cikarang - Bekasi, dan Cikarang - Tanah Abang."},
    {"question": "Jam operasional KRL dari Stasiun Cikarang?", "answer": "KRL dari Stasiun Cikarang beroperasi mulai pukul 05.00 hingga 22.00 WIB dengan frekuensi keberangkatan setiap 30 menit."},
    {"question": "Berapa harga tiket KRL dari Stasiun Cikarang ke Jakarta?", "answer": "Harga tiket KRL dari Stasiun Cikarang ke Jakarta mulai dari Rp 8.000 hingga Rp 15.000, tergantung pada tujuan stasiun."},
    {"question": "Bagaimana cara membeli tiket KRL di Stasiun Cikarang?", "answer": "Tiket KRL dapat dibeli melalui mesin tiket otomatis, aplikasi KAI Access, atau loket stasiun."},
    {"question": "Apakah ada diskon untuk pelajar atau mahasiswa?", "answer": "Ya, pelajar dan mahasiswa dapat menggunakan kartu multi-trip (KMT) untuk mendapatkan diskon pada tiket KRL."},
    {"question": "Bagaimana cara top up kartu KRL (KMT)?", "answer": "Kartu KMT dapat diisi ulang melalui mesin EDC yang tersedia di stasiun atau di minimarket yang bekerja sama dengan KAI."},
    {"question": "Apakah KRL menyediakan layanan WiFi?", "answer": "Saat ini, KRL belum menyediakan layanan Wi-Fi di dalam kereta."},
    {"question": "Bagaimana cara mencari jadwal KRL dari Stasiun Cikarang?", "answer": "Jadwal KRL dapat dilihat di papan informasi stasiun, aplikasi KAI Access, atau melalui website resmi KAI."},
    {"question": "Apakah ada layanan kereta KRL yang melayani rute malam?", "answer": "Layanan KRL di Stasiun Cikarang beroperasi hingga pukul 22.00 WIB, tidak ada layanan KRL pada malam hari."},
    {"question": "Berapa lama perjalanan KRL dari Stasiun Cikarang ke Jakarta?", "answer": "Perjalanan KRL dari Stasiun Cikarang ke Jakarta memakan waktu sekitar 1 hingga 1,5 jam tergantung rute dan tujuan."},
    {"question": "Apakah ada layanan KRL untuk perjalanan jarak jauh dari Stasiun Cikarang?", "answer": "KRL di Stasiun Cikarang hanya melayani perjalanan dalam kota dan sekitarnya, tidak untuk perjalanan jarak jauh."},
    {"question": "Bagaimana jika saya kehilangan tiket KRL?", "answer": "Jika tiket KRL hilang, Anda dapat membeli tiket baru dengan harga yang sesuai pada perjalanan berikutnya."},
    {"question": "Apakah ada stasiun pemberhentian sebelum Stasiun Cikarang?", "answer": "Ya, beberapa stasiun sebelum Stasiun Cikarang antara lain Stasiun Bekasi, Stasiun Cibitung, dan Stasiun Tambun."},
    {"question": "Apakah KRL di Stasiun Cikarang memiliki layanan kursi prioritas?", "answer": "KRL memiliki kursi prioritas bagi penyandang disabilitas, lansia, dan ibu hamil yang tersedia di gerbong tertentu."},
    {"question": "Apa saja hal yang di larang di dalam kereta krl", "answer": "Penumpang dilarang membawa senjata berbahaya, obat obatan seperti narkoba, barang yang melebihi ukuran muatan yang di tentukan oleh pihak KAI, dan hewan langka yang di lindungi"},
    {"question": "Apakah saya boleh makan dan minum di dalam kereta KRL?", "answer": "Makan dan minum di dalam kereta KRL dilarang untuk menjaga kebersihan dan kenyamanan penumpang lainnya."},
    {"question": "Bolehkah membawa hewan peliharaan di dalam kereta KRL?", "answer": "Hewan peliharaan dilarang dibawa ke dalam kereta KRL, kecuali jika hewan tersebut dalam kandang yang aman dan memenuhi ketentuan yang berlaku."},
    {"question": "Apakah boleh merokok di dalam kereta KRL?", "answer": "Merokok di dalam kereta KRL dilarang keras. KRL adalah area bebas asap rokok untuk kenyamanan semua penumpang."},
    {"question": "Bolehkah saya membawa sepeda di dalam kereta KRL?", "answer": "Sepeda diperbolehkan dibawa ke dalam kereta KRL dengan ketentuan tertentu, seperti membayar tiket tambahan dan mengikuti peraturan yang berlaku."},
    {"question": "Apakah boleh membawa barang yang berbau menyengat di dalam kereta?", "answer": "Barang yang memiliki bau menyengat seperti ikan, durian, dan sejenisnya dilarang dibawa ke dalam kereta untuk menjaga kenyamanan penumpang."},
    {"question": "Apakah boleh atau diperbolehkan tidur di kursi selama perjalanan di dalam kereta KRL?", "answer": "Tidur di kursi diperbolehkan, namun penumpang harus tetap menjaga sikap sopan dan tidak menghalangi kursi atau tempat duduk orang lain."},
    {"question": "Apa jam keberangkatan kereta api terakhir dari Stasiun Cikarang?", "answer": "Kereta api terakhir dari Stasiun Cikarang menuju Jakarta berangkat pada pukul 22:00."},
    {"question": "baik terimakasih banyak atas bantuanya", "answer": "sama sama, saya senang dapat membantu jika ada yang ingin di tanyakan jangan ragu untuk menghubungi saya."},
    {"question": "Jam berapa jadwal keberangkatan KRL pertama dari Stasiun Cikarang?", "answer": "Jadwal keberangkatan KRL pertama dari Stasiun Cikarang adalah pukul 04:00 pagi."},
    {"question": "Jam berapa jadwal keberangkatan KRL terakhir dari Stasiun Cikarang?", "answer": "Jadwal keberangkatan KRL terakhir dari Stasiun Cikarang adalah pukul 22:00 malam."},
    {"question": "Rute perjalanan KRL dari Stasiun Cikarang ke Tanah Abang lewat mana?", "answer": "Anda perlu transit di Stasiun Manggarai, kemudian melanjutkan perjalanan dengan kereta tujuan"},
    {"question": "berapa harga tiket krl menuju ke stasiun cibitung?", "answer": "tarif krl menuju stasiun tersebut sebesar Rp.3000 rupiah"},
    {"question": "berapa harga tiket krl menuju ke stasiun tambun?", "answer": "tarif krl menuju stasiun tersebut sebesar Rp.3000 rupiah"},
    {"question": "berapa harga tiket krl menuju ke stasiun bekasi?", "answer": "tarif krl menuju stasiun tersebut sebesar Rp.3000 rupiah"},
    {"question": "berapa harga tiket krl menuju ke stasiun bekasi timur?", "answer": "tarif krl menuju stasiun tersebut sebesar Rp.3000 rupiah"},
    {"question": "berapa harga tiket krl menuju ke stasiun kranji?", "answer": "tarif krl menuju stasiun tersebut sebesar Rp.3000 rupiah"},
    {"question": "berapa harga tiket krl menuju ke stasiun cakung?", "answer": "tarif krl menuju stasiun tersebut sebesar Rp.3000 rupiah"},
    {"question": "berapa harga tiket krl menuju ke stasiun jatinegara?", "answer": "tarif krl menuju stasiun tersebut sebesar Rp.4000 rupiah"},
    {"question": "berapa harga tiket krl menuju ke stasiun manggarai?", "answer": "tarif krl menuju stasiun tersebut sebesar Rp.4000 rupiah"},
    {"question": "berapa harga tiket krl menuju ke stasiun depok?", "answer": "tarif krl menuju stasiun tersebut sebesar Rp.7000 rupiah"},
    {"question": "berapa harga tiket krl menuju ke stasiun juanda?", "answer": "tarif krl menuju stasiun tersebut sebesar Rp.5000 rupiah"},
    {"question": "Kereta api apa yang tujuan menuju Cikarang-Nganjuk?", "answer": "Untuk kereta api tujuan Cikarang-Nganjuk anda bisa memilih beberapa jenis kereta. Salah satu opsi yang tersedia yaitu kereta api singasari."},
    {"question": "Kereta api apa yang tujuan menuju Cikarang-Yogyakarta", "answer": "Untuk kereta api tujuan Cikarang-Yogyakarta, anda bisa memilih beberapa jenis kereta api. Salah satu opsi yang tersedia yaitu kereta api bengawan."},
    {"question": "Kereta api apa saja yang tujuan menuju ke stasiun Surabaya Pasar Turi?", "answer": "Untuk tujuan Surabaya tersedia kereta api Kertajaya, Pandalungan, dan Airlangga."},
    {"question": "Kereta api apa saja yang tujuan menuju ke stasiun Kediri?", "answer": "Untuk saat ini yang tersedia hanya kereta api Brawijaya dan Singasari"},
    {"question": "Kapan jadwal keberangkatan kereta api Singasari?", "answer": "Jam keberangkatan kereta api Singasari yaitu pukul 21:40 dari stasiun Cikarang."},
    {"question": "Kapan jadwal keberangkatan kereta api Bengawan?", "answer": "Jam keberangkatan kereta api Bengawan yaitu pukul 06:46 dari stasiun Cikarang."},
    {"question": "Kapan jadwal keberangkatan kereta api Jaka Tingkir?", "answer": "Jam keberangkatan kereta api Jaka Tingkir yaitu pukul 12:40 dari stasiun Cikarang."},
    {"question": "Kereta api apa saja yang bertujuan Bandung?", "answer": "Untuk saat ini yang tersedia hanya kereta api Cikuray."},
    {"question": "Kapan jadwal keberangkatan kereta api Jayakarta?", "answer": "Jam keberangkatan kereta api Jayakarta yaitu pukul 17:54."},
    {"question": "Kapan jadwal keberangkatan kereta api Airlangga?", "answer": "Jam keberangkatan kereta api Airlangga yaitu pukul 12:06 dari stasiun Cikarang."},
    {"question": "Kapan jadwal keberangkatan kereta api Kertajaya?", "answer": "Jam keberangkatan kereta api Kertajaya yaitu pukul 15:15 dari stasiun Cikarang."},
    {"question": "Kapan jadwal keberangkatan kereta api Pandalungan?", "answer": "Jam keberangkatan kereta api Pandalungan yaitu pukul 20:44 dari stasiun Cikarang."},
    {"question": "Kapan jadwal keberangkatan kereta api Brawijaya?", "answer": "Jam keberangkatan kereta api Brawijaya yaitu pukul 16:25 dari stasuin Cikarang."},
    {"question": "Kapan jadwal keberangkatan kereta api Cikuray?", "answer": "Jam keberangkatan kereta api Cikuray yaitu pukul 18:26 dari stasiun Cikarang."},
    {"question": "KRL ke Jakarta Kota", "answer": "Untuk tujuan Jakarta Kota bisa naik jurusan Angke transit via Manggarai. Setelah berhenti di stasiun Manggarai, anda bisa melanjutkan perjalanan dengan KRL."},
    {"question": "KRL ke Jayakarta", "answer": "Untuk tujuan Stasiun Jayakarta bisa naik jurusan Angke transit via Manggarai. Setelah berhenti di stasiun Manggarai, anda bisa melanjutkan perjalanan dengan KRL."},
    {"question": "KRL ke Stasiun Bogor", "answer": "Untuk tujuan Stasiun Bogor, anda bisa melakukan perjalanan menggunakan KRL jurusan Cikarang-Angke atau Cikarang-Kampung Bandan. Kemudian transit di Stasiun Manggarai, anda bisa melanjutkan perjalan menggunakan KRL jurusan Jakarta Kota-Bogor atau Angke-Bogor."},
    {"question": "Rute untuk ke Stasiun Bogor", "answer": "Berikut Rute perjalanan ke Stasiun Bogor. Diawali di Stasiun Cikarang, Metland Telaga Murni, Cibitung, Tambun, Bekasi Timur, Bekasi, Kranji, Klender Baru, Buaran, Klender, Jatinegara, Matraman, dan anda bisa transit di Stasiun Manggarai. Dari Manggarai anda bisa lanjut menggunakan KRL jurusan Jakarta Kota-Bogor atau Angke-Bogor, berikut rutenya. Stasiun Manggarai, Tebet, Cawang, Duren Kalibata, Pasar Minggu Baru, Pasar Minggu, Tanjung Barat, Lenteng Agung, Univ. Pancasila, Univ. Indonesia, Pondok Cina, Depok baru, Depok, Citayam, Bojong Gede, Cilebut, dan diakhiri di Stasiun Bogor."},
    {"question": "Harga tiket kereta tujuan Surabaya Pasar Turi", "answer": "Harga tiket kereta tujuan Surabaya Pasar Turi bervariasi, mulai dari Rp 104.000 - Rp 600.000, sesuai kereta dan kelas yang anda pilih."},
    {"question": "Harga tiket kereta tujuan Malang", "answer": "Harga tiket kereta tujuan Malang mulai dari Rp 760.000, menggunakan kereta api Brawijaya. Harga tergantung dengan kelas yang anda pilih."},
    {"question": "Harga tiket kereta tujuan Blitar", "answer": "Harga tiket kereta tujuan Blitar mulai dari Rp 350.000, tergantung kereta dan kelas yang anda pilih. Untuk tujuan Blitar tersedia kereta api Singasari dan Brawijaya."},
    {"question": "Harga tiket kereta tujuan Yogyakarta", "answer": "HdcpHandler.dll"},
    {"question": "Harga tiket kereta tujuan Bandung", "answer": "Harga tiket kereta tujuan Bandung yaitu Rp. 45.000. Tersedia kereta api Cikuray."},
    {"question": "Harga tiket kereta tujuan Solo", "answer": "Harga tiket kereta tujuan Solo mulai dari Rp 74.000 tergantung kereta dan kelas yang anda pilih. Tersedia kereta api Bengawan, Jayakarta, dan Singasari."},
    {"question": "Hal yang tidak diperbolehkan di dalam KRL", "answer": "Hal yang dilarang di KRL yaitu, membuka pintu secara paksa, bersandar di dekat pintu kereta, meroko di dalam kereta maupun di area stasiun, membuang sampah sembarangan, Berbicara dengan suara keras, memutar musik tanpa menggunakan earphone, membawa benda tajam, membawa hewan, dan duduk di lantai kereta."},
    {"question": "apa saja fasilitas di stasiun kai cikarang?", "answer": "stasiun kereta api kai cikarang menyediakan wifi, toilet, mushola, tempat parkir dan tempat makan."},
    {"question": "selamat pagi", "answer": "selamat pagi"},
    {"question": "selamat siang", "answer": "selamat siang"},
    {"question": "selamat sore", "answer": "selamat sore"},
    {"question": "selamat malam", "answer": "selamat malam"},
    {"question": "Apakah ada petugas yang membantu di stasiun bagi penumpang yang membutuhkan bantuan?", "answer": "Ya, petugas stasiun siap membantu penumpang yang membutuhkan bantuan, terutama bagi penyandang disabilitas atau penumpang yang membutuhkan bantuan khusus"},
    {"question": "", "answer": ""},
    {"question": "", "answer": ""},
    {"question": "", "answer": ""},
    {"question": "", "answer": ""},
    {"question": "", "answer": ""},
    {"question": "", "answer": ""},
    {"question": "", "answer": ""},
    {"question": "", "answer": ""},
    {"question": "apakah boleh membawa senjata tajam atau sajam?", "answer": "membawa senjata tajam dilarang keras karena dapat membahayakan keselamatan penumpang."}
    
]

# Fungsi untuk memproses input dan mencari jawaban
def get_response(user_input):
    questions = [item['question'] for item in faq_data]
    answers = [item['answer'] for item in faq_data]
    
    # Menggunakan TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(questions)
    
    # Hitung kesamaan cosine
    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, vectors)
    
    # Temukan indeks dengan skor tertinggi
    best_match = np.argmax(similarities)
    best_score = similarities[0, best_match]
    
    # Jika skor kesamaan rendah, berikan jawaban default
    if best_score < 0.2:
        return "Maaf, saya tidak memahami pertanyaan Anda. Bisa dijelaskan lebih spesifik?"
    return answers[best_match]

# Fungsi start
async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text("Halo! Saya adalah asisten virtual stasiun KAI Cikarang. Tanyakan apa saja tentang layanan kami.")

# Fungsi untuk menangani pesan
async def handle_message(update: Update, context: CallbackContext) -> None:
    user_input = update.message.text
    response = get_response(user_input)
    await update.message.reply_text(response)

# Fungsi utama
def main():
    # Token bot dari BotFather
    TOKEN = "8089794946:AAGhtBJCkpB3E0mNrLbWafdKVHEVhgIkqAg"
    
    # Menggunakan Application untuk versi terbaru
    application = Application.builder().token(TOKEN).build()

    # Daftarkan handler
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Jalankan bot
    application.run_polling()

if __name__ == '__main__':
    main()
