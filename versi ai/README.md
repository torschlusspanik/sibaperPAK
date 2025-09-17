# SIBAPER ML Backend (FastAPI)

Backend sederhana untuk **prediksi estimasi total pajak** menggunakan model ML.
Model dilatih dari data sintetis yang meniru rumus perhitungan di HTML Anda.

## 1) Persiapan
```bash
cd sibaper_ml_backend
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Latih Model
```bash
python train_model.py
```
File model akan dibuat di `model/sibaper_total_pipeline.joblib`.

## 3) Jalankan API
```bash
uvicorn app:app --reload --port 8000
```
Endpoint:
- `GET /health` → cek status
- `POST /predict` → prediksi total pajak

Contoh body JSON:
```json
{
  "jns_kend": "R2",
  "pkb_pokok": 250000,
  "usia_kend": 3,
  "tunggakan_tahun": 1
}
```

## 4) Integrasi ke HTML
Tambahkan tombol & fungsi JS berikut ke file HTML SIBAPER Anda:

### Tombol (di bawah form)
```html
<button type="button" onclick="predictAI()">🔮 Prediksi AI (ML)</button>
<div id="mlResult" style="margin-top:10px; font-weight:bold;"></div>
```

### Fungsi JS
```javascript
async function predictAI() {
  const jnsKend = document.getElementById('jns_kend').value;
  const pkbPokok = parseInt(document.getElementById('pkb_pokok').value);
  const mBerlaku = document.getElementById('m_berlaku').value;
  if (!jnsKend || !pkbPokok) { alert('Isi jenis kendaraan & PKB!'); return; }

  // Estimasi usia kendaraan dari masa berlaku STNK (opsional, sesuaikan dengan datamu)
  const mStnk = document.getElementById('m_stnk').value;
  const usiaPerkiraan = Math.max(0, new Date(mStnk).getFullYear() - new Date().getFullYear() + 5);

  // Estimasi tunggakan dari masa berlaku pajak
  const yearsLate = (d => Math.max(0, Math.ceil((Date.now() - d.getTime())/(1000*60*60*24*365))))(new Date(mBerlaku));

  document.getElementById('mlResult').innerText = 'Memproses prediksi AI...';

  const res = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({
      jns_kend: jnsKend,
      pkb_pokok: pkbPokok,
      usia_kend: usiaPerkiraan,
      tunggakan_tahun: yearsLate
    })
  });
  const data = await res.json();
  const rupiah = new Intl.NumberFormat('id-ID',{style:'currency', currency:'IDR', minimumFractionDigits:0});
  document.getElementById('mlResult').innerText = 'Prediksi AI: ' + rupiah.format(data.predicted_total);
}
```

> Catatan: Karena model dilatih dengan data sintetis yang meniru rumus saat ini, hasil prediksi akan sangat dekat dengan perhitungan rule-based. Jika Anda punya **data historis nyata**, latih ulang model dengan data tersebut agar AI memberi nilai tambah (mis. variasi denda, diskon, program pemutihan, dll.).

## 5) Deploy
- Bisa di-host di server internal atau cloud (Railway, Render, Vercel + serverless FastAPI via functions, atau Docker).
- Pastikan mengaktifkan CORS hanya untuk domain aplikasi produksi.
