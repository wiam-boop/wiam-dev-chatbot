# 🤖 Wiam Dev ChatBot

شات بوت ذكي بواجهة ويب أنيقة، من تطوير Wiam، يجيب على أسئلتك بالاعتماد على نموذج لغوي كبير (عبر Groq API)، ويمكنه أيضاً **قراءة ملفاتك (PDF, Word, صور)** والإجابة من محتواها مباشرة عبر تقنية RAG (Retrieval-Augmented Generation).

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-black?logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ المميزات

- 💬 واجهة محادثة أنيقة (سطح مكتب وموبايل) بتصميم مستوحى من ChatGPT
- 📎 رفع ملفات **PDF, Word (.docx), صور (OCR), وملفات نصية** وبناء قاعدة معرفة منها
- 🔍 **بحث ذكي (RAG)**: يسترجع أقرب المعلومات لسؤالك من ملفاتك المرفوعة قبل الإجابة، ويذكر المصدر
- 🧠 ذاكرة محادثة لكل جلسة مستخدم
- 🌐 دعم كامل للغة العربية (واجهة RTL + بحث دلالي متعدد اللغات)
- ⚡ سريع الاستجابة عبر Groq API (بنية LPU المتخصصة في الاستدلال السريع)
- 🔒 مفاتيح API محفوظة بأمان عبر متغيرات بيئة، غير مرفوعة للمستودع

---

## 🧱 التقنيات المستخدمة

| الطبقة | التقنية |
|---|---|
| الخلفية (Backend) | Flask (Python) |
| النموذج اللغوي | Groq API (`openai/gpt-oss-120b`) عبر SDK متوافق مع OpenAI |
| التضمين الدلالي (Embeddings) | `sentence-transformers` (نموذج متعدد اللغات محلي) |
| قاعدة المتجهات (Vector DB) | FAISS |
| استخراج النصوص | `pypdf`، `python-docx`، `pytesseract` (OCR) |
| الواجهة الأمامية | HTML / CSS / JavaScript خام (بدون إطار عمل) |

---

## 📂 هيكل المشروع

```
webchat/
├── app.py                 # نقطة الدخول — سيرفر Flask ومنطق المحادثة/RAG
├── rag.py                 # محرك RAG: استخراج النص، التقسيم، التضمين، البحث
├── templates/
│   └── index.html         # واجهة المستخدم بالكامل
├── requirements.txt        # مكتبات Python المطلوبة
├── .env.example            # نموذج لمتغيرات البيئة (انسخه إلى .env)
├── .gitignore
└── README.md
```

> مجلد `storage/` يُنشأ تلقائياً عند أول رفع ملف، ويحتوي قاعدة المعرفة والملفات المرفوعة. غير مرفوع على GitHub عمداً (بيانات مستخدم، ليست كوداً).

---

## 🚀 التشغيل محلياً

### المتطلبات الأساسية
- [Python 3.10+](https://www.python.org/downloads/)
- مفتاح API مجاني من [Groq Console](https://console.groq.com/keys)
- (اختياري، لدعم قراءة الصور) [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)

### 1) استنساخ المستودع
```bash
git clone https://github.com/USERNAME/wiam-dev-chatbot.git
cd wiam-dev-chatbot
```

### 2) إنشاء بيئة افتراضية وتثبيت المكتبات
```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

> ⏳ أول تشغيل سيُحمّل نموذج التضمين (~470MB) تلقائياً من الإنترنت مرة واحدة فقط، ثم يُخزَّن محلياً.

### 3) إعداد متغيرات البيئة
انسخ الملف النموذجي وعدّل القيم:
```bash
# Windows (PowerShell)
copy .env.example .env
# macOS / Linux
cp .env.example .env
```

افتح `.env` وضع مفتاحك الحقيقي:
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx
FLASK_SECRET_KEY=نص-عشوائي-طويل-غير-متوقع
```

لتوليد `FLASK_SECRET_KEY` عشوائي وآمن:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### 4) تشغيل السيرفر
```bash
python app.py
```

افتح المتصفح على: **http://localhost:5000** 🎉

---

## 🧠 كيف يعمل RAG هنا؟

1. عند رفع ملف → يُستخرج نصه بالكامل (`rag.py`).
2. يُقسَّم النص إلى أجزاء صغيرة متداخلة (chunks) للحفاظ على السياق.
3. يُحوَّل كل جزء إلى متجه رقمي (embedding) عبر نموذج محلي مجاني — لا حاجة لـ API خارجي في هذه الخطوة.
4. تُخزَّن المتجهات في فهرس FAISS على القرص (`storage/kb.index`) — يبقى دائماً حتى بعد إعادة التشغيل.
5. عند كل سؤال، يُحوَّل السؤال أيضاً لمتجه، ويُقارَن بأقرب 4 أجزاء مخزّنة (تشابه جيبي/cosine).
6. الأجزاء الأقرب تُرسل كسياق إضافي للنموذج عبر Groq، مع تعليمات لاستخدامها كمصدر أساسي للإجابة وذكر مصدرها.

> ⚠️ ملاحظة تقنية مهمة: هذا **ليس تدريباً (Fine-tuning)** للنموذج — أوزان النموذج لا تتغير أبداً. هذه تقنية "استرجاع معلومات" (Retrieval) تُغني إجابة النموذج بسياق إضافي في وقت الاستخدام فقط، وهي أخف وأرخص وأسرع تحديثاً من التدريب الحقيقي، وتكفي لمعظم الاستخدامات العملية.

---

## 🌍 النشر على سيرفر (Production)

⚠️ لا تستخدم `python app.py` في الإنتاج (وضع `debug` غير آمن). استخدم `gunicorn`:

```bash
gunicorn -w 2 -b 0.0.0.0:8000 app:app
```

> استخدم عدد workers منخفض (`-w 1` أو `-w 2`) لأن نموذج التضمين يُحمَّل بالذاكرة لكل worker.

### منصات نشر مقترحة
| المنصة | ملاحظات |
|---|---|
| [Railway](https://railway.app) | الأسهل للبدء، يدعم متغيرات البيئة وملفات ثقيلة |
| [Render](https://render.com) | خطة مجانية محدودة، مناسب للتجربة |
| VPS (Hostinger, Contabo, DigitalOcean) | تحكم كامل، يتطلب إعداد يدوي لـ Tesseract و Nginx |

> ❌ **لا يُنصح بـ Vercel** لهذا المشروع تحديداً: بيئته "serverless" لا تدعم تخزيناً دائماً لقاعدة المعرفة، ومكتبات مثل `sentence-transformers` و`faiss` تتجاوز حدود حجم الدوال المسموح بها هناك.

في كل منصة، أضف نفس المتغيرات الموجودة في `.env` كـ Environment Variables من لوحة تحكم المنصة (لا ترفع ملف `.env` نفسه أبداً).

---

## 🔐 ملاحظات أمان

- **لا تضع مفتاح API مباشرة في الكود مطلقاً** — استخدم فقط ملف `.env` (وهو مُستثنى تلقائياً عبر `.gitignore`).
- إذا سبق ورفعت مفتاحاً حقيقياً بالخطأ (حتى في محادثة أو Issue)، **ألغه فوراً** من [console.groq.com/keys](https://console.groq.com/keys) وأنشئ واحداً جديداً.
- `FLASK_SECRET_KEY` يُستخدم لتشفير جلسات المستخدمين — يجب أن يكون طويلاً وعشوائياً وسرياً.

---

## 🗺️ تحسينات مستقبلية ممكنة

- [ ] قاعدة معرفة خاصة بكل مستخدم بدل قاعدة مشتركة
- [ ] دعم توليد الصور عبر API خارجي
- [ ] قاعدة بيانات لحفظ سجل المحادثات بشكل دائم (SQLite/PostgreSQL)
- [ ] دعم تسجيل دخول المستخدمين

---

## 📄 الترخيص

هذا المشروع مرخّص تحت رخصة MIT — استخدمه وعدّله بحرية.

---

## 🙋 بُني بمساعدة

تطوير: Wiam — تم بناء هذا المشروع بالتعاون مع Claude (Anthropic) كمساعد برمجي، كمشروع تعليمي لتعلم أساسيات RAG وبناء تطبيقات ويب بـ Flask.
