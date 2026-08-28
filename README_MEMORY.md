# Wiam Dev AI - Long-Term Memory Add-on

النسخة تضيف:
- Learned Memory منفصلة عن RAG.
- Semantic search عبر نفس embeddings الموجودة في rag.py.
- Memory threshold.
- RAG fallback.
- Tavily Web fallback.
- حفظ إجابات Web في الذاكرة.
- POST /api/memory/teach للتعليم اليدوي.
- GET /api/memory لعرض الذاكرة.
- DELETE /api/memory/<id> للحذف (Admin فقط).

ضع main.py وknowledge_memory.py وrequirements.txt و.env.example مكان النسخ المقابلة في مشروعك.

Environment:
GEMINI_API_KEY=...
HF_API_KEY=...
GROQ_API_KEY=...
TAVILY_API_KEY=...

تشغيل:
python -m pip install -r requirements.txt
python main.py
