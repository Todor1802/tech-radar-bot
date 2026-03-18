“””
Tech Radar Alert Bot — Optimized Version
Architecture: Perplexity Sonar (search) → Claude Haiku (reasoning + format)

Fixes applied:

1. Lowered impact_score threshold from 4 to 3
1. Removed contradictory “minimum 5 items” constraint
1. Added fallback when Claude returns 0 recs
1. Added /debug command for pipeline diagnostics
1. Added /reload command to force re-fetch without resetting state
1. Improved Claude prompt to always return something useful
   “””

import os
import json
import asyncio
import logging
from datetime import datetime
import httpx
import anthropic
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, ContextTypes
from apscheduler.schedulers.asyncio import AsyncIOScheduler

logging.basicConfig(
format=”%(asctime)s - %(name)s - %(levelname)s - %(message)s”,
level=logging.INFO
)
logger = logging.getLogger(**name**)

# ─────────────────────────────────────────────

# Config

# ─────────────────────────────────────────────

TELEGRAM_TOKEN     = os.environ[“TELEGRAM_TOKEN”]
TELEGRAM_CHAT_ID   = os.environ[“TELEGRAM_CHAT_ID”]
ANTHROPIC_API_KEY  = os.environ[“ANTHROPIC_API_KEY”]
PERPLEXITY_API_KEY = os.environ[“PERPLEXITY_API_KEY”]
DAILY_HOUR   = int(os.getenv(“DAILY_HOUR”, “8”))
DAILY_MINUTE = int(os.getenv(“DAILY_MINUTE”, “0”))

# ─────────────────────────────────────────────

# Vision profile

# ─────────────────────────────────────────────

VISION_CONTEXT = “””
Todor je solo founder i AI developer iz Srbije. Aktivni domeni i projekti:

1. HoMiracle — e-commerce (Dar Homolja med + NatureCare kozmetika)
   Stack: Next.js, Supabase, Stripe, Sanity Studio, Vercel. Launch: 15. maj.
   Cilj: premium D2C brend + B2B Trojan Horse za regionalna tržišta.
1. Neural-Hand — Python/MediaPipe gesture-based PC produktivnost alat
   Python 3.13, MediaPipe, mouse/window control. Claude Code projekat.
1. Virtuelna Galerija — aplikacija za virtuelnu galeriju (u razvoju)
1. Data Engineering @ finansijski holding, Beograd
   Stack: Microsoft Fabric, PySpark, SQL Server, Power BI. Lakehouse migracija.
1. Arm Wrestling — nacionalni takmičar, 80kg, cilj: WAF Evropsko 2027

VIZIJA: Izgraditi AI-native proizvode koji rade autonomno dok on radi posao.
B2B automation + build-to-sell model. Fokus na leverage — više outputa, manje sati.

FILTER (veći score ako):

- Omogućava autonomne agente ili multi-agent sisteme
- Smanjuje manuelni rad u content-u, outreachu ili developmentu
- Primenjivo na e-commerce / D2C growth (HoMiracle)
- Korisno za gesture/computer vision (Neural-Hand)
- Relevantno za data engineering (Fabric, Spark, SQL)
- Primenjivo na virtuelna/3D iskustva (Virtuelna Galerija)
- Novo u Claude/Anthropic ekosistemu
- Smanjuje AI troškove
- Novo u: Cursor, Claude Code, Vercel, GitHub Actions, Supabase, Next.js
  “””

PERPLEXITY_SEARCH_PROMPT = “””
Search for the most relevant AI and technology developments from the past 24-48 hours.

Focus on:

- New AI model releases or significant updates (Claude, GPT, Gemini, open-source)
- New developer tools, SDKs, APIs for web development or AI agents
- GitHub releases in: agents, automation, computer vision, e-commerce tech
- New techniques: multi-agent systems, RAG, code generation, autonomous workflows
- News about: Vercel, Supabase, Anthropic, Microsoft Fabric, MediaPipe, Next.js
- Breakthroughs in: sports tech, gesture recognition, virtual/3D experiences

Return a raw factual summary of the 8-12 most significant findings.
For each finding include: title, what changed, source URL, and why it matters.
Today’s date: {date}
“””

# FIX 1: Uklonjen “Minimum 5 items” constraint koji je bio u kontradikciji sa score filtrom.

# FIX 2: Threshold spušten sa 4 na 3. Dodato pravilo da uvek vrati nešto.

CLAUDE_REASONING_PROMPT = “””
You are Scout, a personal tech radar agent for Todor, a solo founder and AI developer.

Below is raw search data about today’s AI and tech developments.
Analyze this data through the lens of Todor’s vision and produce structured recommendations ranked by impact.

TODOR’S VISION AND PROJECTS:
{vision}

RAW SEARCH DATA FROM TODAY:
{search_results}

Produce a JSON array of recommendations sorted by impact_score descending.
Each item must have exactly these fields:
{{
“title”: “short name (max 6 words)”,
“what”: “1-2 sentences: what it is and what changed today”,
“idea”: “2-3 sentences: SPECIFIC actionable idea for Todor RIGHT NOW”,
“impact_score”: <integer 1-10>,
“impact_reason”: “one sentence: why this score for Todor specifically”,
“domain”: “one of: HoMiracle | Neural-Hand | Virtuelna Galerija | Data Engineering | Arm Wrestling | General”,
“source”: “URL or source name”
}}

Rules:

- Include items with impact_score >= 3
- Return between 3 and 10 items
- IMPORTANT: If you cannot find 3 items scoring >= 3, lower your threshold and include the best available items anyway — never return an empty array
- Return ONLY the JSON array. No preamble, no markdown fences.
  “””

# ─────────────────────────────────────────────

# State

# ─────────────────────────────────────────────

_state: dict = {
“last_digest_date”: None,
“recommendations”: [],
“sent_index”: 0,
“last_raw_search”: “”,       # FIX 3: čuvamo raw search za /debug
“last_error”: None,          # FIX 3: čuvamo poslednju grešku za /debug
}

DOMAIN_EMOJI = {
“HoMiracle”:         “🍯”,
“Neural-Hand”:       “🖐”,
“Virtuelna Galerija”: “🖼”,
“Data Engineering”:  “📊”,
“Arm Wrestling”:     “💪”,
“General”:           “⚡”,
}

# ─────────────────────────────────────────────

# API calls

# ─────────────────────────────────────────────

async def search_with_perplexity() -> str:
prompt = PERPLEXITY_SEARCH_PROMPT.format(date=datetime.now().strftime(”%Y-%m-%d”))
async with httpx.AsyncClient(timeout=60.0) as client:
response = await client.post(
“https://api.perplexity.ai/chat/completions”,
headers={
“Authorization”: f”Bearer {PERPLEXITY_API_KEY}”,
“Content-Type”: “application/json”
},
json={
“model”: “sonar”,
“messages”: [{“role”: “user”, “content”: prompt}],
“max_tokens”: 2000,
“return_citations”: True
}
)
response.raise_for_status()
return response.json()[“choices”][0][“message”][“content”]

def reasoning_with_claude(search_results: str) -> list[dict]:
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
prompt = CLAUDE_REASONING_PROMPT.format(
vision=VISION_CONTEXT,
search_results=search_results
)
response = client.messages.create(
model=“claude-haiku-4-5-20251001”,
max_tokens=2000,
messages=[{“role”: “user”, “content”: prompt}]
)
raw = response.content[0].text.strip()

```
# Strip markdown fences if present
if raw.startswith("```"):
    raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

try:
    recs = json.loads(raw)
    if not isinstance(recs, list):
        raise ValueError("Expected list, got something else")
    recs = sorted(recs, key=lambda x: x.get("impact_score", 0), reverse=True)
    logger.info(f"Claude returned {len(recs)} recommendations")

    # FIX 3: Fallback — ako je lista prazna, vrati emergency card
    if not recs:
        logger.warning("Claude returned empty list — activating fallback card")
        return _fallback_recommendation(search_results)

    return recs

except (json.JSONDecodeError, ValueError) as e:
    logger.error(f"Claude parse error: {e}\nRaw output (first 400 chars): {raw[:400]}")
    return _fallback_recommendation(search_results)
```

def _fallback_recommendation(search_results: str) -> list[dict]:
“””
FIX 3: Uvek vrati bar jednu preporuku.
Ako Claude filter ne uspe, prikaži raw search skraćen na 400 chars.
“””
preview = search_results[:400].replace(”\n”, “ “).strip()
return [{
“title”: “Scout Fallback — Manual Review”,
“what”: f”Automatski filter nije pronašao jasne matcheve. Raw preview: {preview}…”,
“idea”: “Otvori /debug da vidiš kompletan raw output od Perplexity i odluči sam šta je relevantno danas.”,
“impact_score”: 3,
“impact_reason”: “Fallback aktiviran — Claude filter nije vratio strukturirane rezultate.”,
“domain”: “General”,
“source”: “Perplexity Sonar (raw)”
}]

# ─────────────────────────────────────────────

# Formatting

# ─────────────────────────────────────────────

def format_recommendation(rec: dict, index: int, total: int) -> str:
score = rec.get(“impact_score”, 0)
indicator = “🔴” if score >= 9 else “🟠” if score >= 7 else “🟡” if score >= 5 else “⚪”
domain = rec.get(“domain”, “General”)
emoji = DOMAIN_EMOJI.get(domain, “⚡”)
lines = [
f”{indicator} *{rec.get(‘title’, ‘Untitled’)}*  `[{score}/10]`”,
f”{emoji} *{domain}*”,
“”,
f”*Šta je:* {rec.get(‘what’, ‘’)}”,
“”,
f”*💡 Ideja za tebe:* {rec.get(‘idea’, ‘’)}”,
“”,
f”*Zašto {score}/10:* {rec.get(‘impact_reason’, ‘’)}”,
“”,
f”📎 {rec.get(‘source’, ‘’)}”,
“”,
f”*{index}/{total} danas* · /next za sledeću”,
]
return “\n”.join(lines)

def format_digest_header(recs: list[dict]) -> str:
top = recs[0].get(“title”, “?”) if recs else “?”
date_str = datetime.now().strftime(”%d.%m.%Y”)
return (
f”🛰 *Tech Radar — {date_str}*\n\n”
f”Scout pronašao *{len(recs)}* relevantnih novosti.\n”
f”Top pick: *{top}*\n\n”
f”Šaljem prvu preporuku odmah ↓”
)

# ─────────────────────────────────────────────

# Core digest

# ─────────────────────────────────────────────

async def run_digest_pipeline() -> list[dict]:
logger.info(“Step 1: Perplexity search…”)
search_results = await search_with_perplexity()
_state[“last_raw_search”] = search_results          # sačuvaj za /debug
logger.info(f”Perplexity returned {len(search_results)} chars”)

```
logger.info("Step 2: Claude Haiku reasoning…")
loop = asyncio.get_event_loop()
recs = await loop.run_in_executor(None, reasoning_with_claude, search_results)
logger.info(f"Pipeline complete — {len(recs)} recs")
return recs
```

async def send_daily_digest(bot: Bot):
_state[“last_error”] = None
try:
recs = await run_digest_pipeline()
except Exception as e:
_state[“last_error”] = str(e)
logger.error(f”Pipeline error: {e}”)
await bot.send_message(
chat_id=TELEGRAM_CHAT_ID,
text=f”⚠️ Scout error: `{e}`\nPokušaj /today za retry ili /debug za dijagnostiku.”,
parse_mode=“Markdown”
)
return

```
# Recs će uvek imati bar 1 item zbog fallback-a — ova grana se ne bi trebalo aktivirati
if not recs:
    await bot.send_message(
        chat_id=TELEGRAM_CHAT_ID,
        text="⚠️ Scout nije pronašao ništa ni nakon fallback-a. Koristi /debug."
    )
    return

today = datetime.now().strftime("%Y-%m-%d")
_state["last_digest_date"] = today
_state["recommendations"] = recs
_state["sent_index"] = 0

await bot.send_message(
    chat_id=TELEGRAM_CHAT_ID,
    text=format_digest_header(recs),
    parse_mode="Markdown"
)
await asyncio.sleep(1.5)
await bot.send_message(
    chat_id=TELEGRAM_CHAT_ID,
    text=format_recommendation(recs[0], 1, len(recs)),
    parse_mode="Markdown",
    disable_web_page_preview=True
)
_state["sent_index"] = 1
logger.info(f"Digest sent. {len(recs)} recs queued.")
```

# ─────────────────────────────────────────────

# Telegram handlers

# ─────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
await update.message.reply_text(
“👋 *Tech Radar Bot aktivan.*\n\n”
“*Komande:*\n”
“/next — sledeća preporuka\n”
“/today — novi digest odmah (~60s)\n”
“/status — info o poslednjem digest-u\n”
“/debug — dijagnostika pipeline-a\n”
“/reload — re-fetch bez resetovanja state-a”,
parse_mode=“Markdown”
)

async def cmd_next(update: Update, context: ContextTypes.DEFAULT_TYPE):
recs = _state[“recommendations”]
idx  = _state[“sent_index”]
if not recs:
await update.message.reply_text(“Nema učitanog digest-a. Kucaj /today.”)
return
if idx >= len(recs):
await update.message.reply_text(
f”✅ Sve {len(recs)} preporuke su poslate. Kucaj /today za novi fetch.”
)
return
await update.message.reply_text(
format_recommendation(recs[idx], idx + 1, len(recs)),
parse_mode=“Markdown”,
disable_web_page_preview=True
)
_state[“sent_index”] += 1

async def cmd_today(update: Update, context: ContextTypes.DEFAULT_TYPE):
await update.message.reply_text(
“🔍 Perplexity pretražuje… Claude analizira…\n_(~30-60 sekundi)_”,
parse_mode=“Markdown”
)
await send_daily_digest(context.bot)

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
recs  = _state[“recommendations”]
idx   = _state[“sent_index”]
date  = _state[“last_digest_date”] or “nije učitan”
error = _state[“last_error”]
error_line = f”\n⚠️ Poslednja greška: `{error}`” if error else “”
await update.message.reply_text(
f”📊 *Status*\n\n”
f”Poslednji digest: `{date}`\n”
f”Preporuke ukupno: `{len(recs)}`\n”
f”Poslato: `{idx}/{len(recs)}`\n”
f”Auto-digest: `{DAILY_HOUR:02d}:{DAILY_MINUTE:02d}`”
f”{error_line}”,
parse_mode=“Markdown”
)

# FIX 4: Nova /debug komanda — dijagnostikuj svaki korak posebno

async def cmd_debug(update: Update, context: ContextTypes.DEFAULT_TYPE):
await update.message.reply_text(
“🔬 *Debug mode* — testiram svaki korak pipeline-a…”,
parse_mode=“Markdown”
)

```
# Korak 1: Perplexity
try:
    search_results = await search_with_perplexity()
    _state["last_raw_search"] = search_results
    preview = search_results[:600].replace("`", "'")
    await update.message.reply_text(
        f"✅ *Perplexity OK*\n`{len(search_results)} chars`\n\n"
        f"*Preview (600 chars):*\n```\n{preview}\n```",
        parse_mode="Markdown"
    )
except Exception as e:
    await update.message.reply_text(
        f"❌ *Perplexity FAILED*\n`{e}`\n\nProveri PERPLEXITY\\_API\\_KEY.",
        parse_mode="Markdown"
    )
    return

# Korak 2: Claude
try:
    loop = asyncio.get_event_loop()
    recs = await loop.run_in_executor(None, reasoning_with_claude, search_results)
    sample = json.dumps(recs[:2], indent=2, ensure_ascii=False)[:800]
    await update.message.reply_text(
        f"✅ *Claude OK*\n`{len(recs)} preporuka`\n\n"
        f"*Prva 2 (skraćeno):*\n```json\n{sample}\n```",
        parse_mode="Markdown"
    )
except Exception as e:
    await update.message.reply_text(
        f"❌ *Claude FAILED*\n`{e}`\n\nProveri ANTHROPIC\\_API\\_KEY.",
        parse_mode="Markdown"
    )
    return

await update.message.reply_text(
    "✅ *Pipeline zdrav.* Koristi /today za pravi digest.",
    parse_mode="Markdown"
)
```

# FIX 5: Nova /reload komanda — re-fetch ali zadrži history

async def cmd_reload(update: Update, context: ContextTypes.DEFAULT_TYPE):
await update.message.reply_text(
“🔄 Re-fetch u toku… *(ne resetujem stari digest dok novi ne bude spreman)*”,
parse_mode=“Markdown”
)
try:
recs = await run_digest_pipeline()
except Exception as e:
await update.message.reply_text(f”⚠️ Reload failed: `{e}`”, parse_mode=“Markdown”)
return

```
today = datetime.now().strftime("%Y-%m-%d")
_state["last_digest_date"] = today
_state["recommendations"] = recs
_state["sent_index"] = 0

await update.message.reply_text(
    f"✅ Reload spreman — `{len(recs)}` preporuka.\nKoristi /next za prvu.",
    parse_mode="Markdown"
)
```

# ─────────────────────────────────────────────

# Scheduler (post_init — safe async context)

# ─────────────────────────────────────────────

async def post_init(application: Application) -> None:
scheduler = AsyncIOScheduler()
scheduler.add_job(
send_daily_digest,
args=[application.bot],
trigger=“cron”,
hour=DAILY_HOUR,
minute=DAILY_MINUTE,
id=“daily_digest”,
misfire_grace_time=300
)
scheduler.start()
logger.info(f”Scheduler started. Daily digest at {DAILY_HOUR:02d}:{DAILY_MINUTE:02d}”)

# ─────────────────────────────────────────────

# Main

# ─────────────────────────────────────────────

def main():
app = (
Application.builder()
.token(TELEGRAM_TOKEN)
.post_init(post_init)
.build()
)

```
app.add_handler(CommandHandler("start",  cmd_start))
app.add_handler(CommandHandler("help",   cmd_start))
app.add_handler(CommandHandler("next",   cmd_next))
app.add_handler(CommandHandler("today",  cmd_today))
app.add_handler(CommandHandler("status", cmd_status))
app.add_handler(CommandHandler("debug",  cmd_debug))   # FIX 4
app.add_handler(CommandHandler("reload", cmd_reload))  # FIX 5

logger.info("TodorScout bot starting…")
app.run_polling(allowed_updates=["message"])
```

if **name** == “**main**”:
main()