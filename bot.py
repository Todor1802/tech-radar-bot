"""
Tech Radar Alert Bot — Cost-Optimized Version
Architecture: Perplexity Sonar (search) → Claude Haiku 3.5 (reasoning + format)
Estimated cost: ~$0.50/month for daily digest + manual /next calls

Setup: No admin rights needed. Uses portable Python via winget or uv.
"""

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
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TELEGRAM_TOKEN    = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID  = os.environ["TELEGRAM_CHAT_ID"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
PERPLEXITY_API_KEY = os.environ["PERPLEXITY_API_KEY"]
DAILY_HOUR   = int(os.getenv("DAILY_HOUR", "8"))
DAILY_MINUTE = int(os.getenv("DAILY_MINUTE", "0"))

# ---------------------------------------------------------------------------
# Vision profile — edit as your projects evolve
# ---------------------------------------------------------------------------
VISION_CONTEXT = """
Todor je solo founder i AI developer iz Srbije. Aktivni domeni i projekti:

1. HoMiracle — e-commerce (Dar Homolja med + NatureCare kozmetika)
   Stack: Next.js, Supabase, Stripe, Sanity Studio, Vercel. Launch: 15. maj.
   Cilj: premium D2C brend + B2B Trojan Horse za regionalna tržišta.

2. Neural-Hand — Python/MediaPipe gesture-based PC produktivnost alat
   Python 3.13, MediaPipe, mouse/window control. Claude Code projekat.

3. Virtuelna Galerija — aplikacija za virtuelnu galeriju (u razvoju)

4. Data Engineering @ finansijski holding, Beograd
   Stack: Microsoft Fabric, PySpark, SQL Server, Power BI. Lakehouse migracija.

5. Arm Wrestling — nacionalni takmičar, 80kg, cilj: WAF Evropsko prvenstvo 2027

VIZIJA: Izgraditi AI-native proizvode koji rade autonomno dok on radi svoj posao.
B2B automation + build-to-sell model. Fokus na leverage — više outputa, manje sati.

FILTER (veći score ako):
- Omogućava autonomne agente ili multi-agent sisteme
- Smanjuje manuelni rad u content-u, outreachu ili developmentu
- Primenjivo na e-commerce / D2C growth (HoMiracle)
- Korisno za gesture/computer vision (Neural-Hand)
- Relevantno za data engineering (Fabric, Spark, SQL)
- Primenjivo na virtuelna/3D iskustva (Virtuelna Galerija)
- Novo u Claude/Anthropic ekosistemu
- Smanjuje AI troškove (jeftiniji modeli, efikasniji API)
- Novo u: Cursor, Claude Code, Vercel, GitHub Actions, Supabase, Next.js
"""

# ---------------------------------------------------------------------------
# Step 1: Perplexity — raw search
# ---------------------------------------------------------------------------

PERPLEXITY_SEARCH_PROMPT = """
Search for the most relevant AI and technology developments from the past 24-48 hours.

Focus on:
- New AI model releases or significant updates (Claude, GPT, Gemini, open-source models)
- New developer tools, SDKs, APIs for web development or AI agents
- GitHub releases in: agents, automation, computer vision, e-commerce tech
- New techniques: multi-agent systems, RAG, code generation, autonomous workflows
- News about: Vercel, Supabase, Anthropic, Microsoft Fabric, MediaPipe, Next.js
- Breakthroughs in: sports tech, gesture recognition, virtual/3D experiences, WebGL/Three.js

Return a raw summary of the 8-12 most significant findings with:
- Title of the development
- What changed or was released
- Source URL
- Why it might matter for developers and entrepreneurs

Be factual and specific. Include version numbers, pricing changes, and benchmark results where available.
Today's date: {date}
"""

async def search_with_perplexity() -> str:
    """Call Perplexity Sonar API for raw search results."""
    prompt = PERPLEXITY_SEARCH_PROMPT.format(date=datetime.now().strftime("%Y-%m-%d"))

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "sonar",           # cheapest: $1/1M tokens + $0.005/search
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 2000,
                "return_citations": True
            }
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Step 2: Claude Haiku — reasoning + formatting
# ---------------------------------------------------------------------------

CLAUDE_REASONING_PROMPT = """
You are Scout, a personal tech radar agent for Todor, a solo founder and AI developer.

Below is raw search data about today's AI and tech developments.
Your job: analyze this data through the lens of Todor's vision and projects, 
then produce structured recommendations ranked by their impact on his specific goals.

TODOR'S VISION AND PROJECTS:
{vision}

RAW SEARCH DATA FROM TODAY:
{search_results}

Produce a JSON array of recommendations. Sort by impact_score descending.
Each item must have exactly these fields:
{{
  "title": "short name (max 6 words)",
  "what": "1-2 sentences: what it is and what changed today",
  "idea": "2-3 sentences: SPECIFIC, actionable idea for Todor RIGHT NOW — name the project, name the feature, name the workflow",
  "impact_score": <integer 1-10>,
  "impact_reason": "one sentence: why this score for Todor specifically",
  "domain": "one of: HoMiracle | Neural-Hand | Virtuelna Galerija | Data Engineering | Arm Wrestling | General",
  "source": "URL or source name from the search data"
}}

Rules:
- Only include items with impact_score >= 4
- Minimum 5 items, maximum 10 items
- The "idea" field must be concrete — not "you could use this for..." but "add X to Neural-Hand's gesture pipeline to achieve Y"
- Return ONLY the JSON array. No preamble, no markdown fences, no explanation.
"""

def reasoning_with_claude(search_results: str) -> list[dict]:
    """Send search results to Claude Haiku for vision-aligned analysis."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    prompt = CLAUDE_REASONING_PROMPT.format(
        vision=VISION_CONTEXT,
        search_results=search_results
    )

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",   # cheapest capable model
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()

    # Strip markdown fences if Claude adds them despite instructions
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        recs = json.loads(raw)
        if not isinstance(recs, list):
            raise ValueError("Expected list")
        return sorted(recs, key=lambda x: x.get("impact_score", 0), reverse=True)
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Claude parse error: {e}\nRaw output: {raw[:500]}")
        return []


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

DOMAIN_EMOJI = {
    "HoMiracle": "🍯",
    "Neural-Hand": "🖐",
    "Virtuelna Galerija": "🖼",
    "Data Engineering": "📊",
    "Arm Wrestling": "💪",
    "General": "⚡",
}

def format_recommendation(rec: dict, index: int, total: int) -> str:
    score = rec.get("impact_score", 0)
    indicator = "🔴" if score >= 9 else "🟠" if score >= 7 else "🟡" if score >= 5 else "⚪"
    domain = rec.get("domain", "General")
    emoji = DOMAIN_EMOJI.get(domain, "⚡")

    lines = [
        f"{indicator} *{rec.get('title', 'Untitled')}*  `[{score}/10]`",
        f"{emoji} _{domain}_",
        "",
        f"*Šta je:* {rec.get('what', '')}",
        "",
        f"*💡 Ideja za tebe:* {rec.get('idea', '')}",
        "",
        f"*Zašto {score}/10:* {rec.get('impact_reason', '')}",
        "",
        f"📎 {rec.get('source', '')}",
        "",
        f"_{index}/{total} danas_ · /next za sledeću",
    ]
    return "\n".join(lines)


def format_digest_header(recs: list[dict]) -> str:
    count = len(recs)
    top = recs[0].get("title", "?") if recs else "?"
    date_str = datetime.now().strftime("%d.%m.%Y")
    return (
        f"🛰 *Tech Radar — {date_str}*\n\n"
        f"Scout pronašao *{count}* relevantnih novosti.\n"
        f"Top pick: *{top}*\n\n"
        f"Šaljem prvu preporuku odmah ↓"
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
_state: dict = {
    "last_digest_date": None,
    "recommendations": [],
    "sent_index": 0,
}


# ---------------------------------------------------------------------------
# Core digest flow
# ---------------------------------------------------------------------------

async def run_digest_pipeline() -> list[dict]:
    """Full pipeline: Perplexity search → Claude reasoning → ranked recs."""
    logger.info("Step 1: Perplexity search...")
    search_results = await search_with_perplexity()
    logger.info(f"Search returned {len(search_results)} chars")

    logger.info("Step 2: Claude Haiku reasoning...")
    recs = await asyncio.get_event_loop().run_in_executor(
        None, reasoning_with_claude, search_results
    )
    logger.info(f"Claude produced {len(recs)} recommendations")
    return recs


async def send_daily_digest():
    """Scheduled job: fetch and send daily digest."""
    bot = Bot(token=TELEGRAM_TOKEN)
    today = datetime.now().strftime("%Y-%m-%d")

    try:
        recs = await run_digest_pipeline()
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=f"⚠️ Scout error: `{e}`\nPokušaj /today za retry.",
            parse_mode="Markdown"
        )
        return

    if not recs:
        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text="⚠️ Scout nije pronašao dovoljno relevantnih novosti danas."
        )
        return

    _state["last_digest_date"] = today
    _state["recommendations"] = recs
    _state["sent_index"] = 0

    await bot.send_message(
        chat_id=TELEGRAM_CHAT_ID,
        text=format_digest_header(recs),
        parse_mode="Markdown"
    )
    await asyncio.sleep(1.5)

    msg = format_recommendation(recs[0], 1, len(recs))
    await bot.send_message(
        chat_id=TELEGRAM_CHAT_ID,
        text=msg,
        parse_mode="Markdown",
        disable_web_page_preview=True
    )
    _state["sent_index"] = 1
    logger.info(f"Digest sent. {len(recs)} recs queued.")


# ---------------------------------------------------------------------------
# Telegram handlers
# ---------------------------------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 *Tech Radar Bot aktivan.*\n\n"
        "*Komande:*\n"
        "/next — sledeća preporuka iz danas\n"
        "/today — novi digest odmah (~60s)\n"
        "/status — info o poslednjem digest-u\n"
        "/help — ova poruka\n\n"
        "Automatski digest: svako jutro u podešeno vreme.",
        parse_mode="Markdown"
    )


async def cmd_next(update: Update, context: ContextTypes.DEFAULT_TYPE):
    recs = _state["recommendations"]
    idx = _state["sent_index"]

    if not recs:
        await update.message.reply_text(
            "Nema učitanog digest-a. Kucaj /today da pokrenem odmah."
        )
        return

    if idx >= len(recs):
        await update.message.reply_text(
            f"✅ Sve {len(recs)} preporuke su poslate.\n"
            "Sutra u jutro dolaze nove. Ili kucaj /today za novi fetch."
        )
        return

    rec = recs[idx]
    await update.message.reply_text(
        format_recommendation(rec, idx + 1, len(recs)),
        parse_mode="Markdown",
        disable_web_page_preview=True
    )
    _state["sent_index"] += 1


async def cmd_today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🔍 Perplexity pretražuje web... Claude analizira...\n_(~30-60 sekundi)_",
        parse_mode="Markdown"
    )
    await send_daily_digest()


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    recs = _state["recommendations"]
    idx = _state["sent_index"]
    date = _state["last_digest_date"] or "nije učitan"
    await update.message.reply_text(
        f"📊 *Status*\n\n"
        f"Poslednji digest: `{date}`\n"
        f"Preporuke ukupno: `{len(recs)}`\n"
        f"Poslato: `{idx}/{len(recs)}`\n"
        f"Auto-digest: `{DAILY_HOUR:02d}:{DAILY_MINUTE:02d}`",
        parse_mode="Markdown"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("help",   cmd_start))
    app.add_handler(CommandHandler("next",   cmd_next))
    app.add_handler(CommandHandler("today",  cmd_today))
    app.add_handler(CommandHandler("status", cmd_status))

    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        send_daily_digest,
        trigger="cron",
        hour=DAILY_HOUR,
        minute=DAILY_MINUTE,
        id="daily_digest",
        misfire_grace_time=300   # retry up to 5min late if bot was offline
    )
    scheduler.start()

    logger.info(f"Bot started. Daily digest at {DAILY_HOUR:02d}:{DAILY_MINUTE:02d}")
    app.run_polling(allowed_updates=["message"])


if __name__ == "__main__":
    main()
