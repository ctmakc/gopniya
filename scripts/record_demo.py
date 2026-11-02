"""Record the demo video: Playwright drives the live site, captions are an overlay.

    python scripts/record_demo.py https://gopniya.pages.dev/ video/
    ffmpeg -i video/<raw>.webm ... video/gopniya-demo.mp4   (see scripts/make_video.sh)

No voice, no faces: on-screen captions only.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "https://gopniya.pages.dev/"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "video")
W, H = 1440, 900

CAPTION_JS = """
(() => {
  if (document.getElementById('cap')) return;
  const c = document.createElement('div');
  c.id = 'cap';
  c.style.cssText = 'position:fixed;left:50%;bottom:28px;transform:translateX(-50%);max-width:1100px;z-index:9999;' +
    'background:rgba(20,26,46,.92);color:#F3F6FA;font:500 22px/1.35 "Anek Latin",system-ui,sans-serif;padding:14px 22px;' +
    'border-left:4px solid #B8322A;border-radius:2px;transition:opacity .25s;opacity:0;pointer-events:none';
  document.body.appendChild(c);
  const t = document.createElement('div');
  t.id = 'card';
  t.style.cssText = 'position:fixed;inset:0;z-index:10000;background:#1E2A5A;color:#F3F6FA;display:flex;flex-direction:column;' +
    'justify-content:center;padding:0 120px;font-family:"Anek Latin",system-ui,sans-serif;transition:opacity .5s;opacity:0;pointer-events:none';
  document.body.appendChild(t);
})();
"""


async def cap(page, text: str, secs: float) -> None:
    await page.evaluate(
        "(t) => { const c = document.getElementById('cap'); c.textContent = t; c.style.opacity = t ? 1 : 0; }", text
    )
    await page.wait_for_timeout(int(secs * 1000))


async def card(page, html: str, secs: float) -> None:
    await page.evaluate(
        "(h) => { const c = document.getElementById('card'); c.innerHTML = h; c.style.opacity = 1; }", html
    )
    await page.wait_for_timeout(int(secs * 1000))
    await page.evaluate("() => { document.getElementById('card').style.opacity = 0; }")
    await page.wait_for_timeout(600)


async def scroll_to(page, selector: str, offset: int = 0) -> None:
    await page.evaluate(
        "([s, o]) => { const e = document.querySelector(s); window.scrollTo({top: e.getBoundingClientRect().top + scrollY - o, behavior: 'smooth'}); }",
        [selector, offset],
    )
    await page.wait_for_timeout(900)


async def run_agent(page) -> None:
    await page.click("#run")
    await page.wait_for_function("document.getElementById('run').textContent === 'Run agent'", timeout=120000)


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        ctx = await browser.new_context(viewport={"width": W, "height": H}, record_video_dir=str(OUT / "raw"),
                                        record_video_size={"width": W, "height": H})
        page = await ctx.new_page()
        await page.goto(URL)
        await page.evaluate(CAPTION_JS)
        await card(page, "<div style='font:700 96px/0.9 \"Anek Latin\";font-stretch:75%'>Gopniya <span style='font:500 40px \"Anek Devanagari\";opacity:.75'>गोपनीय</span></div>"
                   "<div style='font:500 30px/1.4 \"Anek Latin\";margin-top:28px;max-width:1000px'>An open-source agent that keeps Aadhaar, ABHA, phone numbers, names and addresses "
                   "out of cloud AI when clinics use it for clinical notes.</div>"
                   "<div style='font:400 20px \"Spline Sans Mono\";margin-top:36px;opacity:.7'>Code for a Billion 2026 · Health · MIT licence · all data in this video is synthetic</div>", 7)
        await page.wait_for_selector("#run:not([disabled])", timeout=90000)
        await page.evaluate(CAPTION_JS)
        await cap(page, "The redaction engine is Python running inside this browser tab (Pyodide). The raw note never leaves it.", 6)
        await page.click("text=Discharge summary · English")
        await page.wait_for_timeout(1500)
        await cap(page, "A synthetic discharge summary. Each identifier is found locally and replaced with a typed placeholder.", 6)
        await cap(page, "Aadhaar is accepted only with a valid Verhoeff check digit, or next to an 'Aadhaar' label. ABHA number, ABHA address, phone, UHID, DOB and address are covered too.", 8)
        await cap(page, "Names come from context: honorifics (Smt., Shri, Dr.), field labels, S/o / W/o. Every later mention of the same name is replaced as well.", 8)
        await cap(page, "Right side: this is everything the cloud model will receive. Diagnosis, doses and dates stay readable.", 7)
        await cap(page, "Run the agent: ask the cloud model for a patient summary in simple Hindi.", 3)
        await run_agent(page)
        await cap(page, "Agent steps on the perimeter: detect → guard re-scan → cloud call → verify placeholders → restore.", 7)
        await scroll_to(page, "#restored", 160)
        await cap(page, "Back on the clinic side, the placeholders are swapped for the real values. The cloud model never saw them.", 8)
        await scroll_to(page, ".slip", 20)

        await page.click("text=ANC card · हिंदी")
        await page.wait_for_timeout(1800)
        await cap(page, "Hindi notes work the same way: श्रीमती / श्री / पत्नी cues, Devanagari digits in phone and Aadhaar numbers.", 8)
        await run_agent(page)
        await cap(page, "The answer comes back in Hindi with the real name restored locally.", 6)

        await page.click("text=ASHA visit · mixed")
        await page.wait_for_timeout(1800)
        await cap(page, "An ASHA worker's Hinglish visit note: PAN, e-mail, and names after kinship words like 'beti'.", 7)
        await page.click("#edit")
        await page.wait_for_timeout(500)
        await page.evaluate("() => { const t = document.getElementById('note'); t.scrollTop = t.scrollHeight; t.setSelectionRange(t.value.length, t.value.length); }")
        await cap(page, "Now a line with an identifier no rule was written for: a 16-digit insurance card number.", 4)
        await page.keyboard.type("\nInsurance card 4417 2290 8812 3341 seen at the PHC.", delay=45)
        await page.wait_for_timeout(600)
        await page.click("#edit")
        await page.wait_for_timeout(2200)
        await cap(page, "The guard re-scans the redacted text, finds the leftover number, and escalates before anything is sent.", 8)
        await cap(page, "If anything still looks like an identifier after three rounds, the agent blocks the call. The demo endpoint also refuses such text on its own.", 8)

        await scroll_to(page, "#h-facts", 40)
        await cap(page, "Measured on synthetic notes, including a held-out set written by a separate author who never saw the detection code.", 8)
        await cap(page, "No clinic has used this yet, so there is no field data. The next step is a pilot with a clinic or health programme on real, consented notes.", 8)
        await cap(page, "", 0.3)
        await card(page, "<div style='font:700 64px/1 \"Anek Latin\";font-stretch:80%'>Gopniya</div>"
                   "<div style='font:500 28px/1.5 \"Anek Latin\";margin-top:24px'>Live demo: gopniya.pages.dev<br>Python package + CLI + tests + evaluation, MIT licence<br>"
                   "Fully offline mode with a local model (Ollama)</div>", 6)
        await ctx.close()
        await browser.close()
    raws = sorted((OUT / "raw").glob("*.webm"), key=lambda q: q.stat().st_mtime)
    print(raws[-1])


if __name__ == "__main__":
    asyncio.run(main())
