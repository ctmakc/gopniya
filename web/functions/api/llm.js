// Cloudflare Pages Function: POST /api/llm
// Receives ONLY de-identified chat messages from the in-browser agent and forwards them to
// Workers AI. Defence in depth: it re-checks the text and refuses anything that still
// looks like an Indian identifier. Nothing is logged or stored except a hash-keyed answer
// cache (so repeated demo clicks cost nothing) and a per-day call counter.

const MODEL = "@cf/google/gemma-4-26b-a4b-it";
const DAILY_CAP = 60;          // keeps the demo far below the account's free daily allocation
const MAX_CHARS = 9000;
const CACHE_TTL = 60 * 60 * 24 * 14;

const D = [[0,1,2,3,4,5,6,7,8,9],[1,2,3,4,0,6,7,8,9,5],[2,3,4,0,1,7,8,9,5,6],[3,4,0,1,2,8,9,5,6,7],[4,0,1,2,3,9,5,6,7,8],[5,9,8,7,6,0,4,3,2,1],[6,5,9,8,7,1,0,4,3,2],[7,6,5,9,8,2,1,0,4,3],[8,7,6,5,9,3,2,1,0,4],[9,8,7,6,5,4,3,2,1,0]];
const P = [[0,1,2,3,4,5,6,7,8,9],[1,5,7,6,2,8,3,0,9,4],[5,8,0,3,7,9,6,1,4,2],[8,9,1,6,0,4,3,5,2,7],[9,4,5,3,1,2,6,8,7,0],[4,2,8,6,5,7,3,9,0,1],[2,7,9,3,8,0,6,4,1,5],[7,0,4,6,9,1,3,2,5,8]];
function verhoeff(num) {
  let c = 0;
  const r = num.split("").reverse();
  for (let i = 0; i < r.length; i++) c = D[c][P[i % 8][+r[i]]];
  return c === 0;
}

const DEV = { "०":"0","१":"1","२":"2","३":"3","४":"4","५":"5","६":"6","७":"7","८":"8","९":"9" };
function residualIdentifiers(text) {
  const t = text.replace(/[०-९]/g, (d) => DEV[d]).replace(/\[[A-Z_]+_\d+\]/g, " ");
  const hits = [];
  for (const m of t.matchAll(/(?<![\dA-Za-z])[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}(?![\dA-Za-z])/g)) {
    if (verhoeff(m[0].replace(/\D/g, ""))) hits.push("AADHAAR");
  }
  if (/(?<!\d)\d{2}-\d{4}-\d{4}-\d{4}(?!\d)/.test(t)) hits.push("ABHA_NUMBER");
  if (/(?<![A-Za-z0-9])[A-Z]{3}[PCHFATBLJG][A-Z]\d{4}[A-Z](?![A-Za-z0-9])/.test(t)) hits.push("PAN");
  if (/(?<![\dA-Za-z+])(?:\+91[ -]?|0)?[6-9]\d{4}[ -]?\d{5}(?!\d)/.test(t)) hits.push("PHONE");
  if (/[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*/.test(t)) hits.push("EMAIL_OR_ABHA_ADDRESS");
  return [...new Set(hits)];
}

const json = (obj, status = 200) =>
  new Response(JSON.stringify(obj), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" },
  });

async function sha256(s) {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s));
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

export async function onRequestPost({ request, env }) {
  let body;
  try {
    body = await request.json();
  } catch {
    return json({ error: "body must be JSON" }, 400);
  }
  const messages = body && body.messages;
  if (!Array.isArray(messages) || messages.length === 0 || messages.length > 6) {
    return json({ error: "messages: 1 to 6 items" }, 400);
  }
  let total = 0;
  for (const m of messages) {
    if (!m || typeof m.content !== "string" || !["system", "user", "assistant"].includes(m.role)) {
      return json({ error: "each message needs role and content" }, 400);
    }
    total += m.content.length;
  }
  if (total > MAX_CHARS) return json({ error: `note too long for the demo (max ${MAX_CHARS} characters)` }, 413);

  const userText = messages.filter((m) => m.role !== "system").map((m) => m.content).join("\n");
  const residual = residualIdentifiers(userText);
  if (residual.length) {
    return json({ error: `refused: text still looks like it contains ${residual.join(", ")}. This endpoint accepts de-identified text only.` }, 422);
  }

  const key = "c:" + (await sha256(MODEL + JSON.stringify(messages)));
  if (env.DEMO_KV) {
    const hit = await env.DEMO_KV.get(key);
    if (hit) return json({ answer: hit, model: MODEL, cached: true });
    const day = "n:" + new Date().toISOString().slice(0, 10);
    const used = parseInt((await env.DEMO_KV.get(day)) || "0", 10);
    if (used >= DAILY_CAP) {
      return json({ error: `the free demo budget for today (${DAILY_CAP} model calls) is used up; redaction still works, try the model tomorrow or run it locally` }, 429);
    }
    await env.DEMO_KV.put(day, String(used + 1), { expirationTtl: 60 * 60 * 48 });
  }

  let out;
  try {
    out = await env.AI.run(MODEL, {
      messages,
      max_tokens: 700,
      temperature: 0.2,
      chat_template_kwargs: { enable_thinking: false },
    });
  } catch (e) {
    return json({ error: "model call failed: " + String(e && e.message ? e.message : e).slice(0, 200) }, 502);
  }
  const answer = out && (out.response ?? out.choices?.[0]?.message?.content);
  if (typeof answer !== "string") return json({ error: "model returned no text" }, 502);
  if (env.DEMO_KV) await env.DEMO_KV.put(key, answer, { expirationTtl: CACHE_TTL });
  return json({ answer, model: MODEL, cached: false });
}

export function onRequest() {
  return json({ error: "POST only" }, 405);
}
