// Relay server for the two-device pairing prototype, PLUS a small server-side
// translation proxy.
//
// WebSocket relay (unchanged from before):
// - Lets one phone create a session (gets a 6-character code)
// - Lets a second phone join that session using the code
// - Relays every message between the two phones instantly (no polling delay)
// - Tracks who's online so both sides can show a live connection status
//
// HTTP translation proxy (POST /translate):
// - Order: Claude (if ANTHROPIC_API_KEY is set) → Cloudflare Workers AI (if
// CF_ACCOUNT_ID + CF_API_TOKEN are set) → DeepL (if DEEPL_API_KEY is set) →
// Google Translate → LibreTranslate. Google Translate was added because, without
// either optional key configured, translations were falling all the way through to
// LibreTranslate — which is free and reliable but noticeably lower quality than
// Google, especially for Persian/Urdu, and was the actual cause of "the translation
// quality is bad" reports. Google Translate's own endpoint here is the same
// reverse-engineered one everyone's translation tools have used for years (not the
// paid/licensed Cloud Translation API) — proxied from this server instead of the
// browser for the same reason as everything else in this section: some visitors'
// own networks block Google directly, but this server (on Render) reaches it fine.
// LibreTranslate stays as the final, fully-open-source, zero-dependency fallback in
// case every other engine is unavailable.
//
// HTTP TTS proxy (POST /tts):
// - Same reasoning, applied to speech: try Microsoft's Edge neural-voice endpoint
// first (best quality when it works), then fall back to Google Translate's TTS
// endpoint if Edge fails. Microsoft has been actively tightening/blocking the
// specific reverse-engineered trick Edge TTS relies on recently (independently
// confirmed — other open-source projects built on the identical trick have been
// hitting the same 403 this month), so it can no longer be assumed reliable on its
// own; Google TTS is the safety net that keeps Persian/Urdu playback working
// regardless.
//
// What this deliberately does NOT do (kept simple on purpose for a prototype):
// - No persistence — if the server restarts, all sessions are gone (fine for a demo,
// not fine for a real product; a real product would use Redis or a database)
// - No auth beyond the session code itself
// - Only two participants per session (a third joiner replaces the guest slot)
// - No caching/rate-limiting on /translate — fine for demo traffic, but add some
// (e.g. a per-IP limiter) before pointing serious traffic at it
//
// Deploy this on Render.com (free tier, no credit card): see DEPLOY.md in this folder.
// ANTHROPIC_API_KEY and DEEPL_API_KEY are both optional — set either (or neither) in
// Render's dashboard. Google Translate and LibreTranslate always work as free,
// no-key baselines.

const http = require('http');
const crypto = require('crypto');
const { WebSocketServer } = require('ws');
// Same 'ws' package also works as a plain WebSocket *client* (used below to reach
// Microsoft's speech endpoint) — WebSocketServer above is only for our own relay.
const WebSocket = require('ws');

const PORT = process.env.PORT || 3000;

const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY || '';
// Haiku is intentionally used here (not Sonnet/Opus): translating one short spoken
// sentence at a time needs to be fast and cheap far more than it needs frontier
// reasoning, and Haiku 4.5 is Anthropic's current model for exactly that kind of task.
const CLAUDE_MODEL = 'claude-haiku-4-5-20251001';

const DEEPL_API_KEY = process.env.DEEPL_API_KEY || '';
// DeepL free-tier keys end with ":fx" and must be called at api-free.deepl.com;
// paid keys use api.deepl.com. Detected automatically so you don't have to configure it.
const DEEPL_BASE = DEEPL_API_KEY.endsWith(':fx')
  ? 'https://api-free.deepl.com'
  : 'https://api.deepl.com';

// Cloudflare Workers AI — called here over Cloudflare's plain REST API, NOT by
// running this code inside an actual Worker. That REST endpoint is reachable from
// any server (this one included, even though it lives on Render, not Cloudflare)
// with just an account ID + an API token — no Worker deployment, no binding, no
// credit card. Free tier: 10,000 "neurons"/day, roughly a few hundred short
// translation calls. Get CF_ACCOUNT_ID from the URL/sidebar of any page in the
// Cloudflare dashboard, and CF_API_TOKEN from My Profile → API Tokens → Create
// Token → grant it the "Workers AI" (Read/Edit) permission.
const CF_ACCOUNT_ID = process.env.CF_ACCOUNT_ID || '';
const CF_API_TOKEN = process.env.CF_API_TOKEN || '';
// m2m100-1.2b is a dedicated many-to-many translation model (not a chat model
// being asked to translate) and supports Persian/Urdu directly.
const CF_TRANSLATE_MODEL = '@cf/meta/m2m100-1.2b';

// sessions: Map<code, { host: ws|null, guest: ws|null, hostLang, guestLang }>
const sessions = new Map();

function makeCode() {
  const chars = 'ABCDEFGHJKMNPQRSTUVWXYZ23456789'; // no ambiguous chars (0/O, 1/I/L)
  let c = '';
  for (let i = 0; i < 6; i++) c += chars[Math.floor(Math.random() * chars.length)];
  return c;
}

function send(ws, obj) {
  if (ws && ws.readyState === ws.OPEN) ws.send(JSON.stringify(obj));
}

function otherSide(session, role) {
  return role === 'host' ? session.guest : session.host;
}

function broadcastPresence(code) {
  const s = sessions.get(code);
  if (!s) return;
  send(s.host, { type: 'presence', partnerOnline: !!(s.guest && s.guest.readyState === s.guest.OPEN) });
  send(s.guest, { type: 'presence', partnerOnline: !!(s.host && s.host.readyState === s.host.OPEN) });
}

// --- translation proxy helpers -------------------------------------------------

const LANG_NAMES = {
  fa: 'Persian (Farsi)', ar: 'Arabic', en: 'English', tr: 'Turkish', fr: 'French',
  de: 'German', es: 'Spanish', it: 'Italian', ru: 'Russian', ja: 'Japanese',
  ko: 'Korean', hi: 'Hindi', ur: 'Urdu', pt: 'Portuguese', nl: 'Dutch',
  sv: 'Swedish', pl: 'Polish', uk: 'Ukrainian', id: 'Indonesian', vi: 'Vietnamese',
  th: 'Thai', he: 'Hebrew', el: 'Greek', ro: 'Romanian', bn: 'Bengali', ms: 'Malay',
};
function langName(code) {
  // 'auto' isn't a real language — it's sent when we're translating text whose
  // source language isn't already known (photographed text, in practice: the
  // person's own selected chat language tells us nothing about what language is
  // printed on a sign/menu/document they photographed). Claude doesn't need — and
  // can't use — a language *code* for that; it just needs to be told to work out
  // the source language itself from the text, which it's genuinely good at.
  if (code === 'auto') return 'the source language (identify it automatically from the text itself — it may be any language)';
  return LANG_NAMES[code] || code;
}

async function translateWithClaude(text, fromCode, toCode) {
  if (!ANTHROPIC_API_KEY) throw new Error('no-anthropic-key');
  const fromName = langName(fromCode);
  const toName = langName(toCode);
  const resp = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'x-api-key': ANTHROPIC_API_KEY,
      'anthropic-version': '2023-06-01',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: CLAUDE_MODEL,
      max_tokens: 500,
      system: 'You are the translation engine inside a live speech-translation app used for real spoken conversation. ' +
        'Translate the user\'s message from ' + fromName + ' to ' + toName + '. ' +
        'Preserve tone and meaning naturally, the way a fluent bilingual interpreter would speak it out loud. ' +
        'Reply with ONLY the translated text — no quotation marks, no notes, no alternate options, no explanations.',
      messages: [{ role: 'user', content: text }],
    }),
  });
  if (!resp.ok) {
    const body = await resp.text().catch(() => '');
    throw new Error('claude-http-' + resp.status + (body ? ': ' + body.slice(0, 200) : ''));
  }
  const data = await resp.json();
  const block = data && data.content && data.content.find((b) => b.type === 'text');
  const translated = block && block.text && block.text.trim();
  if (!translated) throw new Error('claude-bad-response');
  return translated;
}

// Line-by-line translation for the camera-overlay feature (Google Lens style):
// each detected OCR line needs its own translated string, in the same order, so
// the client can redraw each one back on top of the exact spot on the photo where
// the original line was. Still done as ONE Claude call (not one call per line) so
// the model can use the whole passage as context and stay natural/coherent —
// only the *output* is split back into per-line pieces.
async function translateLinesWithClaude(lines, fromCode, toCode) {
  if (!ANTHROPIC_API_KEY) throw new Error('no-anthropic-key');
  const fromName = langName(fromCode);
  const toName = langName(toCode);
  const numbered = lines.map((l, i) => (i + 1) + '. ' + String(l).replace(/\s+/g, ' ').trim()).join('\n');
  const resp = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'x-api-key': ANTHROPIC_API_KEY,
      'anthropic-version': '2023-06-01',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: CLAUDE_MODEL,
      max_tokens: Math.min(4000, Math.max(500, lines.length * 150)),
      system: 'You are the translation engine behind a live camera-overlay translation feature (like Google Lens), ' +
        'translating text that was detected line-by-line on a photographed image, from ' + fromName + ' to ' + toName + '. ' +
        'You will receive a numbered list, one detected line of text per number, in the order the lines appear on the ' +
        'image (top to bottom). Read all the lines together so you understand the full context and meaning, even if a ' +
        'sentence continues across several lines or a word is split across two lines — but you MUST reply with a ' +
        'translation for EVERY numbered line, in the exact same order and exact same count as the input, one entry per ' +
        'original line. Never merge two input lines into one output entry or split one input line into two. If a line ' +
        'is just a stray character, a number, a logo fragment, or otherwise not real translatable text, still return ' +
        'an entry for it (repeat it as-is or return an empty string), so the count always matches. Keep each translated ' +
        'entry reasonably close in length to its original line, since it gets redrawn in that line\'s original space on ' +
        'the photo. Reply with ONLY a raw JSON array of strings — no markdown, no code fence, no commentary — with ' +
        'exactly ' + lines.length + ' items in order.',
      messages: [{ role: 'user', content: numbered }],
    }),
  });
  if (!resp.ok) {
    const body = await resp.text().catch(() => '');
    throw new Error('claude-lines-http-' + resp.status + (body ? ': ' + body.slice(0, 200) : ''));
  }
  const data = await resp.json();
  const block = data && data.content && data.content.find((b) => b.type === 'text');
  let raw = block && block.text && block.text.trim();
  if (!raw) throw new Error('claude-lines-bad-response');
  raw = raw.replace(/^```(?:json)?\s*/i, '').replace(/```\s*$/, '').trim();
  let arr;
  try {
    arr = JSON.parse(raw);
  } catch (e) {
    throw new Error('claude-lines-unparsable');
  }
  if (!Array.isArray(arr) || arr.length !== lines.length) throw new Error('claude-lines-count-mismatch');
  return arr.map((s) => (s == null ? '' : String(s).trim()));
}

// Fallback if the batched Claude call fails or a non-Claude engine is active:
// translate each line on its own through the normal single-text chain (slower,
// loses cross-line context, but still produces a correct per-line result).
async function translateLinesSequentially(lines, fromCode, toCode) {
  const out = [];
  let engine = null;
  for (const line of lines) {
    const trimmed = String(line).trim();
    if (!trimmed) { out.push(''); continue; }
    const r = await translateText(trimmed, fromCode, toCode);
    out.push(r.translated);
    engine = engine || r.engine;
  }
  return { translated: out, engine: (engine || 'unknown') + '-per-line' };
}

async function translateLines(lines, fromCode, toCode) {
  try {
    const translated = await translateLinesWithClaude(lines, fromCode, toCode);
    return { translated, engine: 'claude-lines' };
  } catch (claudeErr) {
    try {
      return await translateLinesSequentially(lines, fromCode, toCode);
    } catch (fallbackErr) {
      throw new Error('line translation failed — claude: ' + claudeErr.message + ' | fallback: ' + fallbackErr.message);
    }
  }
}

async function translateWithWorkersAI(text, fromCode, toCode) {
  if (!CF_ACCOUNT_ID || !CF_API_TOKEN) throw new Error('no-workers-ai-credentials');
  // Cloudflare's translation model needs an actual source language code, not
  // 'auto' — skip straight to the next engine in the chain rather than sending it
  // something it can't use.
  if (fromCode === 'auto') throw new Error('workers-ai-no-auto-detect-support');
  const resp = await fetch(
    'https://api.cloudflare.com/client/v4/accounts/' + CF_ACCOUNT_ID + '/ai/run/' + CF_TRANSLATE_MODEL,
    {
      method: 'POST',
      headers: {
        'Authorization': 'Bearer ' + CF_API_TOKEN,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text, source_lang: fromCode, target_lang: toCode }),
    }
  );
  if (!resp.ok) {
    const body = await resp.text().catch(() => '');
    throw new Error('workers-ai-http-' + resp.status + (body ? ': ' + body.slice(0, 200) : ''));
  }
  const data = await resp.json();
  if (!data || data.success === false) {
    const apiErr = data && data.errors && data.errors[0] && data.errors[0].message;
    throw new Error('workers-ai-api-error' + (apiErr ? ': ' + apiErr : ''));
  }
  const translated = data && data.result && data.result.translated_text;
  if (!translated) throw new Error('workers-ai-bad-response');
  return translated;
}

function toDeepLTarget(code) {
  if (code === 'en') return 'EN-US';
  return code.toUpperCase();
}
function toDeepLSource(code) {
  return code.toUpperCase();
}

async function translateWithDeepL(text, fromCode, toCode) {
  if (!DEEPL_API_KEY) throw new Error('no-deepl-key');
  const body = { text: [text], target_lang: toDeepLTarget(toCode) };
  // DeepL auto-detects the source language when source_lang is left out of the
  // request entirely — there's no 'AUTO' value it accepts for that field.
  if (fromCode !== 'auto') body.source_lang = toDeepLSource(fromCode);
  const resp = await fetch(DEEPL_BASE + '/v2/translate', {
    method: 'POST',
    headers: {
      'Authorization': 'DeepL-Auth-Key ' + DEEPL_API_KEY,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const body = await resp.text().catch(() => '');
    throw new Error('deepl-http-' + resp.status + (body ? ': ' + body.slice(0, 200) : ''));
  }
  const data = await resp.json();
  const translated = data && data.translations && data.translations[0] && data.translations[0].text;
  if (!translated) throw new Error('deepl-bad-response');
  return translated;
}

// Free, no-key Google Translate — the actual "Google Translate" quality people
// expect, via the same reverse-engineered endpoint every free translation tool has
// used for years (translate_a/single, NOT the paid Cloud Translation API). Response
// is a deeply nested JSON array; translated text is the concatenation of
// data[0][i][0] for each sentence chunk Google split the input into.
async function translateWithGoogle(text, fromCode, toCode) {
  const url = 'https://translate.googleapis.com/translate_a/single?client=gtx&sl='
    + encodeURIComponent(fromCode) + '&tl=' + encodeURIComponent(toCode) + '&dt=t&q=' + encodeURIComponent(text);
  const resp = await fetch(url, {
    headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36' },
  });
  if (!resp.ok) throw new Error('google-translate-http-' + resp.status);
  const data = await resp.json();
  const sentences = data && data[0];
  if (!Array.isArray(sentences) || !sentences.length) throw new Error('google-translate-bad-response');
  const translated = sentences.map((s) => (s && s[0]) || '').join('').trim();
  if (!translated) throw new Error('google-translate-empty');
  return translated;
}

async function translateWithLibreTranslate(text, fromCode, toCode) {
  const url = 'https://translate.terraprint.co/translate';
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 10000);
  try {
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ q: text, source: fromCode, target: toCode, format: 'text' }),
      signal: controller.signal,
    });
    if (!resp.ok) {
      const body = await resp.text().catch(() => '');
      throw new Error('libretranslate-http-' + resp.status + (body ? ': ' + body.slice(0, 200) : ''));
    }
    const data = await resp.json();
    if (data && typeof data.translatedText === 'string') {
      return data.translatedText;
    }
    throw new Error('libretranslate-bad-response');
  } finally {
    clearTimeout(timer);
  }
}

async function translateText(text, fromCode, toCode) {
  try {
    const translated = await translateWithClaude(text, fromCode, toCode);
    return { translated, engine: 'claude' };
  } catch (claudeErr) {
    try {
      const translated = await translateWithWorkersAI(text, fromCode, toCode);
      return { translated, engine: 'workers-ai-fallback', claudeError: claudeErr.message };
    } catch (workersAiErr) {
      try {
        const translated = await translateWithDeepL(text, fromCode, toCode);
        return { translated, engine: 'deepl-fallback', claudeError: claudeErr.message, workersAiError: workersAiErr.message };
      } catch (deeplErr) {
        try {
          const translated = await translateWithGoogle(text, fromCode, toCode);
          return { translated, engine: 'google-fallback', claudeError: claudeErr.message, workersAiError: workersAiErr.message, deeplError: deeplErr.message };
        } catch (googleErr) {
          try {
            const translated = await translateWithLibreTranslate(text, fromCode, toCode);
            return { translated, engine: 'libretranslate-fallback', claudeError: claudeErr.message, workersAiError: workersAiErr.message, deeplError: deeplErr.message, googleError: googleErr.message };
          } catch (libreErr) {
            throw new Error('all engines failed — claude: ' + claudeErr.message + ' | workers-ai: ' + workersAiErr.message + ' | deepl: ' + deeplErr.message + ' | google: ' + googleErr.message + ' | libretranslate: ' + libreErr.message);
          }
        }
      }
    }
  }
}

// --- server-side TTS (POST /tts) --------------------------------------------

// ElevenLabs — tried first when configured (best quality of the three engines
// here). Needs just an API key from elevenlabs.io → Profile → API Keys.
// ELEVENLABS_VOICE_ID is optional; defaults to "Rachel", one of ElevenLabs'
// stock voices available on every account. eleven_multilingual_v2 auto-detects
// the spoken language from the text itself, so no per-language voice map is
// needed the way Edge TTS requires one.
const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY || '';
const ELEVENLABS_VOICE_ID = process.env.ELEVENLABS_VOICE_ID || '21m00Tcm4TlvDq8ikWAM';
async function synthesizeElevenLabsTts(text) {
  if (!ELEVENLABS_API_KEY) throw new Error('no-elevenlabs-key');
  const resp = await fetch('https://api.elevenlabs.io/v1/text-to-speech/' + ELEVENLABS_VOICE_ID, {
    method: 'POST',
    headers: {
    'xi-api-key': ELEVENLABS_API_KEY,
      'Content-Type': 'application/json',
      'Accept': 'audio/mpeg',
    },
    body: JSON.stringify({
      text,
      model_id: 'eleven_multilingual_v2',
    }),
  });
  if (!resp.ok) {
    const detail = await resp.text().catch(() => '');
    throw new Error('elevenlabs-http-' + resp.status + (detail ? ': ' + detail.slice(0, 200) : ''));
  }
  return Buffer.from(await resp.arrayBuffer());
}

const EDGE_TTS_TOKEN = '6A5AA1D4EAFF4E9FB37E23D68491D6F4';
const EDGE_VOICES = {
  'fa-IR':'fa-IR-DilaraNeural', 'ar-SA':'ar-SA-ZariyahNeural', 'en-US':'en-US-AriaNeural',
  'tr-TR':'tr-TR-EmelNeural', 'fr-FR':'fr-FR-DeniseNeural', 'de-DE':'de-DE-KatjaNeural',
  'es-ES':'es-ES-ElviraNeural', 'it-IT':'it-IT-ElsaNeural', 'ru-RU':'ru-RU-SvetlanaNeural',
  'ja-JP':'ja-JP-NanamiNeural', 'ko-KR':'ko-KR-SunHiNeural', 'hi-IN':'hi-IN-SwaraNeural',
  'ur-PK':'ur-PK-UzmaNeural', 'pt-PT':'pt-PT-RaquelNeural', 'nl-NL':'nl-NL-ColetteNeural',
  'sv-SE':'sv-SE-SofieNeural', 'pl-PL':'pl-PL-ZofiaNeural', 'uk-UA':'uk-UA-PolinaNeural',
  'id-ID':'id-ID-GadisNeural', 'vi-VN':'vi-VN-HoaiMyNeural', 'th-TH':'th-TH-PremwadeeNeural',
  'he-IL':'he-IL-HilaNeural', 'el-GR':'el-GR-AthinaNeural', 'ro-RO':'ro-RO-AlinaNeural',
  'bn-BD':'bn-BD-NabanitaNeural', 'ms-MY':'ms-MY-YasminNeural',
};
function uuidNoDashes() {
  return 'xxxxxxxxxxxx4xxxyxxxxxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0, v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}
function edgeSecMsGec() {
  const WIN_EPOCH = 11644473600;
  let ticks = Math.floor(Date.now() / 1000) + WIN_EPOCH;
  ticks -= ticks % 300;
  ticks *= 10000000;
  return crypto.createHash('sha256').update(String(ticks) + EDGE_TTS_TOKEN).digest('hex').toUpperCase();
}
function escapeXml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&apos;');
}

function synthesizeEdgeTts(text, bcp) {
  return new Promise((resolve, reject) => {
    const voice = EDGE_VOICES[bcp] || EDGE_VOICES['en-US'];
    const gec = edgeSecMsGec();
    const wsUrl = 'wss://speech.platform.bing.com/consumer/speech/synthesize/readaloud/edge/v1'
      + '?TrustedClientToken=' + EDGE_TTS_TOKEN + '&Sec-MS-GEC=' + gec + '&Sec-MS-GEC-Version=1-131.0.0.0';
    let ws;
    try { ws = new WebSocket(wsUrl); } catch (e) { reject(e); return; }
    const audioParts = [];
    let settled = false;
    const timer = setTimeout(() => finish(reject, new Error('edge-timeout')), Math.max(8000, text.length * 150));
    const finish = (fn, arg) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      try { ws.close(); } catch (_) {}
      fn(arg);
    };
    ws.on('error', (err) => finish(reject, new Error('edge-socket-error' + (err && err.message ? ': ' + err.message : ''))));
    ws.on('open', () => {
      const ts = new Date().toISOString();
      ws.send('X-Timestamp:' + ts + '\r\nContent-Type:application/json; charset=utf-8\r\nPath:speech.config\r\n\r\n'
        + '{"context":{"synthesis":{"audio":{"metadataoptions":{"sentenceBoundaryEnabled":"false","wordBoundaryEnabled":"false"},"outputFormat":"audio-24khz-48kbitrate-mono-mp3"}}}}');
      const ssml = "<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='" + voice.slice(0, 5) + "'>"
        + "<voice name='" + voice + "'><prosody pitch='+0Hz' rate='+0%' volume='+0%'>" + escapeXml(text) + "</prosody></voice></speak>";
      ws.send('X-RequestId:' + uuidNoDashes() + '\r\nContent-Type:application/ssml+xml\r\nX-Timestamp:' + ts + 'Z\r\nPath:ssml\r\n\r\n' + ssml);
    });
    ws.on('message', (data, isBinary) => {
      if (!isBinary) {
        const str = data.toString();
        if (str.indexOf('Path:turn.end') !== -1) {
          if (!audioParts.length) { finish(reject, new Error('edge-no-audio')); return; }
          finish(resolve, Buffer.concat(audioParts));
        }
      } else {
        const headerLen = data.readUInt16BE(0);
        const audioBytes = data.slice(2 + headerLen);
        if (audioBytes.length) audioParts.push(audioBytes);
      }
    });
  });
}

function splitForGoogleTts(text, maxLen) {
  const parts = [];
  let remaining = text.trim();
  while (remaining.length > maxLen) {
    let cut = remaining.lastIndexOf(' ', maxLen);
    if (cut <= 0) cut = maxLen;
    parts.push(remaining.slice(0, cut));
    remaining = remaining.slice(cut).trim();
  }
  if (remaining) parts.push(remaining);
  return parts;
}
async function synthesizeGoogleTts(text, langCode2) {
  const chunks = splitForGoogleTts(text, 180);
  const buffers = [];
  for (const chunk of chunks) {
    const url = 'https://translate.googleapis.com/translate_tts?ie=UTF-8&q='
      + encodeURIComponent(chunk) + '&tl=' + encodeURIComponent(langCode2)  + '&client=tw-ob';
    const resp = await fetch(url, {
      headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36' },
    });
    if (!resp.ok) throw new Error('google-tts-http-' + resp.status);
    buffers.push(Buffer.from(await resp.arrayBuffer()));
  }
  if (!buffers.length) throw new Error('google-tts-empty');
  return Buffer.concat(buffers);
}

function readJsonBody(req, maxBytes) {
  return new Promise((resolve, reject) => {
    let size = 0;
    const chunks = [];
    req.on('data', (chunk) => {
      size += chunk.length;
      if (size > maxBytes) { reject(new Error('payload-too-large')); req.destroy(); return; }
      chunks.push(chunk);
    });
    req.on('end', () => {
      try { resolve(JSON.parse(Buffer.concat(chunks).toString('utf8') || '{}')); }
      catch (e) { reject(new Error('invalid-json')); }
    });
    req.on('error', reject);
  });
}

// --- HTTP server: health check + /translate + /tts proxies ---------------------

const server = http.createServer(async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  if (req.method === 'POST' && req.url === '/translate') {
    try {
      const body = await readJsonBody(req, 20000);
      const { text, source, target } = body;
      if (!text || !source || !target) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'text, source و target لازم است' }));
        return;
      }
      const result = await translateText(String(text), String(source), String(target));
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(result));
    } catch (err) {
      res.writeHead(502, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: err.message || 'ترجمه انجام نشد' }));
    }
    return;
  }

  if (req.method === 'POST' && req.url === '/translate-lines') {
    try {
      const body = await readJsonBody(req, 40000);
      const { lines, source, target } = body;
      if (!Array.isArray(lines) || !lines.length || !source || !target) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'lines (آرایه), source و target لازم است' }));
        return;
      }
      if (lines.length > 60) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'تعداد خط‌ها بیش از حد مجاز است' }));
        return;
      }
      const result = await translateLines(lines.map(String), String(source), String(target));
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(result));
    } catch (err) {
      res.writeHead(502, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: err.message || 'ترجمه خط به خط انجام نشد' }));
    }
    return;
  }

  if (req.method === 'POST' && req.url === '/tts') {
    try {
      const body = await readJsonBody(req, 5000);
      const { text, bcp } = body;
      if (!text || !bcp) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'text و bcp لازم است' }));
        return;
      }
      let audio, engine;
      try {
        audio = await synthesizeElevenLabsTts(String(text));
        engine = 'elevenlabs';
      } catch (elevenErr) {
        try {
          audio = await synthesizeEdgeTts(String(text), String(bcp));
          engine = 'edge';
        } catch (edgeErr) {
          try {
            audio = await synthesizeGoogleTts(String(text), String(bcp).slice(0, 2));
            engine = 'google-fallback';
          } catch (googleErr) {
            throw new Error('elevenlabs: ' + elevenErr.message + ' | edge: ' + edgeErr.message + ' | google: ' + googleErr.message);
          }
        }
      }
      res.writeHead(200, { 'Content-Type': 'audio/mpeg', 'X-TTS-Engine': engine });
      res.end(audio);
    } catch (err) {
      res.writeHead(502, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: err.message || 'ساخت صدا انجام نشد' }));
    }
    return;
  }

  res.writeHead(200, { 'Content-Type': 'text/plain' });
  const status = [
    ANTHROPIC_API_KEY ? 'Claude configured' : 'Claude NOT configured',
    (CF_ACCOUNT_ID && CF_API_TOKEN) ? 'Workers AI configured' : 'Workers AI NOT configured',
    DEEPL_API_KEY ? 'DeepL configured' : 'DeepL NOT configured',
    'Google Translate + LibreTranslate fallbacks always available (no key needed)',
    ELEVENLABS_API_KEY ? 'ElevenLabs TTS configured' : 'ElevenLabs TTS NOT configured',
    'Edge TTS + Google TTS fallback available at POST /tts',
  ].join(', ');
  res.end('translation relay server is running (' + status + ')');
});

const wss = new WebSocketServer({ server });

wss.on('connection', (ws) => {
  ws.role = null;
  ws.code = null;

  ws.on('message', (raw) => {
    let msg;
    try { msg = JSON.parse(raw); } catch (e) { return; }

    if (msg.type === 'create') {
      const code = makeCode();
      sessions.set(code, { host: ws, guest: null, hostLang: msg.lang, guestLang: null });
      ws.role = 'host';
      ws.code = code;
      send(ws, { type: 'created', code });
      return;
    }

    if (msg.type === 'join') {
      const s = sessions.get(msg.code);
      if (!s) { send(ws, { type: 'error', message: 'جلسه‌ای با این کد پیدا نشد' }); return; }
      s.guest = ws;
      s.guestLang = msg.lang;
      ws.role = 'guest';
      ws.code = msg.code;
      send(ws, { type: 'joined', code: msg.code, partnerLang: s.hostLang });
      send(s.host, { type: 'guestJoined', partnerLang: msg.lang });
      broadcastPresence(msg.code);
      return;
    }

    if (msg.type === 'rejoin') {
      // A phone whose socket dropped (screen lock, backgrounding, a brief network
      // blip, the mobile browser suspending the tab) reconnects and asks to resume
      // its old seat. This only works because the session itself is never deleted
      // just for a socket closing — see ws.on('close') below — so it's still here
      // to resume into.
      const s = sessions.get(msg.code);
      if (!s) { send(ws, { type: 'error', message: 'جلسه‌ای با این کد پیدا نشد' }); return; }
      if (msg.role !== 'host' && msg.role !== 'guest') return;
      s[msg.role] = ws;
      ws.role = msg.role;
      ws.code = msg.code;
      const partnerLang = msg.role === 'host' ? s.guestLang : s.hostLang;
      send(ws, { type: 'rejoined', code: msg.code, partnerLang });
      // Deliver anything that arrived while this side's socket was down (screen
      // off, backgrounded, brief network drop) instead of it being lost for good.
      if (s.pending && s.pending.length) {
        const mine = s.pending.filter(p => p.role === msg.role);
        s.pending = s.pending.filter(p => p.role !== msg.role);
        for (const p of mine) send(ws, p.payload);
      }
      broadcastPresence(msg.code);
      return;
    }

    if (msg.type === 'ping') {
      // Keepalive from the client — see the matching comment in index.html
      // (startPairHeartbeat). Replying isn't even the important part; the mere
      // act of receiving/sending traffic is what stops Render's free-tier proxy
      // from treating this socket as idle and silently closing it — which is
      // exactly what used to happen while a host just sat on the "waiting for
      // guest" screen (e.g. copying/sharing the code) without sending anything.
      send(ws, { type: 'pong' });
      return;
    }

    if (msg.type === 'setLang') {
      const s = sessions.get(ws.code);
      if (!s) return;
      if (ws.role === 'host') s.hostLang = msg.lang;
      if (ws.role === 'guest') s.guestLang = msg.lang;
      send(otherSide(s, ws.role), { type: 'partnerLangChanged', lang: msg.lang });
      return;
    }

    if (msg.type === 'chat') {
      // fromPhoto marks a message that came from the camera/OCR flow; when it does,
      // photoPng (a data: URL) carries the already-translated image and
      // original/translated text are omitted entirely — the photo IS the message,
      // not a caption alongside it.
      const s = sessions.get(ws.code);
      if (!s) return;
      const target = otherSide(s, ws.role);
      const payload = {
        type: 'chat',
        from: ws.role,
        original: msg.original,
        translated: msg.translated,
        fromPhoto: !!msg.fromPhoto,
        photoPng: msg.photoPng || null,
      };
      if (target && target.readyState === target.OPEN) {
        send(target, payload);
      } else {
        // The partner's socket is down right now (their screen turned off, the
        // mobile browser suspended their tab, a brief network drop) — don't just
        // drop this on the floor. Queue it on the session so it's delivered the
        // moment they reconnect and rejoin (see 'rejoin' above), same as a real
        // messaging app would. Capped so a session nobody ever comes back to
        // doesn't grow unbounded in memory.
        const targetRole = ws.role === 'host' ? 'guest' : 'host';
        s.pending = s.pending || [];
        s.pending.push({ role: targetRole, payload });
        if (s.pending.length > 200) s.pending.shift();
      }
      return;
    }

    if (msg.type === 'leave') {
      // Explicit, intentional "end call" — the ONLY thing that expires a code.
      // A dropped socket (screen off, app-switch, tab suspended in background)
      // never reaches here; it only hits ws.on('close') below, which does not
      // delete the session. This is what lets the other phone keep chatting (or
      // rejoin) even if you briefly left the page to copy/share the code.
      endSession(ws.code, ws.role);
      return;
    }
  });

  ws.on('close', () => {
    // Just a dropped socket — NOT an intentional end. Mark this side offline so
    // the partner's presence dot updates, but keep the session alive in memory
    // indefinitely so either side can send {type:'rejoin'} and resume. The code
    // only stops working once someone explicitly sends {type:'leave'} (i.e.
    // taps "end call" / "leave session").
    if (!ws.code) return;
    const s = sessions.get(ws.code);
    if (!s) return;
    if (ws.role === 'host' && s.host === ws) s.host = null;
    if (ws.role === 'guest' && s.guest === ws) s.guest = null;
    broadcastPresence(ws.code);
  });
});

function endSession(code, byRole) {
  const s = sessions.get(code);
  if (!s) return;
  const partner = byRole ? otherSide(s, byRole) : null;
  send(partner, { type: 'sessionEnded' });
  sessions.delete(code);
}

server.listen(PORT, () => {
  const status = [
    ANTHROPIC_API_KEY ? 'Claude configured' : 'Claude NOT configured',
    (CF_ACCOUNT_ID && CF_API_TOKEN) ? 'Workers AI configured' : 'Workers AI NOT configured',
    DEEPL_API_KEY ? 'DeepL configured' : 'DeepL NOT configured',
    ELEVENLABS_API_KEY ? 'ElevenLabs TTS configured' : 'ElevenLabs TTS NOT configured',
  ].join(', ');
  console.log('relay server listening on port ' + PORT + ' — ' + status);
});
    
