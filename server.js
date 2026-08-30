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
// - Order: Claude (if ANTHROPIC_API_KEY is set) → DeepL (if DEEPL_API_KEY is set) →
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

function toDeepLTarget(code) {
  if (code === 'en') return 'EN-US';
  return code.toUpperCase();
}
function toDeepLSource(code) {
  return code.toUpperCase();
}

async function translateWithDeepL(text, fromCode, toCode) {
  if (!DEEPL_API_KEY) throw new Error('no-deepl-key');
  const resp = await fetch(DEEPL_BASE + '/v2/translate', {
    method: 'POST',
    headers: {
      'Authorization': 'DeepL-Auth-Key ' + DEEPL_API_KEY,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      text: [text],
      source_lang: toDeepLSource(fromCode),
      target_lang: toDeepLTarget(toCode),
    }),
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
      const translated = await translateWithDeepL(text, fromCode, toCode);
      return { translated, engine: 'deepl-fallback', claudeError: claudeErr.message };
    } catch (deeplErr) {
      try {
        const translated = await translateWithGoogle(text, fromCode, toCode);
        return { translated, engine: 'google-fallback', claudeError: claudeErr.message, deeplError: deeplErr.message };
      } catch (googleErr) {
        try {
          const translated = await translateWithLibreTranslate(text, fromCode, toCode);
          return { translated, engine: 'libretranslate-fallback', claudeError: claudeErr.message, deeplError: deeplErr.message, googleError: googleErr.message };
        } catch (libreErr) {
          throw new Error('all engines failed — claude: ' + claudeErr.message + ' | deepl: ' + deeplErr.message + ' | google: ' + googleErr.message + ' | libretranslate: ' + libreErr.message);
        }
      }
    }
  }
}

// --- server-side TTS (POST /tts) --------------------------------------------

const EDGE_TTS_TOKEN = '6A5AA1D4EAFF4E9FB37E23D68491D6F4';
// Each language has both a female and male neural voice — picked based on the
// gender detected client-side from whoever spoke (a pitch-based guess, made in the
// browser since only it has access to the raw mic audio; see createGenderDetector()
// in index.html). Falls back to the female voice when gender wasn't detected or
// wasn't sent, exactly matching the previous single-voice behavior.
const EDGE_VOICES = {
  'fa-IR':{female:'fa-IR-DilaraNeural', male:'fa-IR-FaridNeural'},
  'ar-SA':{female:'ar-SA-ZariyahNeural', male:'ar-SA-HamedNeural'},
  'en-US':{female:'en-US-AriaNeural', male:'en-US-GuyNeural'},
  'tr-TR':{female:'tr-TR-EmelNeural', male:'tr-TR-AhmetNeural'},
  'fr-FR':{female:'fr-FR-DeniseNeural', male:'fr-FR-HenriNeural'},
  'de-DE':{female:'de-DE-KatjaNeural', male:'de-DE-ConradNeural'},
  'es-ES':{female:'es-ES-ElviraNeural', male:'es-ES-AlvaroNeural'},
  'it-IT':{female:'it-IT-ElsaNeural', male:'it-IT-DiegoNeural'},
  'ru-RU':{female:'ru-RU-SvetlanaNeural', male:'ru-RU-DmitryNeural'},
  'ja-JP':{female:'ja-JP-NanamiNeural', male:'ja-JP-KeitaNeural'},
  'ko-KR':{female:'ko-KR-SunHiNeural', male:'ko-KR-InJoonNeural'},
  'hi-IN':{female:'hi-IN-SwaraNeural', male:'hi-IN-MadhurNeural'},
  'ur-PK':{female:'ur-PK-UzmaNeural', male:'ur-PK-AsadNeural'},
  'pt-PT':{female:'pt-PT-RaquelNeural', male:'pt-PT-DuarteNeural'},
  'nl-NL':{female:'nl-NL-ColetteNeural', male:'nl-NL-MaartenNeural'},
  'sv-SE':{female:'sv-SE-SofieNeural', male:'sv-SE-MattiasNeural'},
  'pl-PL':{female:'pl-PL-ZofiaNeural', male:'pl-PL-MarekNeural'},
  'uk-UA':{female:'uk-UA-PolinaNeural', male:'uk-UA-OstapNeural'},
  'id-ID':{female:'id-ID-GadisNeural', male:'id-ID-ArdiNeural'},
  'vi-VN':{female:'vi-VN-HoaiMyNeural', male:'vi-VN-NamMinhNeural'},
  'th-TH':{female:'th-TH-PremwadeeNeural', male:'th-TH-NiwatNeural'},
  'he-IL':{female:'he-IL-HilaNeural', male:'he-IL-AvriNeural'},
  'el-GR':{female:'el-GR-AthinaNeural', male:'el-GR-NestorasNeural'},
  'ro-RO':{female:'ro-RO-AlinaNeural', male:'ro-RO-EmilNeural'},
  'bn-BD':{female:'bn-BD-NabanitaNeural', male:'bn-BD-PradeepNeural'},
  'ms-MY':{female:'ms-MY-YasminNeural', male:'ms-MY-OsmanNeural'},
};
function pickEdgeVoice(bcp, gender) {
  const pair = EDGE_VOICES[bcp] || EDGE_VOICES['en-US'];
  return (gender === 'male' && pair.male) ? pair.male : pair.female;
}
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

function synthesizeEdgeTts(text, bcp, gender) {
  return new Promise((resolve, reject) => {
    const voice = pickEdgeVoice(bcp, gender);
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
  const chunks = splitForGoogleTts(text, 100).filter((c) => c.trim().length > 0);
  const buffers = [];
  for (const chunk of chunks) {
    const url = 'https://translate.googleapis.com/translate_tts?ie=UTF-8&q='
      + encodeURIComponent(chunk) + '&tl=' + encodeURIComponent(langCode2) + '&client=tw-ob';
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

  if (req.method === 'POST' && req.url === '/tts') {
    try {
      const body = await readJsonBody(req, 5000);
      const { text, bcp, gender } = body;
      if (!text || !bcp) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'text و bcp لازم است' }));
        return;
      }
      let audio, engine;
      try {
        audio = await synthesizeEdgeTts(String(text), String(bcp), gender);
        engine = 'edge';
      } catch (edgeErr) {
        try {
          // Google's TTS endpoint has no voice/gender selection — one fixed voice
          // per language, so `gender` simply can't be honored on this fallback path.
          audio = await synthesizeGoogleTts(String(text), String(bcp).slice(0, 2));
          engine = 'google-fallback';
        } catch (googleErr) {
          throw new Error('edge: ' + edgeErr.message + ' | google: ' + googleErr.message);
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
    DEEPL_API_KEY ? 'DeepL configured' : 'DeepL NOT configured',
    'Google Translate + LibreTranslate fallbacks always available (no key needed)',
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
      send(target, {
        type: 'chat',
        from: ws.role,
        original: msg.original,
        translated: msg.translated,
        fromPhoto: !!msg.fromPhoto,
        photoPng: msg.photoPng || null,
        gender: msg.gender || null,
      });
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
    DEEPL_API_KEY ? 'DeepL configured' : 'DeepL NOT configured',
  ].join(', ');
  console.log('relay server listening on port ' + PORT + ' — ' + status);
});
