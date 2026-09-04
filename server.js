const http = require('http');
const crypto = require('crypto');
const { WebSocketServer } = require('ws');
const WebSocket = require('ws');

const PORT = process.env.PORT || 3000;
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY || '';
const CLAUDE_MODEL = process.env.CLAUDE_MODEL || 'claude-sonnet-4-6';
const DEEPL_API_KEY = process.env.DEEPL_API_KEY || '';
const DEEPL_BASE = DEEPL_API_KEY.endsWith(':fx')
  ? 'https://api-free.deepl.com'
  : 'https://api.deepl.com';

const CF_ACCOUNT_ID = process.env.CF_ACCOUNT_ID || '';
const CF_API_TOKEN = process.env.CF_API_TOKEN || '';
const CF_TRANSLATE_MODEL = '@cf/meta/m2m100-1.2b';
const CF_WHISPER_MODEL = '@cf/openai/whisper-large-v3-turbo';

const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY || '';
const ELEVENLABS_VOICE_ID = process.env.ELEVENLABS_VOICE_ID || '';

const sessions = new Map();

function makeCode() {
  const chars = 'ABCDEFGHJKMNPQRSTUVWXYZ23456789';
  let c = '';
  for (let i = 0; i < 6; i++) {
    c += chars[Math.floor(Math.random() * chars.length)];
  }
  return c;
}

function send(ws, obj) {
  if (ws && ws.readyState === ws.OPEN) {
    ws.send(JSON.stringify(obj));
  }
}

function otherSide(session, role) {
  return role === 'host' ? session.guest : session.host;
}

function broadcastPresence(code) {
  const s = sessions.get(code);
  if (!s) return;

  send(s.host, {
    type: 'presence',
    partnerOnline: !!(s.guest && s.guest.readyState === s.guest.OPEN)
  });

  send(s.guest, {
    type: 'presence',
    partnerOnline: !!(s.host && s.host.readyState === s.host.OPEN)
  });
}

const LANG_NAMES = {
  fa: 'Persian (Farsi)',
  ar: 'Arabic',
  en: 'English',
  tr: 'Turkish',
  fr: 'French',
  de: 'German',
  es: 'Spanish',
  it: 'Italian',
  ru: 'Russian',
  ja: 'Japanese',
  ko: 'Korean',
  hi: 'Hindi',
  ur: 'Urdu',
  pt: 'Portuguese',
  nl: 'Dutch',
  sv: 'Swedish',
  pl: 'Polish',
  uk: 'Ukrainian',
  id: 'Indonesian',
  vi: 'Vietnamese',
  th: 'Thai',
  he: 'Hebrew',
  el: 'Greek',
  ro: 'Romanian',
  bn: 'Bengali',
  ms: 'Malay'
};

function langName(code) {
  if (code === 'auto') {
    return 'the source language (identify it automatically from the text itself — it may be any language)';
  }
  return LANG_NAMES[code] || code;
}

function protectFigureTokens(text) {
  const tokens = [];
  const protectedText = String(text).replace(
    /(?<![\p{L}\p{N}])(?:\d+(?:[.,:/-]\d+)*)/gu,
    (match) => {
      const i = tokens.length;
      tokens.push(match);
      return `__FIGURE_${i}__`;
    }
  );
  return { text: protectedText, tokens };
}

function restoreFigureTokens(text, tokens) {
  let out = String(text);
  tokens.forEach((value, i) => {
    out = out.replace(new RegExp(`__FIGURE_${i}__`, 'g'), value);
  });
  return out;
}

async function translateWithClaude(text, fromCode, toCode, context = []) {
  if (!ANTHROPIC_API_KEY) {
    throw new Error('no-anthropic-key');
  }

  const fromName = langName(fromCode);
  const toName = langName(toCode);

  const safeContext = Array.isArray(context)
    ? context.slice(-6).map((item) => ({
        source: String(item && item.source || '').slice(0, 500),
        translated: String(item && item.translated || '').slice(0, 500),
        sourceLang: String(item && item.sourceLang || '').slice(0, 40),
        targetLang: String(item && item.targetLang || '').slice(0, 40)
      })).filter((item) => item.source || item.translated)
    : [];

  const contextText = safeContext.length
    ? '\n<conversation_context>\n' +
      safeContext.map((item, i) =>
        `[${i + 1}] ${item.sourceLang} → ${item.targetLang}\n` +
        `source: ${item.source}\n` +
        `translation: ${item.translated}`
      ).join('\n') +
      '\n</conversation_context>\n'
    : '';

  const protectedInput = protectFigureTokens(text);

  const userContent =
    contextText +
    '<current_message>\n' +
    protectedInput.text +
    '\n</current_message>';

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 11000);

  let resp;

  try {
    resp = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'x-api-key': ANTHROPIC_API_KEY,
        'anthropic-version': '2023-06-01',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: CLAUDE_MODEL,
        max_tokens: 500,
        system:
          'You are a professional simultaneous interpreter inside a live speech-translation app. ' +
          `Translate exactly one current spoken/typed message from ${fromName} to ${toName}. ` +
          'Use natural, idiomatic, immediately speakable conversation — never stiff word-for-word translation. ' +
          'Preserve meaning, intent, tone, politeness, urgency, certainty, humor, and register. ' +
          'Use context only to resolve references, pronouns, terminology, or ambiguity. ' +
          'Never copy old messages into the answer. Never invent facts. Never summarize. ' +
          'Preserve names, dates, prices, codes, URLs, and figure numbers exactly. ' +
          'Tokens such as __FIGURE_0__ are protected source figures and MUST remain exactly unchanged. ' +
          'Spelled-out number words such as one, two, یک, دو, سه are ordinary words and must be translated. ' +
          'If source is auto, identify the language from the current message itself. ' +
          'Reply with ONLY the translated text — no quotes, explanations, labels, markdown, or alternatives.',
        messages: [
          {
            role: 'user',
            content: userContent
          }
        ]
      })
    });
  } finally {
    clearTimeout(timer);
  }

  if (!resp.ok) {
    const body = await resp.text().catch(() => '');
    throw new Error(
      'claude-http-' +
      resp.status +
      (body ? ': ' + body.slice(0, 250) : '')
    );
  }

  const data = await resp.json();
  const block = data &&
    data.content &&
    data.content.find((b) => b.type === 'text');

  let translated = block &&
    block.text &&
    block.text.trim();

  if (!translated) {
    throw new Error('claude-bad-response');
  }

  translated = restoreFigureTokens(translated, protectedInput.tokens);

  console.log(
    `[translate] Claude OK model=${CLAUDE_MODEL}`
  );

  return translated;
}

async function translateLinesWithClaude(lines, fromCode, toCode) {
  if (!ANTHROPIC_API_KEY) {
    throw new Error('no-anthropic-key');
  }

  const fromName = langName(fromCode);
  const toName = langName(toCode);

  const numbered = lines.map((line, i) =>
    `${i + 1}. ${String(line).replace(/\s+/g, ' ').trim()}`
  ).join('\n');

  const resp = await fetch(
    'https://api.anthropic.com/v1/messages',
    {
      method: 'POST',
      headers: {
        'x-api-key': ANTHROPIC_API_KEY,
        'anthropic-version': '2023-06-01',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: CLAUDE_MODEL,
        max_tokens: Math.min(
          4000,
          Math.max(500, lines.length * 150)
        ),
        system:
          'You are the translation engine behind a live camera-overlay translation feature. ' +
          `Translate the numbered entries from ${fromName} to ${toName}. ` +
          'Return exactly the same number of entries in exactly the same order. ' +
          'Each entry is a real text block from the image and should be translated naturally. ' +
          'Do not merge or split entries. ' +
          'Standalone numbers, dates, prices, codes and symbols must remain unchanged. ' +
          'Spelled-out number words must be translated. ' +
          'If an entry is not meaningful text, repeat it or return an empty string. ' +
          'Reply ONLY with a raw JSON array of strings.',
        messages: [
          {
            role: 'user',
            content: numbered
          }
        ]
      })
    }
  );

  if (!resp.ok) {
    const body = await resp.text().catch(() => '');
    throw new Error(
      'claude-lines-http-' +
      resp.status +
      (body ? ': ' + body.slice(0, 250) : '')
    );
  }

  const data = await resp.json();
  const block = data &&
    data.content &&
    data.content.find((b) => b.type === 'text');

  let raw = block &&
    block.text &&
    block.text.trim();

  if (!raw) {
    throw new Error('claude-lines-bad-response');
  }

  raw = raw
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/```\s*$/, '')
    .trim();

  let arr;

  try {
    arr = JSON.parse(raw);
  } catch (e) {
    throw new Error('claude-lines-unparsable');
  }

  if (!Array.isArray(arr) || arr.length !== lines.length) {
    throw new Error('claude-lines-count-mismatch');
  }

  console.log(
    `[translate-lines] Claude OK model=${CLAUDE_MODEL} count=${lines.length}`
  );

  return arr.map((s) =>
    s == null ? '' : String(s).trim()
  );
        }
  const started = Date.now();
  try {
    const translated = await translateWithClaude(text, fromCode, toCode, context);
    console.log('[translate] Claude OK model=' + CLAUDE_MODEL + ' ms=' + (Date.now() - started));
    return { translated, engine: 'claude' };
  } catch (claudeErr) {
    console.error('[translate] Claude FAILED model=' + CLAUDE_MODEL + ' error=' + claudeErr.message);
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
      const context = Array.isArray(body.context) ? body.context : [];
      console.log('[translate] request ' + String(source) + ' -> ' + String(target) + ' chars=' + String(text).length + ' context=' + context.length);
      const result = await translateText(String(text), String(source), String(target), context);
      res.writeHead(200, {
        'Content-Type': 'application/json',
        'X-Translation-Engine': result.engine || 'unknown',
        'X-Translation-Model': CLAUDE_MODEL,
      });
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
  if (req.method === 'POST' && req.url === '/transcribe') {
    try {
      const body = await readJsonBody(req, 15 * 1024 * 1024);
      const { audio, language } = body;
      if (!audio) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'audio (base64) لازم است' }));
        return;
      }
      if (!CF_ACCOUNT_ID || !CF_API_TOKEN) {
        res.writeHead(503, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'رونویسی صدا تنظیم نشده — CF_ACCOUNT_ID و CF_API_TOKEN را در سرور تنظیم کن' }));
        return;
      }
      const text = await transcribeWithWorkersAI(String(audio), language ? String(language) : undefined);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ text }));
    } catch (err) {
      res.writeHead(502, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: err.message || 'تبدیل صدا به متن انجام نشد' }));
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
    (ANTHROPIC_API_KEY ? 'Claude configured' : 'Claude NOT configured') + ' (' + CLAUDE_MODEL + ')',
    (CF_ACCOUNT_ID && CF_API_TOKEN) ? 'Workers AI configured' : 'Workers AI NOT configured',
    DEEPL_API_KEY ? 'DeepL configured' : 'DeepL NOT configured',
    'Google Translate + LibreTranslate fallbacks always available (no key needed)',
    (CF_ACCOUNT_ID && CF_API_TOKEN) ? 'Whisper transcription (Workers AI) configured' : 'Whisper transcription (Workers AI) NOT configured',
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
      const s = sessions.get(msg.code);
      if (!s) { send(ws, { type: 'error', message: 'جلسه‌ای با این کد پیدا نشد' }); return; }
      if (msg.role !== 'host' && msg.role !== 'guest') return;
      s[msg.role] = ws;
      ws.role = msg.role;
      ws.code = msg.code;
      const partnerLang = msg.role === 'host' ? s.guestLang : s.hostLang;
      send(ws, { type: 'rejoined', code: msg.code, partnerLang });
      if (s.pending && s.pending.length) {
        const mine = s.pending.filter(p => p.role === msg.role);
        s.pending = s.pending.filter(p => p.role !== msg.role);
        for (const p of mine) send(ws, p.payload);
      }
      broadcastPresence(msg.code);
      return;
    }
    if (msg.type === 'ping') {
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
        const targetRole = ws.role === 'host' ? 'guest' : 'host';
        s.pending = s.pending || [];
        s.pending.push({ role: targetRole, payload });
        if (s.pending.length > 200) s.pending.shift();
      }
      return;
    }
    if (msg.type === 'leave') {
      endSession(ws.code, ws.role);
      return;
    }
  });
  ws.on('close', () => {
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
    (ANTHROPIC_API_KEY ? 'Claude configured' : 'Claude NOT configured') + ' (' + CLAUDE_MODEL + ')',
    (CF_ACCOUNT_ID && CF_API_TOKEN) ? 'Workers AI configured' : 'Workers AI NOT configured',
    DEEPL_API_KEY ? 'DeepL configured' : 'DeepL NOT configured',
    ELEVENLABS_API_KEY ? 'ElevenLabs TTS configured' : 'ElevenLabs TTS NOT configured',
  ].join(', ');
  console.log('relay server listening on port ' + PORT + ' — ' + status);
});
