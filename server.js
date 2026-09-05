const http = require('http');
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
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
// Optional: a Cloudflare Workers KV namespace ID, used ONLY to persist the two
// small JSON stores below (model dislikes + user corrections) across restarts
// on hosts with an ephemeral filesystem (Render/Railway free tier, etc). Same
// Cloudflare account you already use for Workers AI — no new signup, no card.
// If left unset, the server falls back to the local JSON file on disk (fine
// for local dev / any host that DOES keep a persistent disk), which is lost on
// restart wherever the disk itself is ephemeral.
const CF_KV_NAMESPACE_ID = process.env.CF_KV_NAMESPACE_ID || '';
function kvConfigured() { return !!(CF_ACCOUNT_ID && CF_API_TOKEN && CF_KV_NAMESPACE_ID); }
async function kvGetJSON(key, fallback) {
  if (!kvConfigured()) return fallback;
  try {
    const resp = await fetch(
      'https://api.cloudflare.com/client/v4/accounts/' + CF_ACCOUNT_ID + '/storage/kv/namespaces/' + CF_KV_NAMESPACE_ID + '/values/' + encodeURIComponent(key),
      { headers: { 'Authorization': 'Bearer ' + CF_API_TOKEN } }
    );
    if (resp.status === 404) return fallback;
    if (!resp.ok) throw new Error('kv-get-http-' + resp.status);
    const text = await resp.text();
    return text ? JSON.parse(text) : fallback;
  } catch (e) {
    console.error('[kv] get(' + key + ') failed, using fallback: ' + e.message);
    return fallback;
  }
}
async function kvPutJSON(key, value) {
  if (!kvConfigured()) return false;
  try {
    const resp = await fetch(
      'https://api.cloudflare.com/client/v4/accounts/' + CF_ACCOUNT_ID + '/storage/kv/namespaces/' + CF_KV_NAMESPACE_ID + '/values/' + encodeURIComponent(key),
      { method: 'PUT', headers: { 'Authorization': 'Bearer ' + CF_API_TOKEN, 'Content-Type': 'application/json' }, body: JSON.stringify(value) }
    );
    if (!resp.ok) throw new Error('kv-put-http-' + resp.status);
    return true;
  } catch (e) {
    console.error('[kv] put(' + key + ') failed: ' + e.message);
    return false;
  }
}
const CF_TRANSLATE_MODEL = '@cf/meta/m2m100-1.2b';
// LLM_MODEL_POOL: no per-language "this model is scientifically best for Turkish"
// data exists, so instead of guessing 26 "optimal" models we keep one pool of
// solid general-purpose multilingual chat models on Workers AI (all free-tier,
// no card needed) and try them IN ORDER for every language. If the first one
// fails outright, or comes back with a clearly bad answer, the next one in the
// list is tried automatically before we ever fall back to the literal M2M-100
// engine. Order = our best-effort default priority, not a proven ranking.
const LLM_MODEL_POOL = [
  '@cf/meta/llama-3.1-8b-instruct',            // primary — confirmed free-tier, solid multilingual instruct model
  '@cf/mistralai/mistral-small-3.1-24b-instruct', // 2nd try — different model family/training data, free-tier
  '@cf/qwen/qwen2.5-coder-32b-instruct',        // 3rd try — another independent fallback, free-tier
];
// NOTE: glm-4.7-flash and kimi-k2.6 (previously in this pool) now require the
// Workers AI PAID plan — they return HTTP 403 "not available on the Workers
// Free plan" on a free account. gemma-4-26b-a4b-it was returning empty
// responses (workers-ai-llm-bad-response). Swap models here freely if
// Cloudflare's free-tier catalog changes again — check
// https://developers.cloudflare.com/workers-ai/models/ for current model IDs
// and which ones are Free vs Paid before adding one back to this pool.
const CF_LLM_TRANSLATE_MODEL = LLM_MODEL_POOL[0]; // kept for status/log text below
const CF_WHISPER_MODEL = '@cf/openai/whisper-large-v3-turbo';
const sessions = new Map();

// --- Per-model dislike tracking (persisted to a small JSON file on disk) ---
// Three things live here:
//  1) userExclusions[userId][targetLang] = [model, ...]
//     -> once a user dislikes a translation, that model stops being used FOR
//        THAT USER for that target language (but keeps serving everyone else).
//  2) modelDislikes[targetLang][model] = [userId, ...] (deduped)
//     -> counts how many DISTINCT users disliked a given model for a given
//        language, so one person spamming dislikes can't trigger a ban alone.
//  3) globalBans[targetLang] = [model, ...]
//     -> once modelDislikes for a model/language reaches the threshold below,
//        that model is removed from the pool for EVERYONE for that language;
//        the chain above simply moves on to the next model in LLM_MODEL_POOL,
//        since there's no honest way to name a specific "better" replacement.
const GLOBAL_MODEL_BAN_THRESHOLD = 100;
const FEEDBACK_DATA_FILE = path.join(__dirname, 'model-feedback-data.json');
const FEEDBACK_KV_KEY = 'model-feedback-data';
let feedbackStore = { userExclusions: {}, modelDislikes: {}, globalBans: {} };
function loadFeedbackStoreFromDisk() {
  try {
    const raw = fs.readFileSync(FEEDBACK_DATA_FILE, 'utf8');
    const parsed = JSON.parse(raw);
    return {
      userExclusions: (parsed && parsed.userExclusions) || {},
      modelDislikes: (parsed && parsed.modelDislikes) || {},
      globalBans: (parsed && parsed.globalBans) || {},
    };
  } catch (e) {
    return null;
  }
}
async function loadFeedbackStore() {
  if (kvConfigured()) {
    const fromKv = await kvGetJSON(FEEDBACK_KV_KEY, null);
    if (fromKv) {
      feedbackStore = {
        userExclusions: fromKv.userExclusions || {},
        modelDislikes: fromKv.modelDislikes || {},
        globalBans: fromKv.globalBans || {},
      };
      console.log('[feedback] loaded from Cloudflare KV');
      return;
    }
    console.log('[feedback] KV configured but empty — starting fresh');
    return;
  }
  const fromDisk = loadFeedbackStoreFromDisk();
  if (fromDisk) {
    feedbackStore = fromDisk;
    console.log('[feedback] loaded model-feedback-data.json from local disk');
  } else {
    console.log('[feedback] no existing data (KV not configured, no local file) — starting fresh; NOTE: this will not survive a restart on hosts with an ephemeral disk — see CF_KV_NAMESPACE_ID');
  }
}
let feedbackSaveTimer = null;
function saveFeedbackStoreSoon() {
  clearTimeout(feedbackSaveTimer);
  feedbackSaveTimer = setTimeout(() => {
    // Always write the local file too (harmless, and still useful for local dev
    // / hosts that do keep a persistent disk); KV is the one that actually
    // survives a redeploy/restart on ephemeral-disk hosts.
    fs.writeFile(FEEDBACK_DATA_FILE, JSON.stringify(feedbackStore), (err) => {
      if (err) console.error('[feedback] failed to save model-feedback-data.json: ' + err.message);
    });
    if (kvConfigured()) kvPutJSON(FEEDBACK_KV_KEY, feedbackStore);
  }, 500);
}
loadFeedbackStore();
function isModelGloballyBanned(model, toCode) {
  const banned = feedbackStore.globalBans[toCode];
  return Array.isArray(banned) && banned.includes(model);
}
function isModelExcludedForUser(userId, model, toCode) {
  if (!userId) return false;
  const perLang = feedbackStore.userExclusions[userId];
  const list = perLang && perLang[toCode];
  return Array.isArray(list) && list.includes(model);
}
// Returns the subset of LLM_MODEL_POOL this particular user/language combo is
// still allowed to use, in priority order.
function availableModelsFor(userId, toCode) {
  return LLM_MODEL_POOL.filter((m) => !isModelGloballyBanned(m, toCode) && !isModelExcludedForUser(userId, m, toCode));
}
// Records one dislike. Returns what happened so the /feedback endpoint can
// report it back (mostly useful for your own debugging/curiosity).
function registerDislike(userId, model, toCode) {
  feedbackStore.userExclusions[userId] = feedbackStore.userExclusions[userId] || {};
  const userLangList = feedbackStore.userExclusions[userId][toCode] = feedbackStore.userExclusions[userId][toCode] || [];
  const wasNewForUser = !userLangList.includes(model);
  if (wasNewForUser) userLangList.push(model);

  feedbackStore.modelDislikes[toCode] = feedbackStore.modelDislikes[toCode] || {};
  const dislikers = feedbackStore.modelDislikes[toCode][model] = feedbackStore.modelDislikes[toCode][model] || [];
  if (!dislikers.includes(userId)) dislikers.push(userId);

  let globallyBanned = isModelGloballyBanned(model, toCode);
  if (!globallyBanned && dislikers.length >= GLOBAL_MODEL_BAN_THRESHOLD) {
    feedbackStore.globalBans[toCode] = feedbackStore.globalBans[toCode] || [];
    feedbackStore.globalBans[toCode].push(model);
    globallyBanned = true;
    console.log('[feedback] GLOBAL BAN: ' + model + ' removed from the pool for "' + toCode + '" after ' + dislikers.length + ' distinct-user dislikes');
  }
  saveFeedbackStoreSoon();
  return {
    ok: true,
    excludedForUser: wasNewForUser,
    distinctDislikesForModel: dislikers.length,
    globallyBanned,
    remainingModelsForUser: availableModelsFor(userId, toCode),
  };
}
// --- Per-user correction memory (persisted to a small JSON file on disk) ---
// correctionsStore[userId][targetLang] = [{ source, bad, fixed, ts }, ...]
// Each entry is a real edit the user made to a translation that came back into
// that target language. We keep only the most recent MAX_CORRECTIONS_PER_PAIR
// per user+language and feed a few of them back into the prompt as "this user
// has preferred this kind of phrasing before" examples. This is NOT fine-tuning
// and makes no promise the model will reuse the exact wording — it's a nudge,
// not a rule.
const MAX_CORRECTIONS_PER_PAIR = 15;
const CORRECTIONS_FED_INTO_PROMPT = 3;
const CORRECTIONS_DATA_FILE = path.join(__dirname, 'user-corrections-data.json');
const CORRECTIONS_KV_KEY = 'user-corrections-data';
let correctionsStore = {};
function loadCorrectionsStoreFromDisk() {
  try {
    const raw = fs.readFileSync(CORRECTIONS_DATA_FILE, 'utf8');
    const parsed = JSON.parse(raw);
    return (parsed && typeof parsed === 'object') ? parsed : null;
  } catch (e) {
    return null;
  }
}
async function loadCorrectionsStore() {
  if (kvConfigured()) {
    const fromKv = await kvGetJSON(CORRECTIONS_KV_KEY, null);
    if (fromKv && typeof fromKv === 'object') {
      correctionsStore = fromKv;
      console.log('[corrections] loaded from Cloudflare KV');
      return;
    }
    console.log('[corrections] KV configured but empty — starting fresh');
    return;
  }
  const fromDisk = loadCorrectionsStoreFromDisk();
  if (fromDisk) {
    correctionsStore = fromDisk;
    console.log('[corrections] loaded user-corrections-data.json from local disk');
  } else {
    console.log('[corrections] no existing data (KV not configured, no local file) — starting fresh; NOTE: this will not survive a restart on hosts with an ephemeral disk — see CF_KV_NAMESPACE_ID');
  }
}
let correctionsSaveTimer = null;
function saveCorrectionsStoreSoon() {
  clearTimeout(correctionsSaveTimer);
  correctionsSaveTimer = setTimeout(() => {
    fs.writeFile(CORRECTIONS_DATA_FILE, JSON.stringify(correctionsStore), (err) => {
      if (err) console.error('[corrections] failed to save user-corrections-data.json: ' + err.message);
    });
    if (kvConfigured()) kvPutJSON(CORRECTIONS_KV_KEY, correctionsStore);
  }, 500);
}
loadCorrectionsStore();
function addCorrection(userId, targetLang, source, bad, fixed) {
  if (!userId || !targetLang) return;
  correctionsStore[userId] = correctionsStore[userId] || {};
  const list = correctionsStore[userId][targetLang] = correctionsStore[userId][targetLang] || [];
  list.push({
    source: String(source || '').slice(0, 300),
    bad: String(bad || '').slice(0, 300),
    fixed: String(fixed || '').slice(0, 300),
    ts: Date.now(),
  });
  while (list.length > MAX_CORRECTIONS_PER_PAIR) list.shift();
  saveCorrectionsStoreSoon();
}
// Returns the most recent few corrections for this user+language, newest last
// (so they read naturally as "recent examples" in the prompt).
function getCorrectionsFor(userId, targetLang) {
  if (!userId || !targetLang) return [];
  const list = correctionsStore[userId] && correctionsStore[userId][targetLang];
  if (!Array.isArray(list) || !list.length) return [];
  return list.slice(-CORRECTIONS_FED_INTO_PROMPT);
}
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
const LANG_NAMES = {
  fa: 'Persian (Farsi)', ar: 'Arabic', en: 'English', tr: 'Turkish', fr: 'French',
  de: 'German', es: 'Spanish', it: 'Italian', ru: 'Russian', ja: 'Japanese',
  ko: 'Korean', hi: 'Hindi', ur: 'Urdu', pt: 'Portuguese', nl: 'Dutch',
  sv: 'Swedish', pl: 'Polish', uk: 'Ukrainian', id: 'Indonesian', vi: 'Vietnamese',
  th: 'Thai', he: 'Hebrew', el: 'Greek', ro: 'Romanian', bn: 'Bengali', ms: 'Malay',
};
function langName(code, dialectHint) {
  let base;
  if (code === 'auto') base = 'the source language (identify it automatically from the text itself — it may be any language)';
  else base = LANG_NAMES[code] || code;
  const hint = String(dialectHint || '').trim().slice(0, 80);
  // Free-text, client-supplied dialect/regional-variety label (e.g. "Dari
  // (Afghanistan)", "Egyptian Arabic", "Tajik Persian"). Not a fixed code
  // table — the client sends whatever label it showed the user, and we just
  // fold it into the language name the model sees, so no per-language table
  // (voices, END_WORD, etc.) needs to exist for every dialect.
  if (hint) return base + ' — specifically the ' + hint + ' variety/dialect; write naturally the way a native speaker of that variety would';
  return base;
}
function buildCorrectionsBlock(corrections) {
  if (!Array.isArray(corrections) || !corrections.length) return '';
  return '\n<user_preferred_phrasing_examples>\n' +
    'This same user previously corrected translations like these. Use them only as a loose ' +
    'style/preference signal (word choice, formality, regional phrasing) for SIMILAR wording — ' +
    'never copy them in verbatim or force them onto unrelated content:\n' +
    corrections.map((c, i) =>
      '[' + (i + 1) + '] source: ' + c.source + '\n' +
      '    machine translation the user disliked: ' + c.bad + '\n' +
      '    user\'s own corrected version: ' + c.fixed
    ).join('\n') +
    '\n</user_preferred_phrasing_examples>\n';
}
async function translateWithClaude(text, fromCode, toCode, context = [], dialectHints = {}, corrections = []) {
  if (!ANTHROPIC_API_KEY) throw new Error('no-anthropic-key');
  const fromName = langName(fromCode, dialectHints && dialectHints.from);
  const toName = langName(toCode, dialectHints && dialectHints.to);
  const safeContext = Array.isArray(context) ? context.slice(-6).map((item) => ({
    source: String(item && item.source || '').slice(0, 500),
    translated: String(item && item.translated || '').slice(0, 500),
    sourceLang: String(item && item.sourceLang || '').slice(0, 40),
    targetLang: String(item && item.targetLang || '').slice(0, 40),
  })).filter((item) => item.source || item.translated) : [];
  const contextText = safeContext.length
    ? '\n<conversation_context>\n' + safeContext.map((item, i) =>
        '[' + (i + 1) + '] ' + item.sourceLang + ' → ' + item.targetLang + '\n' +
        'source: ' + item.source + '\n' +
        'translation: ' + item.translated
      ).join('\n') + '\n</conversation_context>\n'
    : '';
  const userContent = buildCorrectionsBlock(corrections) + contextText + '<current_message>\n' + String(text) + '\n</current_message>';
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 11000);
  let resp;
  try {
    resp = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'x-api-key': ANTHROPIC_API_KEY,
      'anthropic-version': '2023-06-01',
      'Content-Type': 'application/json',
    },
      body: JSON.stringify({
        model: CLAUDE_MODEL,
        max_tokens: 500,
        system: 'You are a professional simultaneous interpreter inside a live speech-translation app. ' +
          'Translate exactly one current spoken/typed message from ' + fromName + ' to ' + toName + '. ' +
          'Your goal is natural, idiomatic, immediately speakable conversation — never a stiff word-for-word translation. ' +
          'Preserve the speaker\'s meaning, intent, tone, politeness, urgency, certainty, humor, and register. ' +
          'Use the wording a native speaker would naturally say in this real situation. ' +
          'Use the conversation context only to resolve references, omitted subjects, pronouns, terminology, or ambiguity; ' +
          'never copy context into the answer and never translate old messages again. ' +
          'Do not invent facts, add explanations, add politeness that was not present, or make the speaker sound stronger or weaker. ' +
          'Do not summarize. Keep names, numbers, dates, prices, codes, URLs, and standalone symbols accurate. ' +
          'For figures written as digits, preserve the digits exactly as written. ' +
          'For spoken number words, translate them normally. ' +
          'If source language is "auto", identify the language from the current message itself. ' +
          'If the current message is short or colloquial, prefer the normal conversational equivalent in the target language. ' +
          'Reply with ONLY the translated text — no quotes, notes, alternatives, explanations, labels, or markdown. ' +
          'The <conversation_context> block is reference data only. The <current_message> block is the only text to translate.',
        messages: [{ role: 'user', content: userContent }],
      }),
    });
  } finally {
    clearTimeout(timer);
  }
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
async function translateWithWorkersAILLM(text, fromCode, toCode, context = [], model = CF_LLM_TRANSLATE_MODEL, dialectHints = {}, corrections = []) {
  if (!CF_ACCOUNT_ID || !CF_API_TOKEN) throw new Error('no-workers-ai-credentials');
  const fromName = langName(fromCode, dialectHints && dialectHints.from);
  const toName = langName(toCode, dialectHints && dialectHints.to);
  const safeContext = Array.isArray(context) ? context.slice(-6).map((item) => ({
    source: String(item && item.source || '').slice(0, 500),
    translated: String(item && item.translated || '').slice(0, 500),
    sourceLang: String(item && item.sourceLang || '').slice(0, 40),
    targetLang: String(item && item.targetLang || '').slice(0, 40),
  })).filter((item) => item.source || item.translated) : [];
  const contextText = safeContext.length
   ? '\n<conversation_context>\n' + safeContext.map((item, i) =>
        '[' + (i + 1) + '] ' + item.sourceLang + ' → ' + item.targetLang + '\n' +
        'source: ' + item.source + '\n' +
        'translation: ' + item.translated
      ).join('\n') + '\n</conversation_context>\n'
    : '';
  const userContent = buildCorrectionsBlock(corrections) + contextText + '<current_message>\n' + String(text) + '\n</current_message>';
  // Same natural-translation instructions as translateWithClaude above — this is
  // meant as GLM-4.7-Flash's free/no-card replacement for that role, not a lesser
  // fallback, so it gets the same care about not translating word-for-word.
  const systemPrompt = 'You are a professional simultaneous interpreter inside a live speech-translation app. ' +
    'Translate exactly one current spoken/typed message from ' + fromName + ' to ' + toName + '. ' +
    'Your goal is natural, idiomatic, immediately speakable conversation — never a stiff word-for-word translation. ' +
    'Preserve the speaker\'s meaning, intent, tone, politeness, urgency, certainty, humor, and register. ' +
    'Use the wording a native speaker would naturally say in this real situation. ' +
    'Use the conversation context only to resolve references, omitted subjects, pronouns, terminology, or ambiguity; ' +
    'never copy context into the answer and never translate old messages again. ' +
    'Do not invent facts, add explanations, add politeness that was not present, or make the speaker sound stronger or weaker. ' +
    'Do not summarize. Keep names, numbers, dates, prices, codes, URLs, and standalone symbols accurate. ' +
    'For figures written as digits, preserve the digits exactly as written. ' +
    'For spoken number words, translate them normally. ' +
    'If source language is "auto", identify the language from the current message itself. ' +
    'If the current message is short or colloquial, prefer the normal conversational equivalent in the target language. ' +
    'Reply with ONLY the translated text — no quotes, notes, alternatives, explanations, labels, or markdown. ' +
    'The <conversation_context> block is reference data only. The <current_message> block is the only text to translate.';
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 11000);
  let resp;
  try {
    resp = await fetch(
      'https://api.cloudflare.com/client/v4/accounts/' + CF_ACCOUNT_ID + '/ai/run/' + model,
      {
        method: 'POST',
        signal: controller.signal,
        headers: {
          'Authorization': 'Bearer ' + CF_API_TOKEN,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages: [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: userContent },
          ],
          max_tokens: 500,
          temperature: 0.3,
        }),
      }
    );
  } finally {
    clearTimeout(timer);
  }
  if (!resp.ok) {
    const body = await resp.text().catch(() => '');
    throw new Error('workers-ai-llm-http-' + resp.status + (body ? ': ' + body.slice(0, 200) : ''));
  }
  const data = await resp.json();
  if (!data || data.success === false) {
    const apiErr = data && data.errors && data.errors[0] && data.errors[0].message;
    throw new Error('workers-ai-llm-api-error' + (apiErr ? ': ' + apiErr : ''));
  }
  const translated = data && data.result && String(data.result.response || '').trim();
  if (!translated) throw new Error('workers-ai-llm-bad-response');
  return translated;
}
// Rough, cheap sanity check — NOT a quality judge. Only meant to catch the two
// clearest failure shapes (an empty/near-empty reply, or a reply that's just
// the source text handed back untouched) so the chain moves to the next model
// instead of quietly returning a broken translation.
function looksSuspiciousTranslation(source, translated, fromCode, toCode) {
  const s = String(source || '').trim();
  const t = String(translated || '').trim();
  if (!t) return true;
  if (fromCode !== toCode && s.length > 8 && s.toLowerCase() === t.toLowerCase()) return true;
  if (s.length > 40 && t.length < s.length * 0.15) return true;
  return false;
}
// Same idea as looksSuspiciousTranslation above, but for a whole photo/OCR batch
// of lines at once: some entries are legitimately expected to come back
// unchanged (brand names, numbers, a stray logo fragment), so a single matching
// line is normal — but if MOST of the substantial lines are byte-identical to
// the source when the languages actually differ, the model almost certainly
// just echoed the whole block back instead of translating it, and the chain
// should move on rather than silently accept it.
function looksSuspiciousLinesTranslation(sourceLines, translatedLines, fromCode, toCode) {
  if (!Array.isArray(translatedLines) || translatedLines.length !== sourceLines.length) return true;
  if (fromCode === toCode) return false;
  let substantial = 0;
  let identical = 0;
  for (let i = 0; i < sourceLines.length; i++) {
    const s = String(sourceLines[i] || '').trim();
    const t = String(translatedLines[i] || '').trim();
    // Only count lines with real letter content and some length — skip bare
    // numbers, single symbols, empty entries, which are fine to pass through.
    if (s.length < 4 || !/[a-zA-Z\u00C0-\u024F\u0600-\u06FF]/.test(s)) continue;
    substantial++;
    if (s.toLowerCase() === t.toLowerCase()) identical++;
  }
  if (substantial < 2) return false; // too little to judge reliably
  return (identical / substantial) >= 0.7;
}
// Tries each still-allowed model in LLM_MODEL_POOL, in order, for this
// user+language. Skips models the user has disliked before, and models that
// hit the global ban threshold for this language. Moves to the next model on
// either a hard error OR a suspicious-looking result.
async function translateWithLLMChain(text, fromCode, toCode, context, userId, dialectHints = {}) {
  const candidates = availableModelsFor(userId, toCode);
  if (!candidates.length) throw new Error('no-llm-models-available-for-' + toCode);
  const corrections = getCorrectionsFor(userId, toCode);
  let lastErr = null;
  for (const model of candidates) {
    try {
      const translated = await translateWithWorkersAILLM(text, fromCode, toCode, context, model, dialectHints, corrections);
      if (looksSuspiciousTranslation(text, translated, fromCode, toCode)) {
        console.warn('[translate] ' + model + ' returned a suspicious result for ' + fromCode + '->' + toCode + ', trying next model');
        lastErr = new Error(model + '-suspicious-response');
        continue;
      }
      return { translated, model };
    } catch (err) {
      console.error('[translate] ' + model + ' FAILED error=' + err.message);
      lastErr = err;
    }
  }
  throw lastErr || new Error('llm-chain-exhausted');
}
async function translateLinesWithModel(lines, fromCode, toCode, model) {
  if (!CF_ACCOUNT_ID || !CF_API_TOKEN) throw new Error('no-workers-ai-credentials');
  const fromName = langName(fromCode);
  const toName = langName(toCode);
  const numbered = lines.map((l, i) => (i + 1) + '. ' + String(l).replace(/\s+/g, ' ').trim()).join('\n');
  // Same photo-OCR translation instructions as translateLinesWithClaude above.
  const systemPrompt = 'You are the translation engine behind a live camera-overlay translation feature (like Google Lens), ' +
    'translating text that was detected on a photographed image, from ' + fromName + ' to ' + toName + '. ' +
    'You will receive a numbered list. Each number is already a merged block of nearby on-image text that has been ' +
    'grouped together because it likely forms one running sentence/paragraph/caption — NOT an arbitrary single OCR ' +
    'line — so treat each numbered entry as a real chunk of prose to translate as a whole, not as an isolated word ' +
    'or fragment to be guessed at out of context. Read all the entries together so terminology, tone, and any ' +
    'pronoun/reference that continues from one entry to the next stay consistent — but you MUST reply with a ' +
    'translation for EVERY numbered entry, in the exact same order and exact same count as the input, one output ' +
    'entry per input entry. Never merge two input entries into one output entry or split one input entry into two. ' +
    'If an entry is just a stray character, a number, a logo fragment, or otherwise not real translatable text, ' +
    'still return an entry for it (repeat it as-is or return an empty string), so the count always matches. ' +
    'Translate each entry the way a skilled bilingual native speaker would naturally phrase it — smooth, idiomatic, ' +
    'full-sentence phrasing in the target language, never a stiff word-for-word rendering, and never a fragment ' +
    'that only makes sense chained to a neighboring entry. Watch for words that are ambiguous in isolation but not ' +
    'in context (e.g. a verb that can mean either "want/like to" or "love", depending on what follows it) — use the ' +
    'surrounding entries to pick the sense that actually fits, rather than defaulting to the most literal one. ' +
    'Never translate or alter numerals written as figures (e.g. "1", "2024", "۱۲", "01", "2/4"), dates, prices, codes, ' +
    'or standalone symbols/logos — copy those through exactly as they appear in the source text. This does NOT apply ' +
    'to spelled-out number words ("one", "two", "یک", "دو", "سه") — those are ordinary vocabulary and must be ' +
    'translated like any other word, into the equivalent number word in the target language. Only translate the ' +
    'surrounding words around a figure, never the figure itself. Each translated entry gets redrawn as one block covering the merged area its source text occupied ' +
    'on the photo, so it does NOT need to match the original\'s length line-for-line — prioritize a natural, correctly ' +
    'worded sentence over matching length. Reply with ONLY a raw JSON array of strings — no markdown, no code ' +
    'fence, no commentary — with exactly ' + lines.length + ' items in order.';
  const resp = await fetch(
    'https://api.cloudflare.com/client/v4/accounts/' + CF_ACCOUNT_ID + '/ai/run/' + model,
    {
      method: 'POST',
      headers: {
        'Authorization': 'Bearer ' + CF_API_TOKEN,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: numbered },
        ],
        max_tokens: Math.min(4000, Math.max(500, lines.length * 150)),
        temperature: 0.3,
      }),
    }
  );
  if (!resp.ok) {
    const body = await resp.text().catch(() => '');
    throw new Error('workers-ai-llm-lines-http-' + resp.status + (body ? ': ' + body.slice(0, 200) : ''));
  }
  const data = await resp.json();
  if (!data || data.success === false) {
    const apiErr = data && data.errors && data.errors[0] && data.errors[0].message;
    throw new Error('workers-ai-llm-lines-api-error' + (apiErr ? ': ' + apiErr : ''));
  }
  let raw = data && data.result && String(data.result.response || '').trim();
  if (!raw) throw new Error('workers-ai-llm-lines-bad-response');
  raw = raw.replace(/^```(?:json)?\s*/i, '').replace(/```\s*$/, '').trim();
  let arr;
  try {
    arr = JSON.parse(raw);
  } catch (e) {
    throw new Error('workers-ai-llm-lines-unparsable');
  }
  if (!Array.isArray(arr) || arr.length !== lines.length) throw new Error('workers-ai-llm-lines-count-mismatch');
  return arr.map((s) => (s == null ? '' : String(s).trim()));
}
// Same model-pool chain idea as translateWithLLMChain above, applied to the
// photo-OCR batch endpoint.
async function translateLinesWithWorkersAILLM(lines, fromCode, toCode, userId) {
  const candidates = availableModelsFor(userId, toCode);
  if (!candidates.length) throw new Error('no-llm-models-available-for-' + toCode);
  let lastErr = null;
  for (const model of candidates) {
    try {
      const translated = await translateLinesWithModel(lines, fromCode, toCode, model);
      if (looksSuspiciousLinesTranslation(lines, translated, fromCode, toCode)) {
        console.warn('[translate-lines] ' + model + ' returned an echoed/untranslated block for ' + fromCode + '->' + toCode + ', trying next model');
        lastErr = new Error(model + '-suspicious-lines-response');
        continue;
      }
      return translated;
    } catch (err) {
      console.error('[translate-lines] ' + model + ' FAILED error=' + err.message);
      lastErr = err;
    }
  }
  throw lastErr || new Error('llm-lines-chain-exhausted');
}
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
        'translating text that was detected on a photographed image, from ' + fromName + ' to ' + toName + '. ' +
        'You will receive a numbered list. Each number is already a merged block of nearby on-image text that has been ' +
        'grouped together because it likely forms one running sentence/paragraph/caption — NOT an arbitrary single OCR ' +
        'line — so treat each numbered entry as a real chunk of prose to translate as a whole, not as an isolated word ' +
        'or fragment to be guessed at out of context. Read all the entries together so terminology, tone, and any ' +
        'pronoun/reference that continues from one entry to the next stay consistent — but you MUST reply with a ' +
        'translation for EVERY numbered entry, in the exact same order and exact same count as the input, one output ' +
        'entry per input entry. Never merge two input entries into one output entry or split one input entry into two. ' +
        'If an entry is just a stray character, a number, a logo fragment, or otherwise not real translatable text, ' +
        'still return an entry for it (repeat it as-is or return an empty string), so the count always matches. ' +
        'Translate each entry the way a skilled bilingual native speaker would naturally phrase it — smooth, idiomatic, ' +
        'full-sentence phrasing in the target language, never a stiff word-for-word rendering, and never a fragment ' +
        'that only makes sense chained to a neighboring entry. Watch for words that are ambiguous in isolation but not ' +
        'in context (e.g. a verb that can mean either "want/like to" or "love", depending on what follows it) — use the ' +
        'surrounding entries to pick the sense that actually fits, rather than defaulting to the most literal one. ' +
        'Never translate or alter numerals written as figures (e.g. "1", "2024", "۱۲", "01", "2/4"), dates, prices, codes, ' +
        'or standalone symbols/logos — copy those through exactly as they appear in the source text. This does NOT apply ' +
        'to spelled-out number words ("one", "two", "یک", "دو", "سه") — those are ordinary vocabulary and must be ' +
        'translated like any other word, into the equivalent number word in the target language. Only translate the ' +
        'surrounding words around a figure, never the figure itself. Each translated entry gets redrawn as one block covering the merged area its source text occupied ' +
        'on the photo, so it does NOT need to match the original\'s length line-for-line — prioritize a natural, correctly ' +
        'worded sentence over matching length. Reply with ONLY a raw JSON array of strings — no markdown, no code ' +
        'fence, no commentary — with exactly ' + lines.length + ' items in order.',
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
async function translateLinesSequentially(lines, fromCode, toCode, userId) {
  const out = [];
  let engine = null;
  for (const line of lines) {
    const trimmed = String(line).trim();
    if (!trimmed) { out.push(''); continue; }
    const r = await translateText(trimmed, fromCode, toCode, [], userId);
    out.push(r.translated);
    engine = engine || r.engine;
  }
  return { translated: out, engine: (engine || 'unknown') + '-per-line' };
                   }
async function translateLines(lines, fromCode, toCode, userId) {
  try {
    const translated = await translateLinesWithWorkersAILLM(lines, fromCode, toCode, userId);
    return { translated, engine: 'workers-ai-llm-lines' };
  } catch (llmErr) {
    try {
      const translated = await translateLinesWithClaude(lines, fromCode, toCode);
      if (looksSuspiciousLinesTranslation(lines, translated, fromCode, toCode)) {
        throw new Error('claude-lines-echoed-untranslated-block');
      }
      return { translated, engine: 'claude-lines', workersAiLlmError: llmErr.message };
    } catch (claudeErr) {
      try {
        return await translateLinesSequentially(lines, fromCode, toCode, userId);
      } catch (fallbackErr) {
        throw new Error('line translation failed — glm: ' + llmErr.message + ' | claude: ' + claudeErr.message + ' | fallback: ' + fallbackErr.message);
      }
    }
  }
}
async function translateWithWorkersAI(text, fromCode, toCode) {
  if (!CF_ACCOUNT_ID || !CF_API_TOKEN) throw new Error('no-workers-ai-credentials');
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
async function transcribeWithWorkersAI(base64Audio, languageHint) {
  if (!CF_ACCOUNT_ID || !CF_API_TOKEN) throw new Error('no-workers-ai-credentials');
  const payload = { audio: base64Audio };
  if (languageHint) payload.language = languageHint;
  const resp = await fetch(
    'https://api.cloudflare.com/client/v4/accounts/' + CF_ACCOUNT_ID + '/ai/run/' + CF_WHISPER_MODEL,
    {
      method: 'POST',
      headers: {
        'Authorization': 'Bearer ' + CF_API_TOKEN,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    }
  );
  if (!resp.ok) {
    const body = await resp.text().catch(() => '');
    throw new Error('workers-ai-whisper-http-' + resp.status + (body ? ': ' + body.slice(0, 200) : ''));
  }
  const data = await resp.json();
  if (!data || data.success === false) {
    const apiErr = data && data.errors && data.errors[0] && data.errors[0].message;
    throw new Error('workers-ai-whisper-api-error' + (apiErr ? ': ' + apiErr : ''));
  }
  const text = data && data.result && data.result.text;
  if (typeof text !== 'string') throw new Error('workers-ai-whisper-bad-response');
  return text.trim();
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
async function translateText(text, fromCode, toCode, context = [], userId = null, dialectHints = {}) {
  const started = Date.now();
  try {
    const { translated, model } = await translateWithLLMChain(text, fromCode, toCode, context, userId, dialectHints);
    console.log('[translate] ' + model + ' OK ms=' + (Date.now() - started));
    return { translated, engine: 'workers-ai-llm', model };
  } catch (llmErr) {
    console.error('[translate] all Workers AI LLM models FAILED error=' + llmErr.message);
    try {
      const translated = await translateWithClaude(text, fromCode, toCode, context, dialectHints, getCorrectionsFor(userId, toCode));
      console.log('[translate] Claude OK model=' + CLAUDE_MODEL + ' ms=' + (Date.now() - started));
      return { translated, engine: 'claude', model: CLAUDE_MODEL, workersAiLlmError: llmErr.message };
    } catch (claudeErr) {
      try {
        const translated = await translateWithWorkersAI(text, fromCode, toCode);
        return { translated, engine: 'workers-ai-fallback', model: CF_TRANSLATE_MODEL, workersAiLlmError: llmErr.message, claudeError: claudeErr.message };
      } catch (workersAiErr) {
        try {
          const translated = await translateWithDeepL(text, fromCode, toCode);
          return { translated, engine: 'deepl-fallback', model: 'deepl', workersAiLlmError: llmErr.message, claudeError: claudeErr.message, workersAiError: workersAiErr.message };
        } catch (deeplErr) {
          try {
            const translated = await translateWithGoogle(text, fromCode, toCode);
            return { translated, engine: 'google-fallback', model: 'google-translate', workersAiLlmError: llmErr.message, claudeError: claudeErr.message, workersAiError: workersAiErr.message, deeplError: deeplErr.message };
          } catch (googleErr) {
            try {
              const translated = await translateWithLibreTranslate(text, fromCode, toCode);
              return { translated, engine: 'libretranslate-fallback', model: 'libretranslate', workersAiLlmError: llmErr.message, claudeError: claudeErr.message, workersAiError: workersAiErr.message, deeplError: deeplErr.message, googleError: googleErr.message };
            } catch (libreErr) {
              throw new Error('all engines failed — glm-pool: ' + llmErr.message + ' | claude: ' + claudeErr.message + ' | workers-ai: ' + workersAiErr.message + ' | deepl: ' + deeplErr.message + ' | google: ' + googleErr.message + ' | libretranslate: ' + libreErr.message);
            }
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
const EDGE_CLIENT_VERSION = '1-143.0.3650.75';
function synthesizeEdgeTts(text, bcp, gender) {
  return new Promise((resolve, reject) => {
    const voice = pickEdgeVoice(bcp, gender);
    const gec = edgeSecMsGec();
    const wsUrl = 'wss://speech.platform.bing.com/consumer/speech/synthesize/readaloud/edge/v1'
      + '?TrustedClientToken=' + EDGE_TTS_TOKEN + '&Sec-MS-GEC=' + gec + '&Sec-MS-GEC-Version=' + EDGE_CLIENT_VERSION;    let ws;
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
      const { text, source, target, userId, dialectFrom, dialectTo } = body;
      if (!text || !source || !target) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'text, source و target لازم است' }));
        return;
      }
      const context = Array.isArray(body.context) ? body.context : [];
      const safeUserId = userId ? String(userId).slice(0, 80) : null;
      const dialectHints = { from: dialectFrom ? String(dialectFrom) : '', to: dialectTo ? String(dialectTo) : '' };
      console.log('[translate] request ' + String(source) + ' -> ' + String(target) + ' chars=' + String(text).length + ' context=' + context.length);
      const result = await translateText(String(text), String(source), String(target), context, safeUserId, dialectHints);
      res.writeHead(200, {
        'Content-Type': 'application/json',
        'X-Translation-Engine': result.engine || 'unknown',
        'X-Translation-Model': result.model || 'unknown',
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
      const { lines, source, target, userId } = body;
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
      const safeUserId = userId ? String(userId).slice(0, 80) : null;
      const result = await translateLines(lines.map(String), String(source), String(target), safeUserId);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(result));
    } catch (err) {
      res.writeHead(502, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: err.message || 'ترجمه خط به خط انجام نشد' }));
    }
    return;
  }
  if (req.method === 'POST' && req.url === '/feedback') {
    try {
      const body = await readJsonBody(req, 5000);
      const { userId, targetLang, model, action } = body;
      if (!userId || !targetLang || !model) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'userId, targetLang و model لازم است' }));
        return;
      }
      if (action && action !== 'dislike') {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'فقط action=dislike پشتیبانی می‌شود' }));
        return;
      }
      // Only LLM-pool models can be user-excluded / globally banned this way —
      // the literal fallback engines (m2m100, deepl, google, ...) aren't part of
      // this rotation, so a dislike on those is simply ignored.
      const safeModel = String(model).slice(0, 120);
      if (!LLM_MODEL_POOL.includes(safeModel)) {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ ok: true, note: 'این مدل بخشی از چرخه‌ی مدل‌های زبانی نیست، بازخوردی ثبت نشد' }));
        return;
      }
      const result = registerDislike(String(userId).slice(0, 80), safeModel, String(targetLang).slice(0, 10));
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(result));
    } catch (err) {
      res.writeHead(502, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: err.message || 'ثبت بازخورد انجام نشد' }));
    }
    return;
}
 if (req.method === 'POST' && req.url === '/correction') {
    try {
      const body = await readJsonBody(req, 5000);
      const { userId, targetLang, source, bad, fixed } = body;
      if (!userId || !targetLang || !fixed) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'userId, targetLang و fixed لازم است' }));
        return;
      }
      addCorrection(String(userId).slice(0, 80), String(targetLang).slice(0, 10), source, bad, fixed);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ ok: true }));
    } catch (err) {
      res.writeHead(502, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: err.message || 'ثبت اصلاح انجام نشد' }));
    }
    return;
  }
  if (req.method === 'GET' && req.url.startsWith('/model-status')) {
    const urlObj = new URL(req.url, 'http://x');
    const lang = urlObj.searchParams.get('lang');
    const perLangDislikeCounts = {};
    if (lang && feedbackStore.modelDislikes[lang]) {
      for (const m of Object.keys(feedbackStore.modelDislikes[lang])) {
        perLangDislikeCounts[m] = feedbackStore.modelDislikes[lang][m].length;
      }
    }
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      pool: LLM_MODEL_POOL,
      lang: lang || null,
      availableForLang: lang ? availableModelsFor(null, lang) : null,
      globallyBannedForLang: lang ? (feedbackStore.globalBans[lang] || []) : null,
      distinctDislikeCountsForLang: lang ? perLangDislikeCounts : null,
      allGlobalBans: feedbackStore.globalBans,
      banThreshold: GLOBAL_MODEL_BAN_THRESHOLD,
    }));
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
      const { text, bcp, gender } = body;
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
          audio = await synthesizeEdgeTts(String(text), String(bcp), gender);
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
    (CF_ACCOUNT_ID && CF_API_TOKEN) ? ('LLM pool (Workers AI, free tier) configured: ' + LLM_MODEL_POOL.join(' → ')) : 'LLM pool (Workers AI, free tier) NOT configured',
    (ANTHROPIC_API_KEY ? 'Claude configured (optional bonus, not required)' : 'Claude NOT configured (optional — fine to leave unset)') + ' (' + CLAUDE_MODEL + ')',
    (CF_ACCOUNT_ID && CF_API_TOKEN) ? 'Workers AI (M2M-100 fallback) configured' : 'Workers AI (M2M-100 fallback) NOT configured',
    DEEPL_API_KEY ? 'DeepL configured' : 'DeepL NOT configured',
    'Google Translate + LibreTranslate fallbacks always available (no key needed)',
    (CF_ACCOUNT_ID && CF_API_TOKEN) ? 'Whisper transcription (Workers AI) configured' : 'Whisper transcription (Workers AI) NOT configured',
    ELEVENLABS_API_KEY ? 'ElevenLabs TTS configured' : 'ElevenLabs TTS NOT configured',
    'Edge TTS + Google TTS fallback available at POST /tts',
    'Model dislike feedback: POST /feedback, status: GET /model-status?lang=xx',
    'Per-user correction memory: POST /correction',
    kvConfigured() ? 'Persistence: Cloudflare KV (survives restarts)' : 'Persistence: local disk file only (LOST on restart if your host has an ephemeral disk — set CF_KV_NAMESPACE_ID to fix)',
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
        model: msg.model || null,
        targetLang: msg.targetLang || null,
        gender: msg.gender || null,
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
    (CF_ACCOUNT_ID && CF_API_TOKEN) ? ('LLM pool configured: ' + LLM_MODEL_POOL.join(' → ')) : 'LLM pool NOT configured',
    (ANTHROPIC_API_KEY ? 'Claude configured (optional bonus, not required)' : 'Claude NOT configured (optional — fine to leave unset)') + ' (' + CLAUDE_MODEL + ')',
    (CF_ACCOUNT_ID && CF_API_TOKEN) ? 'Workers AI (M2M-100 fallback) configured' : 'Workers AI (M2M-100 fallback) NOT configured',
    DEEPL_API_KEY ? 'DeepL configured' : 'DeepL NOT configured',
    ELEVENLABS_API_KEY ? 'ElevenLabs TTS configured' : 'ElevenLabs TTS NOT configured',
  ].join(', ');
  console.log('relay server listening on port ' + PORT + ' — ' + status);
}); 



  
