const http = require('http');
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const { WebSocketServer } = require('ws');
const WebSocket = require('ws');
const PORT = process.env.PORT || 3000;
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY || '';
const CLAUDE_MODEL = process.env.CLAUDE_MODEL || 'claude-sonnet-4-6';
// Groq (console.groq.com) — free tier, no credit card required to sign up.
// Hosts much larger open models (Llama 3.3 70B) than Cloudflare's free-tier
// pool, at very high speed. Optional: if GROQ_API_KEY is unset, translateWithGroq
// below fails fast with 'no-groq-key' and the chain just moves to the next engine.
const GROQ_API_KEY = process.env.GROQ_API_KEY || '';
const GROQ_MODEL = process.env.GROQ_MODEL || 'llama-3.3-70b-versatile';
// Google AI Studio (aistudio.google.com) — also free tier, no credit card.
// Gemini's multilingual quality is generally strong. Same optional/fail-fast
// pattern as Groq above if GEMINI_API_KEY is unset.
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || '';
const GEMINI_MODEL = process.env.GEMINI_MODEL || 'gemini-2.5-flash';
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
// LLM_MODEL_POOL: deliberately kept to ONE model — the strongest free-tier
// multilingual instruct model on Workers AI. This used to be a 4-model chain
// (70b -> 8b -> mistral-24b -> qwen-32b) that silently rotated down to a
// visibly weaker model the moment the strong one looked even slightly
// suspicious. That's exactly the failure users were hitting: a bad first
// roll would get "fixed" by downgrading quality instead of genuinely
// retrying. Now there is nothing weaker to fall back to within this pool —
// a person hitting 🔄 (retry-same-model, see retranslateWithSameEngineModel)
// always gets another honest attempt from this SAME strong model. If it
// truly can't produce anything usable, the caller falls through to the
// other translateText() engines (Claude, then the literal fallback
// engines) — never to a weaker model still labeled "AI translation".
const LLM_MODEL_POOL = [
  '@cf/meta/llama-3.3-70b-instruct-fp8-fast', // the one and only model in this pool — see note above
];
// Previously-tried weaker fallbacks (llama-3.1-8b-instruct,
// mistral-small-3.1-24b-instruct, qwen2.5-coder-32b-instruct) were removed on
// purpose — do not add them back as "just in case" fallbacks; that recreates
// the silent-downgrade problem this pool was trimmed to avoid. If Cloudflare's
// catalog changes and a genuinely comparable-or-better free-tier model shows
// up, it can replace the entry above — check
// https://developers.cloudflare.com/workers-ai/models/ for current model IDs
// and which ones are Free vs Paid. (glm-4.7-flash and kimi-k2.6 now require
// the Workers AI PAID plan; gemma-4-26b-a4b-it was returning empty responses.)
const CF_LLM_TRANSLATE_MODEL = LLM_MODEL_POOL[0]; // kept for status/log text below
// SMART LANGUAGE ROUTER — Groq/Gemini priority by target language; no Cloudflare key required.
const LANGUAGE_ENGINE_ROUTER = {
  fa:'groq', ar:'groq', en:'gemini', tr:'groq', fr:'gemini', de:'gemini', es:'gemini', it:'gemini',
  ru:'groq', ja:'gemini', ko:'gemini', hi:'gemini', ur:'groq', pt:'gemini', nl:'gemini', sv:'gemini',
  pl:'gemini', uk:'gemini', id:'gemini', vi:'gemini', th:'gemini', he:'gemini', el:'gemini', ro:'gemini',
  bn:'gemini', ms:'gemini'
};
function getLanguageEngine(toCode) {
  const code = String(toCode || '').toLowerCase().split('-')[0];
  return LANGUAGE_ENGINE_ROUTER[code] || 'groq';
}
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
// Shared by every single-message translation engine (Claude, Workers AI LLM
// pool, Groq, Gemini) so the instructions — including the greeting/farewell
// anti-confusion rule — never drift out of sync between engines.
function buildTranslationPromptParts(text, fromCode, toCode, context, dialectHints, corrections, avoidTranslation) {
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
    'Never swap a word for its opposite. This applies especially to greetings and farewells: a message that opens with ' +
    '"hello"/"hi"/"hey" (or the equivalent opening greeting in the source language) must be translated using the ' +
    'target language\'s own OPENING greeting, never its farewell — for example into Persian that is "سلام", never "خداحافظی". ' +
    'Likewise "bye"/"goodbye" must become the target language\'s farewell, never its greeting. ' +
    'Before answering, re-check that the first word of your translation matches the sense (greeting vs. farewell, yes vs. no, etc.) of the first word of the source message. ' +
    'Your entire output must be written in ' + toName + ' — never leave a source-language word, filler word, or phrase untranslated inside an otherwise-' + toName + ' sentence, and never mix two languages in one output. ' +
    'This includes the WRITING SYSTEM, not just the vocabulary: even a common loanword or filler (like "okay", "ok", "wow", "hi") must be written phonetically in ' + toName + '\'s own script/alphabet the way a native speaker would normally write it there — never left in Latin letters (or any other foreign script) inside a ' + toName + ' sentence. ' +
    'The only exceptions are proper nouns (personal names, brand names, place names) and terms that have no real equivalent in ' + toName + ' at all. ' +
    'Additionally, whenever you keep a word as a genuine LOANWORD from another language (most often English) rather than fully translating it — e.g. "okay", "wow", "bye", "cool" kept as loanwords instead of translated — wrap ONLY that one word using this exact marker: {{<phonetic spelling in ' + toName + '\'s own script>|<the word in its original spelling>}}. ' +
    'Example: translating the English filler "okay" into Persian, when you decide to keep it as a loanword, write {{اوکی|okay}} instead of writing "اوکی" or "okay" alone. ' +
    'Only use this marker for genuine loanwords being kept in their borrowed form — never wrap a normal, fully-translated ' + toName + ' word in it, and never use it for proper nouns. ' +
    'Before answering, re-read your own draft translation and rewrite any leftover source-language word you find (applying the loanword marker above where appropriate), so the final text you send is fully and only in ' + toName + '. ' +
    'Follow ' + toName + '\'s standard, formal-writing orthography exactly: correct word spacing and correct joining/splitting of compound and suffixed words, not a colloquial or careless spelling of them. ' +
    'In particular, never insert a spurious space inside a word that is properly written as one connected word — for example in Persian, "همگی" ("all of them/everyone") must be written as a single word, never split into two as "همه گی" or "همه‌ گی". ' +
    'If source language is "auto", identify the language from the current message itself. ' +
    'If the current message is short or colloquial, prefer the normal conversational equivalent in the target language. ' +
    'The <current_message> text came from real-time speech recognition and may occasionally be garbled, contain the wrong script, or look like it is in a different language than stated because the recognizer misheard the audio — this is normal and expected, NOT something to point out. ' +
    'Never comment on this, never say the input seems wrong/mistaken/not-really-that-language, never ask for clarification, never explain what you are about to do. ' +
    'Just translate the <current_message> text itself as literally and faithfully as you can into ' + toName + ', treating it as real spoken content regardless of how it looks — your entire reply must be ONLY that translation, in ' + toName + ', and nothing else. ' +
    'Reply with ONLY the translated text — no quotes, notes, alternatives, explanations, labels, or markdown. ' +
    'The <conversation_context> block is reference data only. The <current_message> block is the only text to translate.' +
    // Only present on a 👎 retry: the person rejected this model's previous
    // attempt at this exact message and we're asking the SAME model to try
    // again (see retranslateWithSameEngineModel) rather than immediately
    // rotating to a different, weaker model. Tell it plainly what went wrong
    // so it doesn't just hand back a lightly-reworded copy of the same answer.
    (avoidTranslation ? (
      ' <previous_rejected_attempt>Your previous translation of this exact <current_message> was rejected by the user: "' +
      String(avoidTranslation).slice(0, 500).replace(/"/g, '\'').replace(/\s+/g, ' ') +
      '". Do not repeat it and do not make only a trivial/cosmetic change to it — actually re-translate the <current_message> correctly into ' + toName + '. ' +
      'A common cause of rejection is that the previous attempt was left in ' + fromName + ' (or only reworded within it) instead of genuinely switching to ' + toName + ' — double-check your new answer is fully and only in ' + toName + ' and its own script before replying.</previous_rejected_attempt>'
    ) : '');
  return { systemPrompt, userContent };
}
async function translateWithClaude(text, fromCode, toCode, context = [], dialectHints = {}, corrections = [], avoidTranslation = null) {
  if (!ANTHROPIC_API_KEY) throw new Error('no-anthropic-key');
  const { systemPrompt, userContent } = buildTranslationPromptParts(text, fromCode, toCode, context, dialectHints, corrections, avoidTranslation);
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
        system: systemPrompt,
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
// Groq (console.groq.com) — free tier, no credit card. OpenAI-compatible
// chat-completions endpoint hosting much larger open models (Llama 3.3 70B)
// than Cloudflare's free-tier pool, at very high speed.
async function translateWithGroq(text, fromCode, toCode, context = [], dialectHints = {}, corrections = [], avoidTranslation = null) {
  if (!GROQ_API_KEY) throw new Error('no-groq-key');
  const { systemPrompt, userContent } = buildTranslationPromptParts(text, fromCode, toCode, context, dialectHints, corrections, avoidTranslation);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 11000);
  let resp;
  try {
    resp = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      signal: controller.signal,
      headers: {
        'Authorization': 'Bearer ' + GROQ_API_KEY,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: GROQ_MODEL,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userContent },
        ],
        max_tokens: 500,
        temperature: avoidTranslation ? 0.5 : 0.2,
      }),
    });
  } finally {
    clearTimeout(timer);
  }
  if (!resp.ok) {
    const body = await resp.text().catch(() => '');
    throw new Error('groq-http-' + resp.status + (body ? ': ' + body.slice(0, 200) : ''));
  }
  const data = await resp.json();
  const translated = data && data.choices && data.choices[0] && data.choices[0].message && String(data.choices[0].message.content || '').trim();
  if (!translated) throw new Error('groq-bad-response');
  return translated;
}
// Google AI Studio (aistudio.google.com) — also free tier, no credit card.
async function translateWithGemini(text, fromCode, toCode, context = [], dialectHints = {}, corrections = [], avoidTranslation = null) {
  if (!GEMINI_API_KEY) throw new Error('no-gemini-key');
  const { systemPrompt, userContent } = buildTranslationPromptParts(text, fromCode, toCode, context, dialectHints, corrections, avoidTranslation);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 11000);
  let resp;
  try {
    resp = await fetch(
      'https://generativelanguage.googleapis.com/v1beta/models/' + GEMINI_MODEL + ':generateContent?key=' + GEMINI_API_KEY,
      {
        method: 'POST',
        signal: controller.signal,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          systemInstruction: { parts: [{ text: systemPrompt }] },
          contents: [{ role: 'user', parts: [{ text: userContent }] }],
          generationConfig: { temperature: avoidTranslation ? 0.5 : 0.2, maxOutputTokens: 1024, thinkingConfig: { thinkingBudget: 0 } },
        }),
      }
    );
  } finally {
    clearTimeout(timer);
    }
  if (!resp.ok) {
    const body = await resp.text().catch(() => '');
    throw new Error('gemini-http-' + resp.status + (body ? ': ' + body.slice(0, 200) : ''));
  }
  const data = await resp.json();
  const parts = data && data.candidates && data.candidates[0] && data.candidates[0].content && data.candidates[0].content.parts;
  const translated = Array.isArray(parts) ? parts.map((p) => p && p.text || '').join('').trim() : '';
  if (!translated) throw new Error('gemini-bad-response');
  return translated;
}
async function translateWithWorkersAILLM(text, fromCode, toCode, context = [], model = CF_LLM_TRANSLATE_MODEL, dialectHints = {}, corrections = [], avoidTranslation = null) {
  if (!CF_ACCOUNT_ID || !CF_API_TOKEN) throw new Error('no-workers-ai-credentials');
  const { systemPrompt, userContent } = buildTranslationPromptParts(text, fromCode, toCode, context, dialectHints, corrections, avoidTranslation);
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
          temperature: avoidTranslation ? 0.5 : 0.2,
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
// Maps a language code to the Unicode script family a real translation into
// it should be dominated by. Codes not listed here (mixed-script or Latin-based
// with no distinctive extra range) are simply skipped by the check below.
const SCRIPT_FAMILY_FOR_LANG = {
  fa: 'arabic', ar: 'arabic', ur: 'arabic',
  ru: 'cyrillic', uk: 'cyrillic',
  ja: 'cjk', ko: 'hangul', hi: 'devanagari', th: 'thai', he: 'hebrew',
  el: 'greek', bn: 'bengali',
  en: 'latin', tr: 'latin', fr: 'latin', de: 'latin', es: 'latin', it: 'latin',
  pt: 'latin', nl: 'latin', sv: 'latin', pl: 'latin', id: 'latin', vi: 'latin',
  ro: 'latin', ms: 'latin',
};
const SCRIPT_RANGES = {
  arabic: /[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]/g,
  cyrillic: /[\u0400-\u04FF]/g,
  cjk: /[\u3040-\u30FF\u4E00-\u9FFF]/g,
  hangul: /[\uAC00-\uD7AF]/g,
  devanagari: /[\u0900-\u097F]/g,
  thai: /[\u0E00-\u0E7F]/g,
  hebrew: /[\u0590-\u05FF]/g,
  greek: /[\u0370-\u03FF]/g,
  bengali: /[\u0980-\u09FF]/g,
  latin: /[A-Za-z\u00C0-\u024F]/g,
};
// Counts characters of each known script family and returns whichever one has
// the most — a cheap, good-enough "what script is this text actually written
// in" check without pulling in a real language-detection library.
function dominantScriptOf(text) {
  const s = String(text || '');
  let best = null, bestCount = 0;
  for (const family of Object.keys(SCRIPT_RANGES)) {
    const matches = s.match(SCRIPT_RANGES[family]);
    const count = matches ? matches.length : 0;
    if (count > bestCount) { bestCount = count; best = family; }
  }
  return bestCount >= 3 ? best : null; // too little script-bearing text to judge
}
// Rough, cheap sanity check — NOT a quality judge. Meant to catch three clear
// failure shapes so the chain moves to the next model instead of quietly
// returning a broken translation: an empty/near-empty reply, a reply that's
// just the source text handed back untouched, or — the sneaky one, seen from
// weaker fallback models — a reply that's REWORDED but still written in the
// SOURCE language's script instead of actually switching to the target one
// (e.g. asked for fa->en and it just paraphrases in Persian again).
function looksSuspiciousTranslation(source, translated, fromCode, toCode) {
  const s = String(source || '').trim();
  const t = String(translated || '').trim();
  if (!t) return true;
  if (fromCode !== toCode && s.length > 8 && s.toLowerCase() === t.toLowerCase()) return true;
  if (s.length > 40 && t.length < s.length * 0.15) return true;
  if (fromCode !== toCode && fromCode !== 'auto') {
    const expectedScript = SCRIPT_FAMILY_FOR_LANG[toCode];
    const sourceScript = SCRIPT_FAMILY_FOR_LANG[fromCode];
    if (expectedScript && sourceScript && expectedScript !== sourceScript) {
      const actualScript = dominantScriptOf(t);
      if (actualScript && actualScript === sourceScript && actualScript !== expectedScript) return true;
    }
  }
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
  const expectedScript = SCRIPT_FAMILY_FOR_LANG[toCode];
  const sourceScript = SCRIPT_FAMILY_FOR_LANG[fromCode];
  const checkScript = fromCode !== 'auto' && expectedScript && sourceScript && expectedScript !== sourceScript;
  let substantial = 0;
  let identical = 0;
  let stillSourceScript = 0;
  for (let i = 0; i < sourceLines.length; i++) {
    const s = String(sourceLines[i] || '').trim();
    const t = String(translatedLines[i] || '').trim();
    // Only count lines with real letter content and some length — skip bare
    // numbers, single symbols, empty entries, which are fine to pass through.
    if (s.length < 4 || !/[a-zA-Z\u00C0-\u024F\u0600-\u06FF]/.test(s)) continue;
    substantial++;
    if (s.toLowerCase() === t.toLowerCase()) identical++;
    if (checkScript && dominantScriptOf(t) === sourceScript) stillSourceScript++;
  }
  if (substantial < 2) return false; // too little to judge reliably
  if ((identical / substantial) >= 0.7) return true;
  // Same "reworded but never actually switched script" failure as the single-
  // message check above, just judged across the whole batch instead of one line.
  if (checkScript && (stillSourceScript / substantial) >= 0.7) return true;
  return false;
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
// Re-runs translation with the SAME engine+model that produced a translation
// the user just disliked, asking it (via the <previous_rejected_attempt> block
// added in buildTranslationPromptParts) to genuinely redo it rather than
// rotate straight down to the next, usually weaker, model in the chain. Only
// on a real failure or another suspicious-looking result does the caller
// (the /retry-same-model route below) fall through to the normal chain.
async function retranslateWithSameEngineModel({ text, fromCode, toCode, context, userId, dialectHints, engine, model, previousTranslation }) {
  const corrections = getCorrectionsFor(userId, toCode);
  let translated;
  switch (engine) {
    case 'gemini':
      translated = await translateWithGemini(text, fromCode, toCode, context, dialectHints, corrections, previousTranslation);
      break;
    case 'groq':
      translated = await translateWithGroq(text, fromCode, toCode, context, dialectHints, corrections, previousTranslation);
      break;
    case 'claude':
      translated = await translateWithClaude(text, fromCode, toCode, context, dialectHints, corrections, previousTranslation);
      break;
    case 'workers-ai-llm':
      if (!model) throw new Error('retry-missing-model-for-workers-ai-llm');
      translated = await translateWithWorkersAILLM(text, fromCode, toCode, context, model, dialectHints, corrections, previousTranslation);
      break;
    default:
      // The literal fallback engines (m2m100, deepl, google, libretranslate)
      // aren't LLMs and have no notion of "try again differently" — nothing to
      // retry, so the caller should go straight to the normal fallback chain.
      throw new Error('retry-not-supported-for-engine-' + engine);
  }
  if (looksSuspiciousTranslation(text, translated, fromCode, toCode)) {
    throw new Error((model || engine) + '-retry-still-suspicious');
  }
  return translated;
}
// Shared by translateLinesWithModel and translateLinesWithClaude below — used
// to be pasted twice ("Same photo-OCR translation instructions as ... above"),
// which is exactly how the two copies could silently drift out of sync.
function buildLinesSystemPrompt(fromName, toName, lineCount) {
  return 'You are the translation engine behind a live camera-overlay translation feature (like Google Lens), ' +
    'translating text that was detected on a photographed image, from ' + fromName + ' to ' + toName + '. ' +
    'You will receive a numbered list. Each number is already a merged block of nearby on-image text that has been ' +
    'grouped together because it likely forms one running sentence/paragraph/caption — NOT an arbitrary single OCR ' +
    'line — so treat each numbered entry as a real chunk of prose to translate as a whole, not as an isolated word ' +
    'or fragment to be guessed at out of context. Read all the entries together so terminology, tone, and any ' +
    'pronoun/reference that continues from one entry to the next stay consistent — but you MUST reply with a ' +
    'translation for EVERY numbered entry, in the exact same order and exact same count as the input, one output ' +
    'entry per input entry. Never merge two input entries into one output entry or split one input entry into two. ' +
    'Every entry you receive already passed a filter that requires real letters in it, so nothing here is actually a ' +
    'bare number/symbol with no real words in it — do not second-guess that and blank one out anyway. A short entry, ' +
    'a stylized heading, a marketing tagline, or something that looks like it could be a brand name (e.g. "AI Labs", ' +
    '"beyond translation", a product name) still contains real words and MUST be genuinely translated like any other ' +
    'entry — never left blank and never returned unchanged just because it is short or looks like branding/a heading. ' +
    'Returning an empty string is reserved ONLY for the rare case an entry truly has no translatable words at all (a ' +
    'lone digit, a bare symbol) — if that ever happens, still return an entry for it (repeat it as-is or return an ' +
    'empty string) so the count always matches. ' +
    'Translate each entry the way a skilled bilingual native speaker would naturally phrase it — smooth, idiomatic, ' +
    'full-sentence phrasing in the target language, never a stiff word-for-word rendering, and never a fragment ' +
    'that only makes sense chained to a neighboring entry. Watch for words that are ambiguous in isolation but not ' +
    'in context (e.g. a verb that can mean either "want/like to" or "love", depending on what follows it) — use the ' +
    'surrounding entries to pick the sense that actually fits, rather than defaulting to the most literal one. ' +
    'Also watch for common English marketing idioms whose literal wording would flip the intended meaning if translated ' +
    'word-for-word — for example "going beyond X" or "more than X" means doing MORE than / in addition to X, never ' +
    'leaving or exiting X; translate the intended sense of the whole phrase, not each word on its own. ' +
    'Follow ' + toName + '\'s standard, formal-writing orthography exactly — correct word spacing and correct joining/splitting of compound and suffixed words (for example in Persian, "همگی" must stay one word, never split into "همه گی"). ' +
    'Never translate or alter numerals written as figures (e.g. "1", "2024", "۱۲", "01", "2/4"), dates, prices, codes, ' +
    'or standalone symbols/logos — copy those through exactly as they appear in the source text. This does NOT apply ' +
    'to spelled-out number words ("one", "two", "یک", "دو", "سه") — those are ordinary vocabulary and must be ' +
    'translated like any other word, into the equivalent number word in the target language. Only translate the ' +
    'surrounding words around a figure, never the figure itself. Each translated entry gets redrawn as one block covering the merged area its source text occupied ' +
    'on the photo, so it does NOT need to match the original\'s length line-for-line — prioritize a natural, correctly ' +
    'worded sentence over matching length. Reply with ONLY a raw JSON array of strings — no markdown, no code ' +
    'fence, no commentary — with exactly ' + lineCount + ' items in order.';
}
async function translateLinesWithModel(lines, fromCode, toCode, model) {
  if (!CF_ACCOUNT_ID || !CF_API_TOKEN) throw new Error('no-workers-ai-credentials');
  const fromName = langName(fromCode);
  const toName = langName(toCode);
  const numbered = lines.map((l, i) => (i + 1) + '. ' + String(l).replace(/\s+/g, ' ').trim()).join('\n');
  const systemPrompt = buildLinesSystemPrompt(fromName, toName, lines.length);
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
      system: buildLinesSystemPrompt(fromName, toName, lines.length),
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
// Server-side mirror of the client's hasTranslatableLetters (index.html) —
// same rule: a real word is never just one stray letter surrounded by
// symbols/digits, so require a 2+ letter run before treating text as
// something that should have been translated.
function hasTranslatableLetters(text) {
  return /\p{L}{2,}/u.test(String(text || ''));
}
// The batch prompt above is now explicit that every entry it receives has
// real words in it and must be genuinely translated — but a model can still
// occasionally ignore that and hand back an empty string for something it
// privately judged to be "just a heading/logo/brand name". Left alone, that
// silently skips drawing a translation box for that one line on the photo,
// so the ORIGINAL text stays visible there — looking like the translation
// randomly stopped partway through a sentence it actually did translate the
// rest of. Every genuinely-blank entry gets a one-off retry through the
// normal single-message pipeline (which has no such "logo" escape hatch)
// instead of being accepted as-is.
async function fillEmptyLineTranslations(lines, translated, fromCode, toCode, userId) {
  const out = translated.slice();
  for (let i = 0; i < lines.length; i++) {
    const src = String(lines[i] || '').trim();
    if (out[i] || !src || !hasTranslatableLetters(src)) continue;
    try {
      const r = await translateText(src, fromCode, toCode, [], userId);
      out[i] = r.translated;
    } catch (e) {
      console.warn('[translate-lines] could not fill blank entry ' + i + ' ("' + src.slice(0, 40) + '"): ' + e.message);
    }
  }
  return out;
}
async function translateLines(lines, fromCode, toCode, userId) {
  try {
    let translated = await translateLinesWithWorkersAILLM(lines, fromCode, toCode, userId);
    translated = await fillEmptyLineTranslations(lines, translated, fromCode, toCode, userId);
    return { translated, engine: 'workers-ai-llm-lines' };
  } catch (llmErr) {
    try {
      let translated = await translateLinesWithClaude(lines, fromCode, toCode);
      if (looksSuspiciousLinesTranslation(lines, translated, fromCode, toCode)) {
        throw new Error('claude-lines-echoed-untranslated-block');
      }
      translated = await fillEmptyLineTranslations(lines, translated, fromCode, toCode, userId);
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
// Names a person can pick in Settings -> "کدام هوش مصنوعی ترجمه کند؟". Anything
// else (missing, 'auto', or an unrecognized value) falls back to the normal
// best-first chain below with no reordering.
const SELECTABLE_ENGINES = ['groq', 'gemini', 'workers-ai-llm', 'claude', 'workers-ai-fallback', 'deepl-fallback', 'google-fallback', 'libretranslate-fallback'];
// --- Naturalizer: a post-translation polishing pass. It never replaces the
// translation engines above; it only takes an already-successful translation
// and asks Groq/Gemini to make it read more natively, falling back to the
// original translated text untouched if both naturalizer calls fail.
const NATURALIZER_ENABLED = String(process.env.NATURALIZER_ENABLED || 'true').toLowerCase() !== 'false';
const NATURALIZER_MIN_LENGTH = Number(process.env.NATURALIZER_MIN_LENGTH || 2);
function buildNaturalizerPrompt(text, fromCode, toCode, dialectHints = {}) {
  const target = String(toCode || '').toLowerCase().split('-')[0];
  const source = String(fromCode || '').toLowerCase().split('-')[0];
  const dialect = dialectHints && typeof dialectHints === 'object' ? JSON.stringify(dialectHints) : '{}';
  const system = 'You are a native-level localization editor. Improve the translation so it sounds natural, fluent, idiomatic and culturally appropriate to a native speaker of the TARGET language. Do NOT translate again from scratch unless necessary. Preserve the exact meaning, intent, tone, names, numbers, dates, URLs, codes, emojis and formatting. Do not add information, remove information, summarize, explain, censor, intensify, soften, or change the speaker intent. Preserve opening greetings as greetings and farewells as farewells. Keep technical terminology accurate. For short conversational text, prefer the wording a real native speaker would naturally use. Output ONLY the improved target-language text. No quotes, explanations, labels or markdown.';
  const user = '<source_language>' + source + '</source_language>\n<target_language>' + target + '</target_language>\n<dialect_hints>' + dialect + '</dialect_hints>\n<translation_to_polish>\n' + text + '\n</translation_to_polish>';
  return { system, user };
}
async function naturalizeWithGroq(text, fromCode, toCode, dialectHints = {}) {
  if (!GROQ_API_KEY) throw new Error('no-groq-key');
  const { system, user } = buildNaturalizerPrompt(text, fromCode, toCode, dialectHints);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 9000);
  let resp;
  try {
    resp = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST', signal: controller.signal,
      headers: { 'Authorization': 'Bearer ' + GROQ_API_KEY, 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: GROQ_MODEL, messages: [{ role: 'system', content: system }, { role: 'user', content: user }], max_tokens: 700, temperature: 0.15 })
    });
  } finally { clearTimeout(timer); }
  if (!resp.ok) throw new Error('groq-naturalizer-http-' + resp.status);
  const data = await resp.json();
  const out = data && data.choices && data.choices[0] && data.choices[0].message && String(data.choices[0].message.content || '').trim();
  if (!out) throw new Error('groq-naturalizer-bad-response');
  return out;
}
async function naturalizeWithGemini(text, fromCode, toCode, dialectHints = {}) {
  if (!GEMINI_API_KEY) throw new Error('no-gemini-key');
  const { system, user } = buildNaturalizerPrompt(text, fromCode, toCode, dialectHints);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 9000);
  let resp;
  try {
    resp = await fetch(
      'https://generativelanguage.googleapis.com/v1beta/models/' + GEMINI_MODEL + ':generateContent?key=' + GEMINI_API_KEY,
      {
        method: 'POST', signal: controller.signal,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ systemInstruction: { parts: [{ text: system }] }, contents: [{ role: 'user', parts: [{ text: user }] }], generationConfig: { temperature: 0.15, maxOutputTokens: 1024, thinkingConfig: { thinkingBudget: 0 } } })
      }
    );
  } finally { clearTimeout(timer); }
  if (!resp.ok) throw new Error('gemini-naturalizer-http-' + resp.status);
  const data = await resp.json();
  const parts = data && data.candidates && data.candidates[0] && data.candidates[0].content && data.candidates[0].content.parts;
  const out = Array.isArray(parts) ? parts.map((p) => p && p.text || '').join('').trim() : '';
  if (!out) throw new Error('gemini-naturalizer-bad-response');
  return out;
}
async function naturalizeTranslation(translated, fromCode, toCode, dialectHints = {}, preferredEngine = null) {
  if (!NATURALIZER_ENABLED || !translated || String(translated).trim().length < NATURALIZER_MIN_LENGTH) {
    return { translated, naturalized: false, naturalizer: null };
  }
  const chain = [
    { name: 'groq', run: () => naturalizeWithGroq(translated, fromCode, toCode, dialectHints) },
    { name: 'gemini', run: () => naturalizeWithGemini(translated, fromCode, toCode, dialectHints) },
  ];
  for (const engine of chain) {
    try {
      const polished = await engine.run();
      if (polished && polished.trim()) return { translated: polished.trim(), naturalized: true, naturalizer: engine.name };
    } catch (err) {
      console.error('[naturalizer] ' + engine.name + ' FAILED error=' + err.message);
    }
  }
  return { translated, naturalized: false, naturalizer: null };
}
// PINNED SINGLE ENGINE, NO SILENT CROSS-ENGINE FALLBACK.
// This used to be an 8-engine chain (strong LLM -> weaker LLM -> literal
// machine-translation services) that would quietly keep sliding down to a
// lower-quality engine on any hiccup from the good ones. Same problem as
// LLM_MODEL_POOL above, one level up: a person could ask for Groq/Gemini
// quality and transparently get Google-Translate-tier output instead, with
// nothing on screen saying so except the small engine badge.
// Now there is exactly ONE engine per call, chosen like this:
//   - Manual pick in Settings (preferredEngine) -> that exact engine, always.
//     Nothing else is ever substituted for it. If it fails, translation
//     fails — the person picked it on purpose, so silently swapping it out
//     from under them would defeat the point of picking it.
//   - "Auto" (no manual pick) -> the ONE strong engine LANGUAGE_ENGINE_ROUTER
//     designates for that target language (Groq or Gemini — see the router
//     above), and only that one. No drop-down through workers-ai-llm, Claude,
//     or the literal fallback engines (M2M-100/DeepL/Google/LibreTranslate).
// The literal fallback engines (translateWithWorkersAI/DeepL/Google/
// LibreTranslate) and the secondary LLM options (workers-ai-llm, claude) are
// NOT deleted — they still work fine and stay reachable by manually picking
// them in Settings (SELECTABLE_ENGINES). They're just no longer something the
// server ever switches you into behind your back.
// Logged the same way translation results have always been logged here
// (console.log('[translate] ...') on success, console.error on failure) and
// surfaced the same way on the client (setTranslationEngineBadge) — this
// change is about WHICH engine gets picked, not about how it's reported.
const ENGINE_RUNNERS = {
  'groq': (text, fromCode, toCode, context, dialectHints, corrections) =>
    ({ model: GROQ_MODEL, run: () => translateWithGroq(text, fromCode, toCode, context, dialectHints, corrections) }),
  'gemini': (text, fromCode, toCode, context, dialectHints, corrections) =>
    ({ model: GEMINI_MODEL, run: () => translateWithGemini(text, fromCode, toCode, context, dialectHints, corrections) }),
  'workers-ai-llm': (text, fromCode, toCode, context, dialectHints, corrections, userId) =>
    ({ model: null, run: () => translateWithLLMChain(text, fromCode, toCode, context, userId, dialectHints) }),
  'claude': (text, fromCode, toCode, context, dialectHints, corrections) =>
    ({ model: CLAUDE_MODEL, run: () => translateWithClaude(text, fromCode, toCode, context, dialectHints, corrections) }),
  'workers-ai-fallback': (text, fromCode, toCode) =>
    ({ model: CF_TRANSLATE_MODEL, run: () => translateWithWorkersAI(text, fromCode, toCode) }),
  'deepl-fallback': (text, fromCode, toCode) =>
    ({ model: 'deepl', run: () => translateWithDeepL(text, fromCode, toCode) }),
  'google-fallback': (text, fromCode, toCode) =>
    ({ model: 'google-translate', run: () => translateWithGoogle(text, fromCode, toCode) }),
  'libretranslate-fallback': (text, fromCode, toCode) =>
    ({ model: 'libretranslate', run: () => translateWithLibreTranslate(text, fromCode, toCode) }),
};
async function translateText(text, fromCode, toCode, context = [], userId = null, dialectHints = {}, preferredEngine = null) {
  const started = Date.now();
  const corrections = getCorrectionsFor(userId, toCode);
  const manualPick = !!(preferredEngine && SELECTABLE_ENGINES.includes(preferredEngine));
  const engineName = manualPick ? preferredEngine : getLanguageEngine(toCode);
  const engine = ENGINE_RUNNERS[engineName](text, fromCode, toCode, context, dialectHints, corrections, userId);
  try {
    const result = await engine.run();
    // translateWithLLMChain resolves to { translated, model }; every other
    // engine resolves to a plain translated string.
    const translated = (result && typeof result === 'object') ? result.translated : result;
    const model = (result && typeof result === 'object' && result.model) ? result.model : engine.model;
    const naturalized = await naturalizeTranslation(translated, fromCode, toCode, dialectHints, engineName);
    console.log('[translate] ' + engineName + (model ? ' (' + model + ')' : '') + ' OK naturalizer=' + (naturalized.naturalizer || 'none') + ' ms=' + (Date.now() - started));
    return { translated: naturalized.translated, engine: engineName, model, naturalized: naturalized.naturalized, naturalizer: naturalized.naturalizer };
  } catch (err) {
    console.error('[translate] ' + engineName + ' FAILED (pinned single-engine — no fallback engine tried) error=' + err.message);
    const isQuotaOrRateLimited = /-http-429\b|quota|rate.?limit/i.test(String(err && err.message || ''));
    if (!manualPick && isQuotaOrRateLimited && (engineName === 'gemini' || engineName === 'groq')) {
      const altName = engineName === 'gemini' ? 'groq' : 'gemini';
      try {
        const altEngine = ENGINE_RUNNERS[altName](text, fromCode, toCode, context, dialectHints, corrections, userId);
        const altResult = await altEngine.run();
        const translated = (altResult && typeof altResult === 'object') ? altResult.translated : altResult;
        const model = (altResult && typeof altResult === 'object' && altResult.model) ? altResult.model : altEngine.model;
        const naturalized = await naturalizeTranslation(translated, fromCode, toCode, dialectHints, altName);
        console.log('[translate] ' + altName + (model ? ' (' + model + ')' : '') + ' OK (fallback from quota-limited ' + engineName + ') naturalizer=' + (naturalized.naturalizer || 'none') + ' ms=' + (Date.now() - started));
        return { translated: naturalized.translated, engine: altName, model, naturalized: naturalized.naturalized, naturalizer: naturalized.naturalizer };
      } catch (altErr) {
        console.error('[translate] ' + altName + ' fallback also FAILED error=' + altErr.message);
        throw new Error(engineName + ' failed: ' + err.message + ' | ' + altName + ' fallback also failed: ' + altErr.message);
      }
    }
    throw new Error(engineName + ' failed: ' + err.message);
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
// --- Loanword accent markers (mirror of the client-side copy in index.html) --
// The translation prompt above asks the model to wrap a genuine loanword like
// "okay" as {{اوکی|okay}} — native-script spelling first, original spelling
// after "|". Edge TTS's SSML can switch just that one word to a real accent
// via <lang>; engines that can't (ElevenLabs, the Google fallback) instead
// get the marker stripped down to its native-script spelling.
const LOANWORD_MARKER_RE = /\{\{([^{}|]+)\|([^{}]+)\}\}/g;
function stripLoanwordMarkers(text) {
  LOANWORD_MARKER_RE.lastIndex = 0;
  return String(text || '').replace(LOANWORD_MARKER_RE, (m, native) => native);
}
function buildLoanwordSsmlBody(text) {
  const s = String(text || '');
  let out = '', last = 0, m;
  LOANWORD_MARKER_RE.lastIndex = 0;
  while ((m = LOANWORD_MARKER_RE.exec(s))) {
    out += escapeXml(s.slice(last, m.index));
    out += "<lang xml:lang='en-US'>" + escapeXml(m[2]) + '</lang>';
    last = m.index + m[0].length;
  }
  out += escapeXml(s.slice(last));
  return out;
}
const EDGE_CLIENT_VERSION = '1-143.0.3650.75';
function synthesizeEdgeTts(text, bcp, gender) {
  return new Promise((resolve, reject) => {
    const voice = pickEdgeVoice(bcp, gender);
    const gec = edgeSecMsGec();
    const wsUrl = 'wss://speech.platform.bing.com/consumer/speech/synthesize/readaloud/edge/v1'
      + '?TrustedClientToken=' + EDGE_TTS_TOKEN + '&Sec-MS-GEC=' + gec + '&Sec-MS-GEC-Version=' + EDGE_CLIENT_VERSION;    let ws;
     try {
      const uaVersion = EDGE_CLIENT_VERSION.split('-')[1] || '131.0.2903.0';
      ws = new WebSocket(wsUrl, {
        headers: {
          'Pragma': 'no-cache',
          'Cache-Control': 'no-cache',
          'Origin': 'chrome-extension://jdiccldimpdaibmpdkjnbmckianbfold',
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/' + uaVersion + ' Safari/537.36 Edg/' + uaVersion,
          'Accept-Encoding': 'gzip, deflate, br',
          'Accept-Language': 'en-US,en;q=0.9',
        },
      });
    } catch (e) { reject(e); return; }
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
        + "<voice name='" + voice + "'><prosody pitch='+0Hz' rate='+0%' volume='+0%'>" + buildLoanwordSsmlBody(text) + "</prosody></voice></speak>";
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
      const { text, source, target, userId, dialectFrom, dialectTo, engine } = body;
      if (!text || !source || !target) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'text, source و target لازم است' }));
        return;
      }
      const context = Array.isArray(body.context) ? body.context : [];
      const safeUserId = userId ? String(userId).slice(0, 80) : null;
      const dialectHints = { from: dialectFrom ? String(dialectFrom) : '', to: dialectTo ? String(dialectTo) : '' };
      const preferredEngine = engine ? String(engine) : null;
      console.log('[translate] request ' + String(source) + ' -> ' + String(target) + ' chars=' + String(text).length + ' context=' + context.length + (preferredEngine ? ' preferredEngine=' + preferredEngine : ''));
      const result = await translateText(String(text), String(source), String(target), context, safeUserId, dialectHints, preferredEngine);
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
  if (req.method === 'POST' && req.url === '/retry-same-model') {
    try {
      const body = await readJsonBody(req, 5000);
      const { userId, fromCode, toCode, text, engine, model, previousTranslation, context, dialectHints } = body;
      if (!userId || !fromCode || !toCode || !text || !engine) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'userId, fromCode, toCode, text و engine لازم است' }));
        return;
      }
      const safeUserId = String(userId).slice(0, 80);
      const safeFromCode = String(fromCode).slice(0, 10);
      const safeToCode = String(toCode).slice(0, 10);
      const safeText = String(text).slice(0, 4000);
      const safeEngine = String(engine).slice(0, 40);
      const safeModel = model ? String(model).slice(0, 120) : null;
      const safePrevious = previousTranslation ? String(previousTranslation).slice(0, 2000) : null;
      const safeContext = Array.isArray(context) ? context : [];
      const safeHints = dialectHints && typeof dialectHints === 'object' ? dialectHints : {};
      let retried = false;
      let fellBack = false;
      let result;
      try {
        const translated = await retranslateWithSameEngineModel({
          text: safeText, fromCode: safeFromCode, toCode: safeToCode, context: safeContext,
          userId: safeUserId, dialectHints: safeHints, engine: safeEngine, model: safeModel,
          previousTranslation: safePrevious,
        });
        retried = true;
        result = { translated, engine: safeEngine, model: safeModel };
      } catch (err) {
        console.warn('[retry-same-model] same-model retry did not work out (' + safeEngine + (safeModel ? '/' + safeModel : '') + '): ' + err.message + ' — falling back to the normal engine chain');
        // Only Workers AI LLM pool models are ever actually rotated away from a
        // given user for a given language — this is the ONE place that exclusion
        // gets recorded, and only once the same model has already had its own
        // fair retry and still came back bad/suspicious.
        if (safeModel && LLM_MODEL_POOL.includes(safeModel)) {
          registerDislike(safeUserId, safeModel, safeToCode);
        }
        result = await translateText(safeText, safeFromCode, safeToCode, safeContext, safeUserId, safeHints);
        fellBack = true;
      }
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ ok: true, retried, fellBack, ...result }));
    } catch (err) {
      res.writeHead(502, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: err.message || 'تلاش دوباره برای ترجمه انجام نشد' }));
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
      console.log('[tts] request bcp=' + String(bcp) + ' gender=' + String(gender || '-') + ' chars=' + String(text).length);
      let audio, engine;
      try {
        audio = await synthesizeElevenLabsTts(stripLoanwordMarkers(String(text)));
        engine = 'elevenlabs';
      } catch (elevenErr) {
        try {
          // Edge TTS is the only engine here that actually honors the loanword
          // accent marker (see buildLoanwordSsmlBody) — it gets the raw text.
          audio = await synthesizeEdgeTts(String(text), String(bcp), gender);
          engine = 'edge';
        } catch (edgeErr) {
          try {
            audio = await synthesizeGoogleTts(stripLoanwordMarkers(String(text)), String(bcp).slice(0, 2));
            engine = 'google-fallback';
          } catch (googleErr) {
            console.error('[tts] ALL ENGINES FAILED bcp=' + String(bcp) + ' elevenlabs=' + elevenErr.message + ' | edge=' + edgeErr.message + ' | google=' + googleErr.message);
            throw new Error('elevenlabs: ' + elevenErr.message + ' | edge: ' + edgeErr.message + ' | google: ' + googleErr.message);
          }
        }
      }
      console.log('[tts] OK bcp=' + String(bcp) + ' engine=' + engine + ' bytes=' + audio.length);
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
    GROQ_API_KEY ? ('Groq configured (' + GROQ_MODEL + ', free tier no card)') : 'Groq NOT configured (optional — free, no card, get a key at console.groq.com)',
    GEMINI_API_KEY ? ('Gemini configured (' + GEMINI_MODEL + ', free tier no card)') : 'Gemini NOT configured (optional — free, no card, get a key at aistudio.google.com)',
    (CF_ACCOUNT_ID && CF_API_TOKEN) ? ('LLM pool (Workers AI, free tier) configured: ' + LLM_MODEL_POOL.join(' → ')) : 'LLM pool (Workers AI, free tier) NOT configured',
    (ANTHROPIC_API_KEY ? 'Claude configured (optional bonus, not required)' : 'Claude NOT configured (optional — fine to leave unset)') + ' (' + CLAUDE_MODEL + ')',
    (CF_ACCOUNT_ID && CF_API_TOKEN) ? 'Workers AI (M2M-100 fallback) configured' : 'Workers AI (M2M-100 fallback) NOT configured',
    DEEPL_API_KEY ? 'DeepL configured' : 'DeepL NOT configured',
    'Google Translate + LibreTranslate fallbacks always available (no key needed)',
    (CF_ACCOUNT_ID && CF_API_TOKEN) ? 'Whisper transcription (Workers AI) configured' : 'Whisper transcription (Workers AI) NOT configured',
    ELEVENLABS_API_KEY ? 'ElevenLabs TTS configured' : 'ElevenLabs TTS NOT configured',
    'Edge TTS + Google TTS fallback available at POST /tts',
    'Model dislike feedback: POST /feedback, same-model retry: POST /retry-same-model, status: GET /model-status?lang=xx',
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
        engine: msg.engine || null,
        targetLang: msg.targetLang || null,
        gender: msg.gender || null,
        msgId: msg.msgId || null,
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
    // Read receipts: relay a delivered/seen ack for a given msgId straight back
    // to whichever side originally sent that message — same "other side of this
    // session" lookup as 'chat' above. If that side is momentarily disconnected
    // (reconnecting after a network blip, exactly when this ack tends to fire),
    // queue it in the same s.pending used for 'chat' messages instead of just
    // dropping it — otherwise the sender's tick got stuck on a single gray
    // check forever, since nothing ever retried a lost ack.
    if (msg.type === 'delivered' || msg.type === 'seen') {
      const s = sessions.get(ws.code);
      if (!s || !msg.msgId) return;
      const target = otherSide(s, ws.role);
      const payload = { type: msg.type, msgId: String(msg.msgId) };
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
    // A person disliked their own outgoing message and it got retranslated
    // (see createDislikeButton's onTranslated in index.html) — relay the fixed
    // text to the other side's already-shown copy of that same message. Same
    // "other side of this session" + pending-queue-on-disconnect pattern as
    // 'chat' and the read receipts above.
    if (msg.type === 'correction') {
      const s = sessions.get(ws.code);
      if (!s || !msg.msgId) return;
      const target = otherSide(s, ws.role);
      const payload = { type: 'correction', msgId: String(msg.msgId), translated: msg.translated };
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
    // Someone picked (or cleared) a reaction emoji on a message — relay the new
    // state to the other side's copy of that same message by msgId, so a
    // reaction on either person's screen shows up on both. Same relay +
    // pending-queue-on-disconnect pattern as 'correction'/'delivered'/'seen'
    // above. msg.emoji is a short emoji string, or null/absent to clear it —
    // never trusted as arbitrary long text, just capped and passed through.
    if (msg.type === 'reaction') {
      const s = sessions.get(ws.code);
      if (!s || !msg.msgId) return;
      const target = otherSide(s, ws.role);
      const emoji = (typeof msg.emoji === 'string' && msg.emoji) ? msg.emoji.slice(0, 8) : null;
      const payload = { type: 'reaction', msgId: String(msg.msgId), emoji };
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
    GROQ_API_KEY ? ('Groq configured (' + GROQ_MODEL + ')') : 'Groq NOT configured',
    GEMINI_API_KEY ? ('Gemini configured (' + GEMINI_MODEL + ')') : 'Gemini NOT configured',
    (CF_ACCOUNT_ID && CF_API_TOKEN) ? ('LLM pool configured: ' + LLM_MODEL_POOL.join(' → ')) : 'LLM pool NOT configured',
    (ANTHROPIC_API_KEY ? 'Claude configured (optional bonus, not required)' : 'Claude NOT configured (optional — fine to leave unset)') + ' (' + CLAUDE_MODEL + ')',
    (CF_ACCOUNT_ID && CF_API_TOKEN) ? 'Workers AI (M2M-100 fallback) configured' : 'Workers AI (M2M-100 fallback) NOT configured',
    DEEPL_API_KEY ? 'DeepL configured' : 'DeepL NOT configured',
    ELEVENLABS_API_KEY ? 'ElevenLabs TTS configured' : 'ElevenLabs TTS NOT configured',
  ].join(', ');
  console.log('relay server listening on port ' + PORT + ' — ' + status);
}); 
