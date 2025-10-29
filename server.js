// server.js
require('dotenv').config();

const fs = require('fs');
const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const Vision = require('@google-cloud/vision');
const OpenAI = require('openai');
const axios = require('axios');
const cheerio = require('cheerio');
const sharp = require('sharp');

// -------------------- required env --------------------
if (!process.env.OPENAI_API_KEY) {
  console.error('OPENAI_API_KEY is missing.');
  process.exit(1);
}
process.env.GOOGLE_AUTH_DISABLE_GCE_CHECK = process.env.GOOGLE_AUTH_DISABLE_GCE_CHECK || 'true';

// -------------------- clients --------------------
function createVisionClient() {
  try {
    if (process.env.GOOGLE_APPLICATION_CREDENTIALS) {
      const keyPath = process.env.GOOGLE_APPLICATION_CREDENTIALS;
      if (!fs.existsSync(keyPath)) {
        console.error('Vision key file not found:', keyPath);
        process.exit(1);
      }
      console.log('Vision client: using key file:', keyPath);
      return new Vision.ImageAnnotatorClient({ keyFilename: keyPath });
    }
    if (process.env.GOOGLE_CREDENTIALS) {
      const creds = JSON.parse(process.env.GOOGLE_CREDENTIALS);
      console.log('Vision client: using inline GOOGLE_CREDENTIALS.');
      return new Vision.ImageAnnotatorClient({
        credentials: { client_email: creds.client_email, private_key: creds.private_key },
      });
    }
    console.warn('Vision client: using default ADC (not recommended locally).');
    return new Vision.ImageAnnotatorClient();
  } catch (e) {
    console.error('Failed to create Vision client:', e);
    process.exit(1);
  }
}
const visionClient = createVisionClient();
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// -------------------- express --------------------
const app = express();
app.use(bodyParser.json({ limit: '25mb' }));
const allowedOrigins = process.env.CORS_ORIGIN
  ? process.env.CORS_ORIGIN.split(',').map(s => s.trim())
  : ['http://localhost:8080'];
app.use(cors({ origin: allowedOrigins }));

// -------------------- helpers --------------------
function sanitizeUrl(url) {
  if (!url) return null;
  try {
    // Accept absolute http/https
    const u = new URL(url, 'https://example.com');
    const href = url.startsWith('http') ? url : (u.origin === 'https://example.com' ? null : u.href);
    if (!href) return null;
    if (!/^https?:\/\//i.test(href)) return null;
    return href;
  } catch { return null; }
}
function normCurrency(cur) {
  if (!cur) return 'EUR';
  if (cur === '€') return 'EUR';
  return cur.toUpperCase();
}
function trimTitle(t) {
  return (t || '').replace(/\s+/g, ' ').trim();
}

async function runVisionAllFeatures(imageBytes) {
  const [resp] = await visionClient.annotateImage({
    image: { content: imageBytes },
    features: [
      { type: 'LABEL_DETECTION', maxResults: 20 },
      { type: 'LOGO_DETECTION', maxResults: 10 },
      { type: 'OBJECT_LOCALIZATION', maxResults: 10 },
      { type: 'DOCUMENT_TEXT_DETECTION' },
      { type: 'WEB_DETECTION' },
      { type: 'SAFE_SEARCH_DETECTION' },
      { type: 'FACE_DETECTION' },
    ],
    imageContext: { languageHints: ['en','fr','de','es','it','nl','pl'] }
  });

  const labels = (resp.labelAnnotations || []).map(l => ({ description: l.description, score: l.score }));
  const logos = (resp.logoAnnotations || []).map(x => x.description);
  const objects = (resp.localizedObjectAnnotations || []).map(x => ({
    name: x.name, score: x.score, box: x.boundingPoly?.normalizedVertices || []
  }));
  const safeSearch = resp.safeSearchAnnotation || {};
  const faceCount = (resp.faceAnnotations || []).length;
  const ocrText =
    (resp.fullTextAnnotation && resp.fullTextAnnotation.text) ||
    (resp.textAnnotations && resp.textAnnotations[0] && resp.textAnnotations[0].description) ||
    '';

  const web = resp.webDetection || {};
  const bestGuess = (web.bestGuessLabels && web.bestGuessLabels[0] && web.bestGuessLabels[0].label) || '';
  const webEntities = (web.webEntities || [])
    .filter(e => e.description)
    .sort((a, b) => (b.score || 0) - (a.score || 0))
    .slice(0, 12)
    .map(e => e.description);

  return { labels, logos, objects, safeSearch, faceCount, ocrText, bestGuess, webEntities };
}

function extractModelHints(ocrText = '') {
  const hints = new Set();
  (ocrText.match(/\b[0-9]{4,6}\b/g) || []).forEach(n => hints.add(n));
  (ocrText.match(/\bSM-[A-Z0-9-]+\b/gi) || []).forEach(n => hints.add(n.toUpperCase()));
  (ocrText.match(/\b(?=[A-Z0-9-]*[0-9])(?=[A-Z0-9-]*[A-Z])[A-Z0-9-]{5,}\b/gi) || [])
    .forEach(n => hints.add(n.toUpperCase()));
  return Array.from(hints).slice(0, 10);
}

async function prepForOCR(buf){
  return await sharp(buf)
    .rotate()
    .resize({ width: 1600, withoutEnlargement: true })
    .sharpen()
    .gamma()
    .normalise()
    .toBuffer();
}

async function cropPrimaryObject(imageBytes) {
  const [objResp] = await visionClient.objectLocalization({ image: { content: imageBytes } });
  const anns = objResp.localizedObjectAnnotations || [];
  if (!anns.length) return null;
  const top = anns.sort((a, b) => (b.score || 0) - (a.score || 0))[0];
  const verts = top.boundingPoly?.normalizedVertices || [];
  if (verts.length < 4) return null;

  const img = sharp(imageBytes);
  const meta = await img.metadata();
  const W = meta.width || 0, H = meta.height || 0;
  if (!W || !H) return null;

  const xs = verts.map(v => Math.max(0, Math.min(1, v.x || 0)));
  const ys = verts.map(v => Math.max(0, Math.min(1, v.y || 0)));
  let minX = Math.min(...xs), maxX = Math.max(...xs);
  let minY = Math.min(...ys), maxY = Math.max(...ys);

  const pad = 0.06;
  minX = Math.max(0, minX - pad); maxX = Math.min(1, maxX + pad);
  minY = Math.max(0, minY - pad); maxY = Math.min(1, maxY + pad);

  const left = Math.floor(minX * W);
  const topY = Math.floor(minY * H);
  const width = Math.min(W - left, Math.ceil((maxX - minX) * W));
  const height = Math.min(H - topY, Math.ceil((maxY - minY) * H));
  if (width < 50 || height < 50) return null;

  return await img.extract({ left, top: topY, width, height }).toBuffer();
}

// -------------------- OpenAI Vision “second opinion” (compact) --------------------
async function runOpenAIVisionExtract(base64) {
  try {
    const dataUrl = `data:image/jpeg;base64,${base64}`;
    const system = `Extract product facts from an image. Return ONLY compact JSON:
{"brand":"","model":"","product_type":"","synonyms":[],"confidence":0..1}`;
    const messages = [
      { role: 'system', content: system },
      { role: 'user', content: [
          { type: 'input_text', text: 'Identify brand and model if visible.' },
          { type: 'input_image', image_url: dataUrl }
      ]}
    ];
    const out = await openai.chat.completions.create({
      model: process.env.OPENAI_MODEL || 'gpt-4o-mini',
      messages,
      temperature: 0
    });
    const raw = out?.choices?.[0]?.message?.content?.trim?.() || '{}';
    let json = {};
    try { json = JSON.parse(raw); } catch { json = {}; }
    return {
      brand: json.brand || '',
      model: json.model || '',
      product_type: json.product_type || '',
      synonyms: Array.isArray(json.synonyms) ? json.synonyms : [],
      confidence: typeof json.confidence === 'number' ? json.confidence : 0
    };
  } catch (e) {
    console.warn('OpenAI Vision second-opinion failed:', e.message);
    return null;
  }
}

// -------------------- /api/vision --------------------
app.post('/api/vision', async (req, res) => {
  try {
    const { imageBase64 } = req.body;
    if (!imageBase64) return res.status(400).json({ error: 'imageBase64 required' });
    const raw = Buffer.from(imageBase64, 'base64');
    if (!raw.length) return res.status(400).json({ error: 'bad image' });

    const cleaned = await prepForOCR(raw);
    const base = await runVisionAllFeatures(cleaned);

    if ((base.faceCount || 0) > 0) {
      return res.status(400).json({ error: 'Image contains faces. Please upload only product images.' });
    }

    let merged = { ...base };
    try {
      const crop = await cropPrimaryObject(cleaned);
      if (crop) {
        const cropClean = await prepForOCR(crop);
        const cropSignals = await runVisionAllFeatures(cropClean);
        merged = {
          ...merged,
          labels: [...(merged.labels||[]), ...(cropSignals.labels||[])],
          logos: Array.from(new Set([...(merged.logos||[]), ...(cropSignals.logos||[])])),
          objects: [...(merged.objects||[]), ...(cropSignals.objects||[])],
          ocrText: [merged.ocrText||'', cropSignals.ocrText||''].filter(Boolean).join('\n'),
          bestGuess: merged.bestGuess || cropSignals.bestGuess,
          webEntities: Array.from(new Set([...(merged.webEntities||[]), ...(cropSignals.webEntities||[])])),
        };
      }
    } catch (e) { console.warn('Auto-crop skipped:', e.message); }

    const modelHints = extractModelHints(merged.ocrText);
    const secondOpinion = await runOpenAIVisionExtract(imageBase64);
    if (secondOpinion) {
      if (secondOpinion.brand) merged.logos = Array.from(new Set([...(merged.logos||[]), secondOpinion.brand]));
      if (secondOpinion.product_type) merged.webEntities = Array.from(new Set([...(merged.webEntities||[]), secondOpinion.product_type]));
    }

    return res.json({ ...merged, modelHints, secondOpinion });
  } catch (err) {
    console.error('Vision error:', err);
    return res.status(500).json({ error: err.message || 'vision error' });
  }
});

// -------------------- /api/openai/intent --------------------
app.post('/api/openai/intent', async (req, res) => {
  try {
    const {
      labels = [],
      logos = [],
      webEntities = [],
      bestGuess = '',
      ocrText = '',
      modelHints = [],
      userPrompt = '',
      secondOpinion = null
    } = req.body;

    const labelList = labels.slice(0, 12).map(l => (typeof l === 'string' ? l : l.description)).join(', ');
    const brandList = logos.slice(0, 8).join(', ');
    const entities  = (webEntities || []).slice(0, 10).join(', ');
    const hints     = (modelHints || []).slice(0, 8).join(' ');
    const soBrand   = secondOpinion?.brand || '';
    const soModel   = secondOpinion?.model || '';
    const soType    = secondOpinion?.product_type || '';
    const soConf    = secondOpinion?.confidence || 0;

    const system = `You classify intent and craft a concrete marketplace query.
Return JSON:
- intent: ["buy","sell","compare","clarify"]
- confidence: 0..1
- bestQuery: <=5 tokens; prefer BRAND + MODEL (e.g., "lego 75288", "dyson v8 absolute")
- followUp: optional`;

    const user = `Signals:
Logos: ${brandList || 'none'}
OpenAI Vision: ${[soBrand, soModel, soType].filter(Boolean).join(' ')} (conf=${soConf})
Best guess: ${bestGuess || 'none'}
Entities: ${entities || 'none'}
Labels: ${labelList || 'none'}
OCR: ${(ocrText || '').slice(0, 180)}
Model hints: ${hints || 'none'}
User prompt: ${userPrompt || 'none'}

Rule: Prefer BRAND + MODEL when available. No filler words.`;

    const completion = await openai.chat.completions.create({
      model: process.env.OPENAI_MODEL || 'gpt-4o-mini',
      messages: [{ role: 'system', content: system }, { role: 'user', content: user }],
      temperature: 0
    });

    const raw = completion?.choices?.[0]?.message?.content?.trim?.() || '';
    let parsed;
    try { parsed = JSON.parse(raw); }
    catch {
      const fallback = [soBrand, soModel].filter(Boolean).join(' ') || hints || entities.split(',')[0] || 'product';
      parsed = { intent: 'clarify', confidence: 0.6, bestQuery: fallback, followUp: 'Which exact product name/model?' };
    }
    parsed.intent ||= 'clarify';
    if (typeof parsed.confidence !== 'number') parsed.confidence = 0.6;
    if (!parsed.bestQuery) parsed.bestQuery = [soBrand, soModel].filter(Boolean).join(' ') || hints || 'product';

    return res.json(parsed);
  } catch (err) {
    console.error('OpenAI intent error:', err);
    return res.status(500).json({ error: err.message || 'openai error' });
  }
});

// -------------------- eBay (Browse + Finding fallback) --------------------
let _ebayToken = null;
let _ebayTokenExp = 0;

async function getEbayAppToken() {
  const now = Math.floor(Date.now() / 1000);
  if (_ebayToken && now < _ebayTokenExp - 60) return _ebayToken;
  const id = process.env.EBAY_CLIENT_ID;
  const secret = process.env.EBAY_CLIENT_SECRET;
  if (!id || !secret) return null;

  const basic = Buffer.from(`${id}:${secret}`).toString('base64');
  const form = new URLSearchParams({
    grant_type: 'client_credentials',
    scope: 'https://api.ebay.com/oauth/api_scope'
  });
  const resp = await axios.post('https://api.ebay.com/identity/v1/oauth2/token', form.toString(), {
    headers: { 'Content-Type': 'application/x-www-form-urlencoded', 'Authorization': `Basic ${basic}` },
    timeout: 10000
  });
  _ebayToken = resp.data.access_token;
  _ebayTokenExp = now + (resp.data.expires_in || 7200);
  return _ebayToken;
}

async function callEbayBrowse(results, query) {
  try {
    const token = await getEbayAppToken();
    if (!token) return false;

    // Prioritize GB for “Irish & British marketplaces”
    const marketplaces = ['EBAY_GB'];
    for (const mp of marketplaces) {
      const url = 'https://api.ebay.com/buy/browse/v1/item_summary/search';
      const r = await axios.get(url, {
        params: { q: query, limit: 20 },
        headers: {
          'Authorization': `Bearer ${token}`,
          'X-EBAY-C-MARKETPLACE-ID': mp,
          'Accept': 'application/json'
        },
        timeout: 10000
      });
      const items = r.data.itemSummaries || [];
      items.forEach(it => {
        const title = trimTitle(it.title);
        const price = it.price ? parseFloat(it.price.value) : null;
        const currency = normCurrency(it.price?.currency || 'EUR');
        const link = sanitizeUrl(it.itemWebUrl || it.itemAffiliateWebUrl);
        if (title) results.push({ title, source: `eBay (${mp})`, price, currency, url: link });
      });
      if (results.length >= 6) break;
    }
    return true;
  } catch (e) {
    console.warn('eBay Browse failed:', e.response?.status || e.message);
    return false;
  }
}

async function callEbayFinding(results, query) {
  if (!process.env.EBAY_APP_ID) return false;

  const callFinding = async (globalId) => {
    const endpoint = 'https://svcs.ebay.com/services/search/FindingService/v1';
    const params = new URLSearchParams({
      'OPERATION-NAME': 'findItemsByKeywords',
      'SERVICE-VERSION': '1.13.0',
      'SECURITY-APPNAME': process.env.EBAY_APP_ID,
      'RESPONSE-DATA-FORMAT': 'JSON',
      'REST-PAYLOAD': 'true',
      'GLOBAL-ID': globalId,
      'paginationInput.entriesPerPage': '20',
      'keywords': query
    });
    try {
      const r = await axios.get(`${endpoint}?${params.toString()}`, {
        timeout: 10000,
        headers: { 'User-Agent': 'ProductScanner/1.0 (+local)', 'Accept': 'application/json' }
      });
      const resp = r.data?.findItemsByKeywordsResponse?.[0] || {};
      const items = resp.searchResult?.[0]?.item || [];
      items.forEach(it => {
        const title = trimTitle(it.title?.[0]);
        const pinfo = it.sellingStatus?.[0]?.currentPrice?.[0] || {};
        const currency = normCurrency(pinfo['@currencyId'] || 'EUR');
        const price = pinfo['__value__'] ? parseFloat(pinfo['__value__']) : null;
        const link = sanitizeUrl(it.viewItemURL?.[0]);
        if (title) results.push({ title, source: `eBay (${globalId})`, price, currency, url: link });
      });
      return true;
    } catch (e) {
      console.warn(`[eBay] ${globalId} failed:`, e.response?.status || e.message);
      return false;
    }
  };

  // IE & GB first; US only if both fail
  let ok = await callFinding('EBAY-IE');
  if (!ok) ok = await callFinding('EBAY-GB');
  if (!ok) ok = await callFinding('EBAY-US');
  return ok;
}

// -------------------- marketplaces: DoneDeal / Adverts --------------------
async function scrapeDoneDeal(results, query) {
  try {
    const url = `https://www.donedeal.ie/search?q=${encodeURIComponent(query)}`;
    const r = await axios.get(url, {
      headers: { 'User-Agent': 'Mozilla/5.0 (compatible; ProductScanner/1.0)' },
      timeout: 9000
    });
    const $ = cheerio.load(r.data);
    $('[data-testid="search-result"], .ad, article').slice(0, 12).each((i, el) => {
      const title = trimTitle($(el).find('[data-testid="search-result-title"], .ad-title, h3').first().text());
      const priceText = $(el).find('[data-testid="search-result-price"], .ad-price, .price').first().text()
        .replace(/[^0-9.,]/g, '').replace(',', '.');
      const price = priceText ? parseFloat(priceText) : null;
      let href = $(el).find('a').attr('href') || '';
      if (href && !/^https?:\/\//i.test(href)) href = 'https://www.donedeal.ie' + href;
      const link = sanitizeUrl(href);
      if (title) results.push({ title, source: 'DoneDeal', price, currency: 'EUR', url: link });
    });
  } catch {/* ignore */}
}

async function scrapeAdverts(results, query) {
  try {
    const url = `https://www.adverts.ie/for-sale/q_${encodeURIComponent(query)}/`;
    const r = await axios.get(url, {
      headers: { 'User-Agent': 'Mozilla/5.0 (compatible; ProductScanner/1.0)' },
      timeout: 9000
    });
    const $ = cheerio.load(r.data);
    $('.srp-listing, .item-info, article').slice(0, 12).each((i, el) => {
      const title = trimTitle($(el).find('.heading, h3, a[title]').first().text());
      const priceText = $(el).find('.price, .amount').first().text()
        .replace(/[^0-9.,]/g, '').replace(',', '.');
      const price = priceText ? parseFloat(priceText) : null;
      let href = $(el).find('a').attr('href') || '';
      if (href && !/^https?:\/\//i.test(href)) href = 'https://www.adverts.ie' + href;
      const link = sanitizeUrl(href);
      if (title) results.push({ title, source: 'Adverts.ie', price, currency: 'EUR', url: link });
    });
  } catch {/* ignore */}
}

// -------------------- /api/search --------------------
app.post('/api/search', async (req, res) => {
  try {
    const { query } = req.body;
    if (!query) return res.status(400).json({ error: 'query required' });

    const results = [];
    let used = false;

    // eBay (GB Browse)
    used = await callEbayBrowse(results, query);

    // Finding fallback (IE/GB -> US)
    if (!used || results.length < 4) {
      await callEbayFinding(results, query);
    }

    // Local Irish sources
    await scrapeDoneDeal(results, query);
    await scrapeAdverts(results, query);

    // de-duplicate & clean
    const dedup = [];
    const seen = new Set();
    for (const r of results) {
      const key = (r.title || '') + '|' + (r.price || '') + '|' + (r.source || '');
      if (!seen.has(key) && r.title) { seen.add(key); dedup.push(r); }
    }

    return res.json({
      regionNote: 'Irish & British marketplaces',
      results: dedup
    });
  } catch (err) {
    console.error('Search error:', err);
    return res.status(500).json({ error: err.message || 'search error' });
  }
});

// -------------------- start --------------------
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log('Server listening on', PORT));
