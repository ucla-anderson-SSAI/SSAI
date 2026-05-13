import dotenv from 'dotenv';
import express from 'express';
import { readFile } from 'node:fs/promises';
import { readdirSync, existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { GoogleGenAI } from '@google/genai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

dotenv.config({ path: path.join(__dirname, 'local.env') });

const app = express();
const port = process.env.PORT || 3000;
const modelName = process.env.GEMINI_MODEL || 'gemini-2.5-flash';

app.use(express.json({ limit: '1mb' }));
app.use(express.static(path.join(__dirname, 'public')));

app.get('/api/health', (_req, res) => {
  res.json({
    ok: true,
    model: modelName,
    hasGeminiKey: Boolean(process.env.GEMINI_API_KEY)
  });
});

app.post('/api/generate', async (req, res) => {
  try {
    if (!process.env.GEMINI_API_KEY) {
      return res.status(400).json({
        error: 'Missing GEMINI_API_KEY. Add it to local.env locally and to Railway variables before deployment.'
      });
    }

    const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
    const {
      appName = 'MBA AI Assistant',
      audience = 'MBA students',
      task = 'help the user reason through a business question',
      question = '',
      useKnowledgeBase = false
    } = req.body || {};

    if (!question.trim()) {
      return res.status(400).json({ error: 'Please enter a user question or task.' });
    }

    const context = useKnowledgeBase ? await retrieveContext(question) : '';
    const prompt = buildPrompt({ appName, audience, task, question, context });

    const response = await ai.models.generateContent({
      model: modelName,
      contents: prompt
    });

    res.json({
      answer: response.text,
      usedKnowledgeBase: Boolean(context),
      model: modelName
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({
      error: error.message || 'The model call failed. Check your API key, network, and server logs.'
    });
  }
});

function buildPrompt({ appName, audience, task, question, context }) {
  const sourceBlock = context
    ? `Use this source material when it is relevant. If the source does not answer the question, say so clearly.\\n\\nSOURCE MATERIAL:\\n${context}\\n\\n`
    : '';
  const systemPrompt = `You are ${appName}, an AI assistant for ${audience}.

Your job is to ${task}.

Keep the answer useful, specific, and concise.`;

  return `${sourceBlock}${systemPrompt}

User request:
${question}

Respond with:
1. A direct answer
2. The reasoning or assumptions that matter
3. A concrete next step the user could take`;
}

async function retrieveContext(query) {
  const docsDir = path.join(__dirname, 'docs');
  if (!existsSync(docsDir)) return '';

  const files = readdirSync(docsDir).filter((name) => /\\.(md|txt)$/i.test(name));
  const chunks = [];

  for (const file of files) {
    const text = await readFile(path.join(docsDir, file), 'utf8');
    for (const chunk of chunkText(text)) {
      chunks.push({ file, chunk, score: scoreChunk(query, chunk) });
    }
  }

  return chunks
    .filter((item) => item.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 3)
    .map((item) => `[${item.file}]\\n${item.chunk}`)
    .join('\\n\\n---\\n\\n');
}

function chunkText(text) {
  return text
    .split(/\\n\\s*\\n/g)
    .map((chunk) => chunk.trim())
    .filter(Boolean)
    .slice(0, 30);
}

function scoreChunk(query, chunk) {
  const queryTerms = new Set(
    query
      .toLowerCase()
      .split(/[^a-z0-9]+/)
      .filter((term) => term.length > 3)
  );

  if (!queryTerms.size) return 0;
  const lowerChunk = chunk.toLowerCase();
  let score = 0;
  for (const term of queryTerms) {
    if (lowerChunk.includes(term)) score += 1;
  }
  return score;
}

app.listen(port, () => {
  console.log(`App running on http://localhost:${port}`);
});
