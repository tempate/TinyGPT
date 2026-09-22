// The model half of the bot: no WhatsApp and no network, so it can be tested
// on its own. bot.js adds the messaging on top.
import { readFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { buildModel, generate } from '../web/tinygpt.js';

const WEB_DIR = join(dirname(fileURLToPath(import.meta.url)), '..', 'web');

export async function loadModel(dir = WEB_DIR) {
  const manifest = JSON.parse(await readFile(join(dir, 'model.json'), 'utf8'));
  const bytes = await readFile(join(dir, 'weights.bin'));
  return buildModel(
    manifest,
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
}

// One running transcript per chat, trimmed to what the model can still see.
export class Conversation {
  constructor(model, { from, to, temperature = 0.8, maxChars = 120 } = {}) {
    this.model = model;
    this.from = from ?? model.senders.you;
    this.to = to ?? model.senders.bot;
    this.temperature = temperature;
    this.maxChars = maxChars;
    this.transcript = '';
  }

  // Characters the corpus never contained have no encoding, so they are dropped
  keepKnown(text) {
    return [...text].filter((ch) => this.model.stoi.has(ch)).join('');
  }

  async reply(message) {
    const said = this.keepKnown(message.trim().replace(/\s+/g, ' '));
    if (!said) return '';

    this.transcript += `${this.from}: ${said}\n${this.to}: `;

    let line = '';
    await generate(this.model, this.transcript.slice(-this.model.config.block_size), {
      maxChars: this.maxChars,
      temperature: this.temperature,
      onChar: (ch) => {
        if (ch === '\n') return false; // one message, then stop
        line += ch;
      },
    });

    this.transcript = `${this.transcript}${line}\n`.slice(-2 * this.model.config.block_size);
    return line.trim();
  }
}

// Accepts either an initial ("G") or a name ("Gómez"), returns the initial.
export function speakerId(model, wanted) {
  if (!wanted) return null;
  const match = model.speakers.find(
    (s) => s.id === wanted || s.name.toLowerCase() === String(wanted).toLowerCase(),
  );
  return match ? match.id : null;
}
