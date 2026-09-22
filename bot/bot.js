// Puts the model on WhatsApp.
//
// Baileys signs in as a real account, so the bot can sit in an ordinary group.
// That is not something WhatsApp's terms allow, and numbers running bots do get
// banned, so sign in with a spare number rather than your own.
//
//   npm install && node bot.js          # then scan the QR code once
//
// Environment:
//   REPLIER      who the model answers as, a name or an initial (default: the
//                busiest speaker in the corpus)
//   SPEAKER      who it treats you as (default: the runner-up)
//   TEMPERATURE  0.8 by default; lower is more predictable
//   ALLOW        comma-separated group ids it will answer in. Nothing else is
//                answered: with none set the bot is silent, and private chats
//                are never answered at all.
//   MENTION_ONLY set to 1 to answer only messages that mention the bot, rather
//                than everything said in an allowed group
import makeWASocket, {
  DisconnectReason,
  useMultiFileAuthState,
} from '@whiskeysockets/baileys';
import qrcode from 'qrcode-terminal';

import { Conversation, loadModel, speakerId } from './engine.js';
import { cleanText, shouldAnswer, textOf } from './routing.js';

const TEMPERATURE = Number(process.env.TEMPERATURE ?? 0.8);
const ALLOW = (process.env.ALLOW ?? '').split(',').map((s) => s.trim()).filter(Boolean);
const MENTION_ONLY = process.env.MENTION_ONLY === '1';
// Lets you test from the same account you linked: your own messages are
// ignored unless they start with this.
const SELF_TRIGGER = process.env.SELF_TRIGGER ?? '.gpt';
const VERBOSE = process.env.VERBOSE === '1';

const model = await loadModel();
const replier = speakerId(model, process.env.REPLIER) ?? model.senders.bot;
const speaker = speakerId(model, process.env.SPEAKER) ?? model.senders.you;
const nameOf = (id) => model.speakers.find((s) => s.id === id)?.name ?? id;

console.log(`model: ${model.corpus}, ${model.chars.length} characters of vocabulary`);
console.log(`answering as ${nameOf(replier)} (${replier}), hearing you as ${nameOf(speaker)} (${speaker})`);
console.log(
  ALLOW.length
    ? `answering in: ${ALLOW.join(', ')}${MENTION_ONLY ? ' (only when mentioned)' : ' (every message)'}`
    : 'no groups allowed yet — nothing will be answered. Set ALLOW.',
);
console.log('private chats are never answered');

const chats = new Map(); // one transcript per conversation
const mine = new Set(); // ids of messages this bot sent, so it never answers itself

function remember(id) {
  if (!id) return;
  mine.add(id);
  // The set only has to cover what could still arrive back at us
  if (mine.size > 500) mine.delete(mine.values().next().value);
}

// The ids you need for ALLOW are not guessable, so print them on sign-in.
async function listGroups(sock) {
  try {
    const groups = Object.values(await sock.groupFetchAllParticipating());
    if (!groups.length) {
      console.log('this account is in no groups');
      return;
    }
    console.log('\ngroups — pass one of these ids to ALLOW:');
    for (const group of groups) {
      const mark = ALLOW.includes(group.id) ? '*' : ' ';
      console.log(`  ${mark} ${group.id}  ${group.subject}`);
    }
    console.log('');
  } catch (error) {
    // Not fatal: VERBOSE=1 prints the id of every chat a message arrives in
    console.log('could not list groups:', error.message);
  }
}

async function start() {
  const { state, saveCreds } = await useMultiFileAuthState('auth');
  const sock = makeWASocket({ auth: state, printQRInTerminal: false });

  sock.ev.on('creds.update', saveCreds);

  sock.ev.on('connection.update', async ({ connection, lastDisconnect, qr }) => {
    if (qr) {
      console.log('\nScan this with WhatsApp > Linked devices:\n');
      qrcode.generate(qr, { small: true });
    }
    if (connection === 'open') {
      console.log('connected. waiting for messages.');
      await listGroups(sock);
    }
    if (connection === 'close') {
      const status = lastDisconnect?.error?.output?.statusCode;
      if (status === DisconnectReason.loggedOut) {
        console.log('logged out; delete auth/ and scan again');
        return;
      }
      console.log('connection dropped, reconnecting…');
      start();
    }
  });

  sock.ev.on('messages.upsert', async ({ messages, type }) => {
    if (type !== 'notify') return;

    for (const message of messages) {
      const selfId = sock.user?.id ?? '';
      const route = {
        selfId,
        allow: ALLOW,
        mentionOnly: MENTION_ONLY,
        selfTrigger: SELF_TRIGGER,
        isOwnReply: (m) => mine.has(m.key?.id),
      };
      const answering = shouldAnswer(message, route);
      if (VERBOSE) {
        console.log(`[${answering ? 'answer' : 'ignore'}] ${message.key.remoteJid}  ${textOf(message).slice(0, 40)}`);
      }
      if (!answering) continue;

      const said = cleanText(textOf(message), SELF_TRIGGER);
      if (!said) continue;

      const chatId = message.key.remoteJid;
      if (!chats.has(chatId)) {
        chats.set(chatId, new Conversation(model, { from: speaker, to: replier, temperature: TEMPERATURE }));
      }

      await sock.sendPresenceUpdate('composing', chatId);
      const reply = await chats.get(chatId).reply(said);
      await sock.sendPresenceUpdate('paused', chatId);

      if (reply) {
        const sent = await sock.sendMessage(chatId, { text: reply }, { quoted: message });
        remember(sent?.key?.id);
        console.log(`${chatId}  <- ${said}\n${chatId}  -> ${reply}`);
      }
    }
  });
}

start();
