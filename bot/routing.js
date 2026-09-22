// Deciding what to answer. Plain functions over plain message objects, so this
// is testable without a WhatsApp connection.

export function textOf(message) {
  const m = message.message ?? {};
  return m.conversation ?? m.extendedTextMessage?.text ?? m.imageMessage?.caption ?? '';
}

const isGroup = (chatId) => chatId.endsWith('@g.us');
const sameUser = (a, b) => a.split(':')[0].split('@')[0] === b.split(':')[0].split('@')[0];
const startsWithTrigger = (text, trigger) =>
  text.trimStart().toLowerCase().startsWith(trigger.toLowerCase());

/**
 * Answers only in the groups named in `allow`, and never in a private chat.
 *
 * `isOwnReply` says whether a message is one the bot itself sent. That is the
 * only thing separating the bot's output from the owner's own typing when both
 * come from the same linked account, and skipping it is what stops the bot
 * from answering itself in a loop.
 */
export function shouldAnswer(
  message,
  { selfId = '', allow = [], mentionOnly = false, selfTrigger = '', isOwnReply = () => false },
) {
  const chatId = message.key?.remoteJid ?? '';
  if (!chatId || chatId === 'status@broadcast') return false;

  // Private chats are off limits, whoever is writing
  if (!isGroup(chatId)) return false;
  if (!allow.includes(chatId)) return false;

  // Never reply to something the bot itself just said
  if (message.key.fromMe && isOwnReply(message)) return false;

  if (!mentionOnly) return true;

  if (selfTrigger && startsWithTrigger(textOf(message), selfTrigger)) return true;
  const mentioned = message.message?.extendedTextMessage?.contextInfo?.mentionedJid ?? [];
  return mentioned.some((jid) => sameUser(jid, selfId));
}

// Strips the @12345 of a mention, and the trigger word if one was used
export function cleanText(text, selfTrigger = '') {
  let out = text.replace(/@\d+/g, ' ');
  if (selfTrigger && startsWithTrigger(out, selfTrigger)) {
    out = out.trimStart().slice(selfTrigger.length);
  }
  return out.replace(/\s+/g, ' ').trim();
}
