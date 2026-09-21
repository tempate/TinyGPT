// Wires the model up to a chat box. See tinygpt.js for the model itself.
import { buildModel, generate } from './tinygpt.js';

const MAX_REPLY = 120;

export async function mount(root, baseUrl = '.') {
  const log = root.querySelector('.tg-log');
  const form = root.querySelector('.tg-form');
  const input = form.querySelector('input');
  const button = form.querySelector('button');
  const status = root.querySelector('.tg-status');
  const tempInput = root.querySelector('[name="temperature"]');
  const tempValue = root.querySelector('.tg-temp-value');
  const senderPick = root.querySelector('.tg-sender');
  const replierPick = root.querySelector('.tg-replier');

  const say = (who, text, you = false) => {
    const p = document.createElement('p');
    p.className = `tg-msg${you ? ' is-you' : ''}`;
    p.innerHTML = '<span class="tg-who"></span>';
    p.firstChild.textContent = who ? `${who}: ` : '';
    p.append(document.createTextNode(text));
    log.append(p);
    log.scrollTop = log.scrollHeight;
    return p;
  };

  status.textContent = 'loading the model…';
  let model;
  try {
    const [manifest, buffer] = await Promise.all([
      fetch(`${baseUrl}/model.json`).then((r) => r.json()),
      fetch(`${baseUrl}/weights.bin`).then((r) => r.arrayBuffer()),
    ]);
    model = buildModel(manifest, buffer);
  } catch (err) {
    status.textContent = `could not load the model: ${err.message}`;
    return;
  }

  // The dropdowns hold one entry per person in the corpus, busiest first
  const { speakers } = model;
  const nameOf = new Map(speakers.map((s) => [s.id, s.name]));
  for (const pick of [senderPick, replierPick]) {
    for (const { id, name, messages } of speakers) {
      const option = document.createElement('option');
      option.value = id;
      option.textContent = `${name} (${messages.toLocaleString()})`;
      pick.append(option);
    }
    pick.disabled = speakers.length === 0;
  }
  senderPick.value = model.senders.you;
  replierPick.value = model.senders.bot;

  status.textContent = `${model.chars.length} characters of vocabulary, ${model.config.num_layers} layers. Say something.`;
  button.disabled = false;
  input.disabled = false;

  // Only characters the corpus contained can be encoded
  const keepKnown = (text) => [...text].filter((c) => model.stoi.has(c)).join('');

  let transcript = '';
  let busy = false;

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (busy) return;

    const typed = keepKnown(input.value.trim());
    if (!typed) return;

    const from = senderPick.value;
    const to = replierPick.value;

    input.value = '';
    say(nameOf.get(from) ?? from, typed, true);

    busy = true;
    button.disabled = true;
    status.textContent = 'thinking…';

    const bubble = say(nameOf.get(to) ?? to, '');
    bubble.classList.add('tg-cursor');
    const text = document.createTextNode('');
    bubble.append(text);

    // What the model actually sees: initials, and the replier's prefix left
    // dangling so the next character it writes is the start of their message
    transcript += `${from ? `${from}: ` : ''}${typed}\n${to ? `${to}: ` : ''}`;
    let reply = '';
    await generate(model, transcript.slice(-model.config.block_size), {
      maxChars: MAX_REPLY,
      temperature: Number(tempInput.value),
      onChar: async (ch) => {
        if (ch === '\n') return false; // one message, then stop
        reply += ch;
        text.textContent = reply;
        log.scrollTop = log.scrollHeight;
        await new Promise(requestAnimationFrame); // let the page paint
      },
    });

    transcript = `${transcript}${reply}\n`.slice(-2 * model.config.block_size);
    bubble.classList.remove('tg-cursor');
    busy = false;
    button.disabled = false;
    status.textContent = 'your turn.';
    input.focus();
  });

  tempInput.addEventListener('input', () => {
    tempValue.textContent = Number(tempInput.value).toFixed(2);
  });
  tempValue.textContent = Number(tempInput.value).toFixed(2);
}
