# TinyGPT on WhatsApp

Signs in as a real account, so the bot can sit in an ordinary group chat.

**Use a spare number.** This is not something WhatsApp's terms allow, and numbers
running bots do get banned.

## Setup

```bash
python -m scripts.export_web --corpus jaggers.txt   # from the repo root, once
cd bot
npm install
node bot.js                                         # scan the QR code with the spare phone
```

The login is saved in `auth/`, so the QR is a one-time step. Delete that
directory to sign out.

## Settings

All optional, all environment variables:

| | |
| --- | --- |
| `REPLIER` | who it answers as — a name or an initial (default: the busiest speaker) |
| `SPEAKER` | who it treats you as (default: the runner-up) |
| `TEMPERATURE` | 0.8 by default; lower is more predictable |
| `ALLOW` | comma-separated group ids it may answer in — nothing else is answered |
| `MENTION_ONLY` | `1` to answer only messages that mention it, rather than everything |
| `SELF_TRIGGER` | prefix that also triggers it in mention-only mode (default `.gpt`) |
| `VERBOSE` | `1` to log every message it sees and whether it answered |

It answers every message in the groups listed in `ALLOW`, and nothing anywhere
else. **Private chats are never answered**, and with no `ALLOW` set the bot
stays completely silent.

Running on your own number, the bot cannot tell your typing from its own
replies — both are "from me" — so it remembers the id of every message it
sends and never answers one of those. That is what keeps it from talking to
itself.

On sign-in it prints every group the account is in, with the id to pass to
`ALLOW` and a `*` beside the ones already allowed:

```
groups — pass one of these ids to ALLOW:
    120363000000000000@g.us  Jaggers
  * 120363111111111111@g.us  Test
```

If that listing fails, `VERBOSE=1` logs the chat id of every message that
arrives, answered or not.

```bash
REPLIER=Gómez ALLOW=120363000000000000@g.us node bot.js
```

## What to expect

The model sees 64 characters of context, which is about two messages. It cannot
follow a thread in a busy group — it answers the last thing said and nothing
before it. Replies take a few seconds, since generation runs on the CPU at
roughly eleven characters a second.

## Files

| | |
| --- | --- |
| `engine.js` | loads the model and keeps one transcript per chat — no WhatsApp |
| `routing.js` | decides which messages deserve an answer |
| `bot.js` | the connection itself |
