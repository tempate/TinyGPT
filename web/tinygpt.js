// A character-level GPT running in the browser.
//
// This mirrors core/ in the Python repo: the same four blocks of masked
// self-attention and feed-forward, the same weights, no dependencies. Shapes
// follow PyTorch, so a Linear's weight is (out, in) and y = x @ W.T + b.

const EPS = 1e-5; // torch.nn.LayerNorm's default

export function buildModel(manifest, buffer) {
  const weights = {};
  let offset = 0;
  for (const { name, shape } of manifest.tensors) {
    const size = shape.reduce((a, b) => a * b, 1);
    weights[name] = new Float32Array(buffer, offset * 4, size);
    offset += size;
  }

  const chars = manifest.chars;
  const stoi = new Map(chars.map((c, i) => [c, i]));
  const senders = manifest.senders ?? { you: '', bot: '' };
  const speakers = manifest.speakers ?? [];
  return { config: manifest.config, chars, stoi, senders, speakers, corpus: manifest.corpus, weights };
}

export function encode(model, text) {
  const out = [];
  for (const ch of text) if (model.stoi.has(ch)) out.push(model.stoi.get(ch));
  return out;
}

export const decode = (model, tokens) => tokens.map((i) => model.chars[i]).join("");

function layerNorm(out, x, T, C, weight, bias) {
  for (let t = 0; t < T; t++) {
    const row = t * C;
    let mean = 0;
    for (let c = 0; c < C; c++) mean += x[row + c];
    mean /= C;
    let variance = 0;
    for (let c = 0; c < C; c++) {
      const d = x[row + c] - mean;
      variance += d * d;
    }
    const scale = 1 / Math.sqrt(variance / C + EPS);
    for (let c = 0; c < C; c++) out[row + c] = (x[row + c] - mean) * scale * weight[c] + bias[c];
  }
}

// out(T, dout) = x(T, din) @ W(dout, din).T + b
function linear(out, x, T, din, dout, W, b) {
  for (let t = 0; t < T; t++) {
    const xrow = t * din;
    const orow = t * dout;
    for (let o = 0; o < dout; o++) {
      const wrow = o * din;
      let sum = b ? b[o] : 0;
      for (let i = 0; i < din; i++) sum += x[xrow + i] * W[wrow + i];
      out[orow + o] = sum;
    }
  }
}

function softmaxInPlace(v, n) {
  let max = -Infinity;
  for (let i = 0; i < n; i++) if (v[i] > max) max = v[i];
  let total = 0;
  for (let i = 0; i < n; i++) {
    v[i] = Math.exp(v[i] - max);
    total += v[i];
  }
  for (let i = 0; i < n; i++) v[i] /= total;
}

// Returns the logits for the final position only: that is all sampling needs.
export function forward(model, tokens) {
  const { block_size: B, embd_dim: C, num_heads: H, num_layers: L,
          head_size: hs, vocab_size: V } = model.config;
  const w = model.weights;
  const T = tokens.length;
  if (T > B) throw new Error(`context of ${T} exceeds block_size ${B}`);

  // Token and position embeddings
  const x = new Float32Array(T * C);
  const tokEmb = w["token_embedding_table.weight"];
  const posEmb = w["position_embedding_table.weight"];
  for (let t = 0; t < T; t++)
    for (let c = 0; c < C; c++) x[t * C + c] = tokEmb[tokens[t] * C + c] + posEmb[t * C + c];

  const normed = new Float32Array(T * C);
  const attnOut = new Float32Array(T * C);
  const projOut = new Float32Array(T * C);
  const q = new Float32Array(T * hs);
  const k = new Float32Array(T * hs);
  const v = new Float32Array(T * hs);
  const scores = new Float32Array(T);
  const hidden = new Float32Array(T * 4 * C);
  const scale = 1 / Math.sqrt(hs);

  for (let l = 0; l < L; l++) {
    const b = `blocks.${l}`;

    layerNorm(normed, x, T, C, w[`${b}.ln1.weight`], w[`${b}.ln1.bias`]);
    for (let h = 0; h < H; h++) {
      const head = `${b}.attn.heads.${h}`;
      linear(q, normed, T, C, hs, w[`${head}.query.weight`], null);
      linear(k, normed, T, C, hs, w[`${head}.key.weight`], null);
      linear(v, normed, T, C, hs, w[`${head}.value.weight`], null);

      for (let t = 0; t < T; t++) {
        // Causal mask: position t may only attend to 0..t
        for (let s = 0; s <= t; s++) {
          let dot = 0;
          for (let i = 0; i < hs; i++) dot += q[t * hs + i] * k[s * hs + i];
          scores[s] = dot * scale;
        }
        softmaxInPlace(scores, t + 1);

        const out = t * C + h * hs;
        for (let i = 0; i < hs; i++) attnOut[out + i] = 0;
        for (let s = 0; s <= t; s++) {
          const p = scores[s];
          for (let i = 0; i < hs; i++) attnOut[out + i] += p * v[s * hs + i];
        }
      }
    }
    linear(projOut, attnOut, T, C, C, w[`${b}.attn.proj.weight`], w[`${b}.attn.proj.bias`]);
    for (let i = 0; i < T * C; i++) x[i] += projOut[i]; // residual

    layerNorm(normed, x, T, C, w[`${b}.ln2.weight`], w[`${b}.ln2.bias`]);
    linear(hidden, normed, T, C, 4 * C, w[`${b}.ffwd.net.0.weight`], w[`${b}.ffwd.net.0.bias`]);
    for (let i = 0; i < T * 4 * C; i++) if (hidden[i] < 0) hidden[i] = 0; // ReLU
    linear(projOut, hidden, T, 4 * C, C, w[`${b}.ffwd.net.2.weight`], w[`${b}.ffwd.net.2.bias`]);
    for (let i = 0; i < T * C; i++) x[i] += projOut[i]; // residual
  }

  layerNorm(normed, x, T, C, w["ln_f.weight"], w["ln_f.bias"]);

  const last = normed.subarray((T - 1) * C, T * C);
  const logits = new Float32Array(V);
  linear(logits, last, 1, C, V, w["lm_head.weight"], w["lm_head.bias"]);
  return logits;
}

export function sample(logits, temperature, random = Math.random) {
  const probs = Float32Array.from(logits, (z) => z / temperature);
  softmaxInPlace(probs, probs.length);
  let roll = random();
  for (let i = 0; i < probs.length; i++) {
    roll -= probs[i];
    if (roll <= 0) return i;
  }
  return probs.length - 1;
}

// Generates one character at a time, handing each to onChar so the page can
// show them as they arrive. Stops early if onChar returns false.
export async function generate(model, prompt, { maxChars = 200, temperature = 0.8, onChar } = {}) {
  const B = model.config.block_size;
  let tokens = encode(model, prompt);
  let produced = "";

  for (let n = 0; n < maxChars; n++) {
    const window = tokens.slice(Math.max(0, tokens.length - B));
    const next = sample(forward(model, window), temperature);
    const ch = model.chars[next];
    tokens.push(next);
    produced += ch;
    if (onChar && (await onChar(ch)) === false) break;
  }
  return produced;
}
