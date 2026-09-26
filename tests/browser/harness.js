// A tiny in-page test runner. tests/run-browser.mjs opens a page, waits for window.__results and
// fails the process if any test failed. Opening the page by hand shows the same results.

const tests = [];

export function test(name, fn) {
  tests.push({ name, fn });
}

export function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

/** max |a - b| over two arrays, and the index where it happens. */
export function maxAbsDiff(a, b) {
  if (a.length !== b.length) throw new Error(`length ${a.length} != ${b.length}`);
  let worst = 0;
  let at = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (!(d <= worst)) {
      worst = d;
      at = i;
    }
  }
  return { diff: worst, at };
}

/** Deterministic normal-ish random numbers for test data. */
export function rng(seed = 1) {
  let a = seed >>> 0;
  const u = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), a | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return () => (u() + u() + u() + u() - 2) * 1.7; // Irwin-Hall, variance ~1
}

export async function run(title) {
  const out = document.getElementById('out') || document.body.appendChild(document.createElement('pre'));
  const log = (s) => {
    out.textContent += s + '\n';
    console.log(s);
  };
  log(`# ${title}`);
  const results = [];
  for (const t of tests) {
    const t0 = performance.now();
    try {
      const info = await t.fn();
      const ms = performance.now() - t0;
      results.push({ name: t.name, ok: true, ms, info });
      log(`ok   ${t.name}${info ? '  ' + info : ''}  (${ms.toFixed(0)} ms)`);
    } catch (e) {
      results.push({ name: t.name, ok: false, error: String(e && e.stack ? e.stack : e) });
      log(`FAIL ${t.name}: ${e && e.message ? e.message : e}`);
    }
  }
  const failed = results.filter((r) => !r.ok).length;
  log(failed ? `${failed} of ${results.length} failed` : `all ${results.length} passed`);
  window.__results = results;
  return results;
}
