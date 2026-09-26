// Run the in-browser test pages under headless Chromium and report the results.
//   node tests/run-browser.mjs kernels parity        (default)
//   SMOLLM_DIR=/path/to/SmolLM2-135M-Instruct node tests/run-browser.mjs smollm
// Uses SwiftShader (CPU Vulkan) so it runs on machines without a GPU; set LLAMA_WGSL_GPU=1 to
// let Chromium pick a real adapter instead.
import { chromium } from 'playwright';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { serve } from '../tools/serve.mjs';

const suites = process.argv.slice(2).length ? process.argv.slice(2) : ['kernels', 'parity'];
const root = resolve(fileURLToPath(new URL('..', import.meta.url)));
const mounts = {};
if (process.env.SMOLLM_DIR) mounts['/models/smollm2-135m'] = resolve(process.env.SMOLLM_DIR);
if (suites.includes('smollm') && !process.env.SMOLLM_DIR) {
  console.error('the smollm suite needs SMOLLM_DIR=<directory with config.json, tokenizer.json, model.safetensors>');
  process.exit(2);
}

const args = ['--enable-unsafe-webgpu', '--enable-features=Vulkan'];
if (!process.env.LLAMA_WGSL_GPU) {
  args.push('--use-vulkan=swiftshader', '--use-webgpu-adapter=swiftshader', '--disable-vulkan-surface', '--use-angle=swiftshader');
}

const server = await serve({ root, mounts });
const port = server.address().port;
const browser = await chromium.launch({ headless: true, args });
let failed = 0;
try {
  for (const suite of suites) {
    const page = await browser.newPage();
    page.on('console', (m) => console.log(m.text()));
    page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`));
    await page.goto(`http://127.0.0.1:${port}/tests/browser/${suite}.html`);
    await page.waitForFunction(() => window.__results, null, { timeout: 60 * 60 * 1000, polling: 500 });
    const results = await page.evaluate(() => window.__results);
    for (const r of results.filter((x) => !x.ok)) console.log(`\n--- ${suite}: ${r.name}\n${r.error}`);
    failed += results.filter((r) => !r.ok).length;
    await page.close();
  }
} finally {
  await browser.close();
  server.close();
}
process.exit(failed ? 1 : 0);
