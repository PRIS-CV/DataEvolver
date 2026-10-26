import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, access } from 'node:fs/promises';
import { resolve } from 'node:path';
import vm from 'node:vm';
import { root, publicAsset } from './build.mjs';

test('both homepages publish every referenced asset and resolve local section links', async () => {
  for (const file of ['index.html', 'index_zh.html']) {
    const html = await readFile(resolve(root, 'frontend', file), 'utf8');
    const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]);
    assert.equal(new Set(ids).size, ids.length, `${file}: no duplicate ids`);
    for (const [, value] of html.matchAll(/\b(?:src|href)="([^"]+)"/g)) {
      if (value.startsWith('#')) {
        if (value.length > 1) assert.ok(ids.includes(value.slice(1)), `${file}: ${value}`);
      } else if (!/^[a-z]+:/i.test(value)) {
        const path = value.split(/[?#]/)[0];
        await access(resolve(root, 'frontend', path));
        if (!path.endsWith('/')) assert.ok(publicAsset(path), `${path} survives the build filter`);
      }
    }
    for (const name of ['datarsi-method.jpg', 'datarsi-occlusion-repair.png', 'datarsi-qualitative.png']) {
      assert.ok(html.includes(`images/paper/${name}`));
    }
    for (const id of ['publications', 'pipeline', 'research', 'worldmodel', 'evolution', 'results', 'gallery', 'dataset', 'start', 'citation', 'recognition']) {
      assert.ok(ids.includes(id), `${file}: keeps ${id}`);
    }
    assert.match(html, /ICASSP 2027/);
    assert.match(html, /2\.370 ± 0\.046°/);
    assert.match(html, /Rollback/);
    for (const [, script] of html.matchAll(/<script>([\s\S]*?)<\/script>/g)) new vm.Script(script);
  }
  new vm.Script(await readFile(resolve(root, 'frontend/homepage.js'), 'utf8'));
});

test('reading position ignores the Trace route and advances between local sections', async () => {
  const source = await readFile(resolve(root, 'frontend/homepage.js'), 'utf8');
  const code = source.slice(source.indexOf('var sectionLinks ='), source.indexOf('var progress ='));
  const names = ['publications', 'pipeline', 'research', 'evolution', 'results', 'gallery', 'start'];
  const links = names.map(id => ({
    attrs: { href: '#' + id },
    getAttribute(key) { return this.attrs[key]; },
    setAttribute(key, value) { this.attrs[key] = value; },
    removeAttribute(key) { delete this.attrs[key]; },
  }));
  const sections = names.map((id, i) => ({ id, top: 500 + i * 1000, getBoundingClientRect() { return { top: this.top }; } }));
  const trace = { attrs: { href: 'traces/?lang=en' } };
  let observe;
  function IntersectionObserver(callback) {
    observe = callback;
    this.observe = section => assert.ok(section);
  }
  vm.runInNewContext(code, {
    window: { innerHeight: 1000, IntersectionObserver }, IntersectionObserver,
    navLinks: { querySelectorAll(selector) {
      assert.equal(selector, 'a[href^="#"]');
      return [...links, trace].filter(link => link.attrs.href.startsWith('#'));
    } },
    document: { querySelector: selector => sections.find(section => '#' + section.id === selector) },
  });
  const current = () => links.filter(link => link.attrs['aria-current']).map(link => link.attrs.href);
  observe();
  assert.deepEqual(current(), []);
  sections[0].top = 64;
  observe();
  assert.deepEqual(current(), ['#publications']);
  sections[1].top = 64;
  observe();
  assert.deepEqual(current(), ['#pipeline']);
  sections.forEach(section => { section.top = -100; });
  observe();
  assert.deepEqual(current(), ['#start']);
});
