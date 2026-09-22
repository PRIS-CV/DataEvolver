'use strict';
import { loadTraces } from './data-source.js';
(async () => {
  const zh = new URLSearchParams(location.search).get('lang') === 'zh';
  const t = (en, cn) => zh ? cn : en;
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const sourceStatus = document.querySelector('#source-status');
  sourceStatus.textContent = t('Loading trace archive…', '正在加载轨迹记录…');
  const source = await loadTraces({ snapshot: window.TRACE_SHOWCASE, apiBase: window.TRACE_CONFIG?.apiBase });
  const data = source.data;
  sourceStatus.dataset.mode = source.mode;
  sourceStatus.textContent = source.mode === 'connected' ? t('Service connected · read-only archive', '服务已连接 · 只读归档') : source.mode === 'fallback' ? t('Service unavailable · showing bundled archive. Reload to retry.', '服务暂不可用 · 当前显示归档副本，可刷新重试。') : t('Public archive · no training controls', '公开归档 · 不含训练控制');
  const evidenceUrl = id => source.mode === 'connected' ? `${source.base}/api/traces${id ? '/'+id : ''}` : id ? `evidence/${id}.json` : 'traces.json';
  document.querySelector('.footer a').href = evidenceUrl();
  document.documentElement.lang = zh ? 'zh-Hans' : 'en';
  document.title = t('Repair Traces — DataEvolver', '修复轨迹 — DataEvolver');
  const language = document.querySelector('#language');
  language.href = zh ? '?lang=en' : '?lang=zh';
  language.textContent = zh ? 'English' : '中文';
  language.lang = zh ? 'en' : 'zh-Hans';
  const home = document.querySelector('#home');
  home.href = zh ? '../index_zh.html#evolution' : '../index.html#evolution';
  home.textContent = t('Project page ↗', '项目主页 ↗');
  document.querySelector('.brand').href = home.href;
  document.querySelector('.skip').textContent = t('Skip to content', '跳到正文');
  document.querySelector('#footer-note').textContent = t('DataEvolver · Historical traces, curated September 2026', 'DataEvolver · 历史轨迹，整理于 2026 年 9 月');
  const themeButton = document.querySelector('#theme');
  function themeLabel() {
    const dark = document.documentElement.dataset.theme === 'dark';
    themeButton.textContent = dark ? t('Light mode', '浅色模式') : t('Dark mode', '深色模式');
    themeButton.setAttribute('aria-label', themeButton.textContent);
  }
  themeLabel();
  themeButton.addEventListener('click', () => {
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem('de-theme', next); } catch (_) { /* Optional persistence. */ }
    themeLabel();
  });
  const labels = {
    c_front_back_anchors: t('Front / back anchors', '前后位置'),
    c_local_parts: t('Bottle-neck details', '瓶颈细节'),
    c_count_primary: t('Bottle count', '瓶子数量'),
    c_controlled_occlusion: t('Targeted occlusion', '指定遮挡')
  };
  const failed = r => r.failed_constraints.map(x => labels[x] || x).join(t(', ', '、'));
  function figure(r, caption, id) {
    return `<figure><a class="image-link" href="${esc(r.image)}" target="_blank" rel="noopener" aria-label="${esc(caption+' · '+t('open original in a new tab','新标签页查看原图'))}"><img src="${esc(r.image)}" width="${id === 'grounding' ? 910 : 512}" height="512" alt="${esc(caption)}" loading="lazy"></a><figcaption><span>${esc(caption)}</span><span>${t('Original image ↗','原图 ↗')}</span></figcaption></figure>`;
  }
  const fact = (label, value, note) => `<div class="fact"><dt>${label}</dt><dd>${value}<small>${note}</small></dd></div>`;
  const step = (title, text) => `<div><h3>${title}</h3><p>${text}</p></div>`;
  function shell(c, index) {
    const ground = c.id === 'grounding';
    const title = ground ? t('Bring the objects back to the ground.', '让悬浮物体重新落地。') : t('Recover the missing third bottle.', '找回缺失的第三个瓶子。');
    const description = ground
      ? t('A targeted 3D scene repair: the review flags floating objects, the controller records ground snapping, and render metadata confirms their lower positions.', '一次有执行证据的 3D 场景修复：评价指出物体悬浮，控制器记录接地操作，渲染元数据确认物体底部下降。')
      : t('A prompt-repair loop that recovers from a regression. Round 2 loses a bottle; round 3 restores three. Other constraints remain unresolved.', '一次从退步中恢复的提示词修复：第 2 轮少了一个瓶子，第 3 轮恢复为三个。其他约束仍有未解决项。');
    return `<section class="case" id="${c.id}" aria-labelledby="${c.id}-title"><div class="case-heading"><div><p class="case-number">0${index+1} / ${ground ? '3D SCENE REPAIR' : 'TEXT-TO-IMAGE REPAIR'} · ${c.date}</p><h2 id="${c.id}-title">${title}</h2><p class="case-description">${description}</p></div><span class="badge ${ground?'':'partial'}">${ground?t('Historical gate: accepted','历史 gate：通过'):t('Partial repair · not fully passed','局部修复 · 未完全通过')}</span></div>
      <div class="round-controls" role="group" aria-label="${t('Select a round','选择轮次')}">${c.rounds.map(r=>`<button type="button" data-case="${c.id}" data-round="${r.round}" aria-pressed="false">${r.round === 0 ? t('R0 · Initial','R0 · 初始生成') : `R${r.round} · ${ground?t('Ground snap','接地修复'):r.round===2?t('Regression','发生退步'):r.round===3?t('Count recovered','数量恢复'):t('First repair','首次修复')}`}</button>`).join('')}</div>
      <div id="${c.id}-panel"></div>
      <div class="evidence-links"><a href="${esc(evidenceUrl(c.id))}" download>${t('Download trace JSON ↓','下载轨迹 JSON ↓')}</a><a href="${esc(evidenceUrl(c.id))}" target="_blank" rel="noopener">${t('Inspect evidence ↗','查看证据 ↗')}</a></div>
      <p class="source-id">${esc(c.source_run)}${ground ? ' / '+esc(c.source_case) : ''}</p></section>`;
  }
  document.querySelector('#showcase').innerHTML = `<section class="hero"><div><p class="kicker">DataEvolver / Repair traces</p><h1>${t('Repair, made visible.', '修复轨迹。<br>每一步都有依据。')}</h1><p class="hero-description">${t('Follow the image, the review, and the action. A close look at what changed—and what still needs work.', '从原图到评价，从实际修改到下一轮结果。看清改变，也保留尚未解决的问题。')}</p><a class="primary-link" href="#grounding">${t('Explore the traces','浏览修复案例')} <span aria-hidden="true">↗</span></a></div><aside class="archive-card"><p class="kicker">${t('Inside the archive','本页收录')}</p><div class="archive-numbers"><strong>${data.cases.length}<small>${t('curated traces','精选轨迹')}</small></strong><strong>${data.cases.reduce((n,c)=>n+c.rounds.length,0)}<small>${t('original rounds','原始轮次')}</small></strong></div><p>${t('Original frames. Recorded actions. Open evidence.', '原始图像、修改记录与可下载证据。')}</p><ol><li>${t('Review findings','发现问题')}</li><li>${t('Actual changes','实际修改')}</li><li>${t('Images & decisions','结果与决定')}</li></ol></aside></section>
    <nav class="case-nav" aria-label="${t('Trace cases','轨迹案例')}"><a href="#grounding">01 ${t('Ground contact','接地修复')}</a><a href="#count">02 ${t('Bottle count','数量修复')}</a><a href="#method">${t('How to read the evidence','如何理解这些证据')}</a></nav>
    <p class="scope">${t('Curated examples, not a benchmark. These are archived scene and prompt-repair loops, not new Harness runs and not evidence of improved trained-model performance. All source frames are unchanged.', '这是精选案例，不是总体评测。它们来自历史场景与提示词修复循环，不是本次新跑的 Harness，也不代表训练后模型性能提升。所有原图保持不变。')}</p>
    ${data.cases.map(shell).join('')}
    <section class="method" id="method"><h2>${t('What the evidence can tell us.', '证据能说明什么。')}</h2><ul>
      <li>${t('Observation, proposed action, saved changes, render execution, and quality gain are different claims. This page keeps them separate.', '发现问题、建议动作、参数保存、渲染执行、质量改善是不同层面的证据，不能相互替代。')}</li>
      <li>${t('The scene trace has render-level ground-snap evidence, but no isolated shadow-effect proof. Its realized camera also moves slightly, so this is not a fixed-camera ablation.', '场景案例有渲染层面的接地记录，但没有独立的阴影改善证据。实际相机位置也略有变化，因此不是固定相机的单因素消融。')}</li>
      <li>${t('The bottle trace repairs one failed constraint, after an earlier regression. The full four-round history remains available; it is not monotonic progress or final success.', '瓶子案例在前轮退步后修复了一个失败约束。完整四轮都可查看，不能解释为持续进步或最终达标。')}</li>
      <li>${t('Downloads contain scoped structured excerpts, image hashes, and source-file hashes. Private host paths and raw model transcripts are omitted. Missing intermediate images disqualify score-only candidates.', '下载内容包含结构化摘录、图片哈希与源文件哈希。内部主机路径和原始模型文本不公开。只有分数、缺少中间图片的候选不纳入展示。')}</li>
    </ul><p>${t('No new model calls, rendering, training, or evaluator calibration were performed for this page.', '本页整理没有重新调用模型、渲染、训练或校准评测器。')}</p></section><section class="reuse"><div><p class="kicker">Build / Deploy</p><h2>${t('Run your own trace page.', '部署与复用')}</h2><p>${t('The frontend and read-only service are available in the repository, with npm commands and deployment examples.', '前端与只读服务代码已开放，包含 npm 命令、部署配置与使用说明。')}</p></div><a class="primary-link" href="https://github.com/PRIS-CV/DataEvolver/blob/main/deploy/traces/README.md" target="_blank" rel="noopener">${t('Deployment guide ↗','部署文档 ↗')}</a></section>`;
  document.querySelector('#showcase').setAttribute('aria-busy', 'false');

  function render(id, index) {
    const c = data.cases.find(x => x.id === id);
    const r = c.rounds[index], previous = c.rounds[Math.max(0,index-1)];
    const initial = index === 0;
    const ground = id === 'grounding';
    const caption = `${t('Round','第')} ${index}${t('',' 轮')}`;
    let content = `<div class="comparison ${initial?'initial':''}">${initial?'':figure(previous, `${t('Before · Round','修改前 · 第')} ${index-1}${t('',' 轮')}`, id)}${figure(r, `${initial?t('Initial · Round','初始生成 · 第'):t('After · Round','修改后 · 第')} ${index}${t('',' 轮')}`, id)}</div>
      <p class="round-note" role="status">${initial?t('Initial generation. No preceding repair is implied.','初始生成，没有对应的前轮修复。'):t('Adjacent rounds · uncropped source frames · click either image for full resolution.','相邻轮次对照 · 未裁剪原图 · 点击可查看完整分辨率。')} ${caption}</p>`;
    if (ground) {
      if (!initial) {
        content += `<details class="details" open><summary>${t('Inspect the contact area · same display crop', '查看接地细节 · 相同显示区域放大')}</summary><div class="content"><div class="comparison contact-detail">${c.rounds.map(x=>`<figure><a class="image-link" href="${esc(x.image)}" target="_blank" rel="noopener"><svg viewBox="330 190 230 125" role="img" aria-label="${t('Ground-contact detail, round','接地细节，第')} ${x.round}${t('',' 轮')}"><image href="${esc(x.image)}" width="910" height="512" /></svg></a><figcaption>R${x.round} · ${x.round?t('After ground snapping','接地后'):t('Before ground snapping','接地前')}</figcaption></figure>`).join('')}</div><p class="review-note">${t('Display-only magnification of the same 230 × 125 px region (x=330, y=190). Original files are unchanged. Realized camera positions differ, so this is not a pixel-aligned comparison.', '仅放大显示同一 230 × 125 像素区域（x=330，y=190），原文件未修改。实际相机位置略有变化，因此不是逐像素对齐对照。')}</p></div></details>`;
      }
      const record = c.execution_evidence.b[0];
      const gapBefore = (record.bottom_z_before-record.ground_z).toFixed(4);
      const gapAfter = (record.bottom_z_after-record.ground_z).toFixed(4);
      content += `<dl class="facts">${fact(t('Object B / ground clearance','物体 B / 离地间隙'),initial?gapBefore:`${gapBefore} → ${gapAfter}`,t('Native scene units; render metadata','原生场景单位；来自渲染元数据'))}${fact(t('Object XY & scale','物体 XY 与尺寸'),t('Unchanged','保持不变'),t('Only vertical placement is corrected','接地修改作用于竖直位置'))}${fact(t('Recorded gate outcome','记录中的 gate 结果'),initial?t('Needs repair','需要修复'):t('Accepted','通过'),`hybrid ${r.hybrid_score.toFixed(4)} / ${t('threshold','阈值')} 0.75`)}</dl>`;
      content += `<p class="review-note">${t('Complete repair chain: R0 → R1. The actions and outcome below occur after the initial frame.', '完整修复链：R0 → R1。以下修改与结果发生在初始图之后。')}</p><div class="process">${step(t('01 / Observation','01 / 发现问题'),t('Round 0 records floating_object, shadow_missing, and scene_light_mismatch. The review recommends ground snapping and stronger contact shadows.', '第 0 轮记录物体悬浮、阴影缺失和光照不匹配；评价建议接地与增强接触阴影。'))}${step(t('02 / Recorded changes','02 / 实际记录的修改'),t('Both ground-snap flags: 0 → 1. Contact-shadow and AO controls: 1 → 1.35. Spotlight angle: 45° → 38.25°. These are saved-state changes, not four separately proven effects.', '两物体 ground_snap：0 → 1；接触阴影与 AO 控制值：1 → 1.35；聚光角：45° → 38.25°。这表示保存的状态变化，不是四项效果都已得到证明。'))}${step(t('03 / Verified outcome','03 / 已有执行证据'),t('Render metadata records object B’s bottom moving from 0.28961 to 0.015059, above local ground 0.010059. The historical gate accepts round 1; the paired review prefers the current frame.', '渲染记录显示物体 B 底部从 0.28961 移至 0.015059，局部地面为 0.010059。历史 gate 接受第 1 轮，成对评价偏好当前图。'))}</div>`;
      content += `<p class="caveat">${t('Boundary: camera controls are unchanged, but the realized camera position changes. Shadow execution lists no touched lights. Do not attribute all visible change to one action or claim proven shadow improvement. Hybrid scores are not pure VLM scores; evaluation context differs across the first and later rounds.', '边界：相机控制值未变，但实际相机位置有变化；阴影执行记录中 lights_touched 为空。不能把全部视觉差异归因于单个动作，也不宣称阴影改善已证实。hybrid 不是纯 VLM 分，首轮与后续轮评价上下文也不同。')}</p>
      <details class="details"><summary>${t('Saved parameter diff & camera evidence','展开参数差异与相机证据')}</summary><div class="content"><div class="table-wrap" role="region" tabindex="0" aria-label="${t('Parameter differences','参数差异表')}"><table><thead><tr><th>${t('Parameter','参数')}</th><th>R0</th><th>R1</th></tr></thead><tbody>${c.parameter_diff.map(x=>`<tr><td><code>${esc(x.field)}</code></td><td>${esc(JSON.stringify(x.before))}</td><td>${esc(JSON.stringify(x.after))}</td></tr>`).join('')}</tbody></table></div><p class="review-note">${t('Realized camera position (scene coordinates):','实际相机位置（场景坐标）：')}<br>R0 <code>${esc(JSON.stringify(c.rounds[0].camera.location))}</code><br>R1 <code>${esc(JSON.stringify(c.rounds[1].camera.location))}</code></p></div></details>`;
    } else {
      const count = r.passed_constraints.length;
      content += `<dl class="facts">${fact(t('Bottle-count judgment','瓶子数量判断'),r.failed_constraints.includes('c_count_primary')?t('Fail · 2 visible','失败 · 可见 2 个'):t('Pass · 3 visible','通过 · 可见 3 个'),t('Archived review; visually inspectable','归档评价；可直接检查原图'))}${fact(t('Recorded constraints passed','记录中通过的约束'),`${count} / 9`,t('VLM checklist, not an independent audit','VLM 清单，不是独立审计'))}${fact(t('Remaining failures','仍未通过'),String(r.failed_constraints.length),esc(failed(r)))}</dl>`;
      const findings = initial ? t('The initial scene contains three bottles. The archived review flags the front/back anchors.', '初始场景有三个瓶子，归档评价指出前后位置约束未满足。') : t(`Round ${index-1} flags: ${failed(previous)}.`, `第 ${index-1} 轮指出：${failed(previous)}。`);
      const actionText = initial ? t('No repair yet. The first prompt expands the original request into a dataset-oriented generation prompt.', '尚未执行修复。初始提示词将原始要求扩写为数据集生成指令。') : t('The next prompt adds a failed-constraint repair checklist and an anti-regression checklist. It regenerates the entire image; it is not a pixel-preserving edit.', '下一轮提示词加入失败约束修复清单与防退步清单，重新生成整张图；不是保持原像素不变的局部编辑。');
      const outcome = index===3 ? t('Bottle count recovers from two to three. The archived failed-constraint count falls from two to one, but targeted occlusion still fails. The loop stops at its local-improvement criterion.', '瓶子从两个恢复为三个；归档失败约束从两项降为一项，但指定遮挡仍失败。循环按“局部改善”条件停止。') : index===2 ? t('Regression: neck-detail judgment improves, but bottle count and targeted occlusion fail. This failed round is preserved rather than hidden.', '发生退步：瓶颈细节判断改善，但数量和指定遮挡失败。这一失败轮次保留展示。') : index===1 ? t('Front/back anchors improve, but neck details regress. There is no net reduction in failed constraints.', '前后位置判断改善，但瓶颈细节退步，失败约束总数没有减少。') : t('Eight of nine constraints pass in the archived review; one needs repair.', '归档评价中九项通过八项，一项需要修复。');
      content += `<div class="process">${step(t('01 / Observation','01 / 发现问题'),esc(findings))}${step(t('02 / Prompt revision','02 / 提示词修改'),esc(actionText))}${step(t('03 / Result & decision','03 / 结果与决定'),esc(outcome))}</div><p class="caveat">${t('Partial recovery, not final success. Round 3 restores a count that round 0 already had. Occlusion remains wrong, and visible ruler markings also warrant review despite the archived no-text pass. Image regeneration does not establish appearance stability.', '这是局部恢复，不是最终成功。第 3 轮找回的是第 0 轮已有、后来丢失的数量；遮挡仍不正确，尺子上的可见标记也说明归档“无文字通过”需要复核。整图再生成不证明外观稳定性。')}</p>`;
      content += `<details class="details"><summary>${t('All rounds, including regressions','展开完整四轮记录（含退步）')}</summary><div class="content"><div class="table-wrap" role="region" tabindex="0" aria-label="${t('Complete round history','完整轮次记录')}"><table><thead><tr><th>${t('Round','轮次')}</th><th>${t('Passed','通过')}</th><th>${t('Failed constraints','失败约束')}</th><th>${t('Source image','原图')}</th></tr></thead><tbody>${c.rounds.map(x=>`<tr><td>R${x.round}</td><td>${x.passed_constraints.length}/9</td><td>${esc(failed(x))}</td><td><a href="${esc(x.image)}" target="_blank" rel="noopener">PNG ↗</a></td></tr>`).join('')}</tbody></table></div></div></details>
      <details class="details"><summary>${t('Exact prompt used for this round','展开本轮实际使用的完整提示词')}</summary><div class="content"><p class="review-note"><code>${esc(r.rewrite_reason)}</code></p><pre class="prompt" tabindex="0">${esc(r.prompt)}</pre></div></details>`;
    }
    document.querySelector(`#${id}-panel`).innerHTML=content;
    document.querySelectorAll(`[data-case="${id}"]`).forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.round)===index)));
  }
  document.querySelectorAll('[data-case]').forEach(button=>button.addEventListener('click',()=>render(button.dataset.case,Number(button.dataset.round))));
  render('grounding',1);
  render('count',3);
})().catch(() => {
  document.querySelector('#source-status').textContent = new URLSearchParams(location.search).get('lang') === 'zh' ? '轨迹记录无法加载，请刷新重试或下载下方 JSON。' : 'Trace archive could not be loaded. Reload or download the JSON below.';
  document.querySelector('#source-status').dataset.mode = 'fallback';
  document.querySelector('#showcase').setAttribute('aria-busy', 'false');
});
