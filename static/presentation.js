/* Progressive enhancement: the original image links remain usable without JS. */
(() => {
  const root = document.querySelector('.qy-home');
  if (!root || root.dataset.enhanced) return;
  root.dataset.enhanced = 'true';
  const dialog = root.querySelector('.qy-lightbox');
  const largeImage = dialog.querySelector('img');
  const dialogTitle = dialog.querySelector('h2');
  let opener;
  root.querySelectorAll('[data-fullscreen]').forEach(link => {
    link.addEventListener('click', event => {
      event.preventDefault();
      opener = link;
      largeImage.src = link.href;
      largeImage.alt = link.dataset.title;
      dialogTitle.textContent = link.dataset.title;
      dialog.classList.remove('qy-detail');
      dialog.querySelector('[data-zoom]').textContent = '查看细节';
      dialog.querySelector('[data-zoom]').setAttribute('aria-pressed','false');
      dialog.showModal();
    });
  });
  dialog.querySelector('[data-close]').addEventListener('click', () => dialog.close());
  dialog.querySelector('[data-zoom]').addEventListener('click', event => {
    const zoomed = dialog.classList.toggle('qy-detail');
    event.currentTarget.textContent = zoomed ? '适应屏幕' : '查看细节';
    event.currentTarget.setAttribute('aria-pressed',String(zoomed));
  });
  dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
  dialog.addEventListener('close', () => { if (opener?.isConnected) opener.focus({preventScroll:true}); });
  const cleanups = [];
  const svgNS = 'http://www.w3.org/2000/svg';
  const svgElement = (tag, attrs = {}, text) => {
    const element = document.createElementNS(svgNS, tag);
    Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, value));
    if (text !== undefined) element.textContent = text;
    return element;
  };
  const observeSize = (element, render) => {
    const observer = new ResizeObserver(() => {
      if (!root.isConnected) { observer.disconnect(); return; }
      if (element.clientWidth) render();
    });
    observer.observe(element);
    cleanups.push(() => observer.disconnect());
  };
  root.querySelectorAll('.qy-team-map').forEach((map, mapIndex) => {
    const edges = JSON.parse(map.dataset.edges);
    const svg = svgElement('svg', {'aria-hidden': 'true', class: 'qy-map-links'});
    map.prepend(svg);
    const draw = () => {
      svg.replaceChildren();
      svg.setAttribute('viewBox', `0 0 ${map.clientWidth} ${map.clientHeight}`);
      const definitions = svgElement('defs');
      const marker = svgElement('marker', {id: `qy-arrow-${mapIndex}`, viewBox: '0 0 8 8', refX: 7, refY: 4, markerWidth: 5, markerHeight: 5, orient: 'auto-start-reverse'});
      marker.append(svgElement('path', {d: 'M 1 1 L 7 4 L 1 7', fill: 'none', stroke: 'currentColor', 'stroke-width': 1.5}));
      definitions.append(marker); svg.append(definitions);
      edges.forEach(([from, to]) => {
        const a = map.querySelector(`[data-node="${from}"]`);
        const b = map.querySelector(`[data-node="${to}"]`);
        if (!a || !b) return;
        const horizontal = Math.abs(a.offsetTop - b.offsetTop) < 10;
        const direction = horizontal ? Math.sign(b.offsetLeft - a.offsetLeft) : Math.sign(b.offsetTop - a.offsetTop);
        const x1 = a.offsetLeft + a.offsetWidth / 2 + (horizontal ? direction * a.offsetWidth / 2 : 0);
        const y1 = a.offsetTop + a.offsetHeight / 2 + (horizontal ? 0 : direction * a.offsetHeight / 2);
        const x2 = b.offsetLeft + b.offsetWidth / 2 - (horizontal ? direction * (b.offsetWidth / 2 + 6) : 0);
        const y2 = b.offsetTop + b.offsetHeight / 2 - (horizontal ? 0 : direction * (b.offsetHeight / 2 + 6));
        const d = horizontal ? `M${x1},${y1} C${(x1+x2)/2},${y1} ${(x1+x2)/2},${y2} ${x2},${y2}` : `M${x1},${y1} C${x1},${(y1+y2)/2} ${x2},${(y1+y2)/2} ${x2},${y2}`;
        svg.append(svgElement('path', {d, class: (a.classList.contains('is-added') || b.classList.contains('is-added')) ? 'is-new' : '', 'marker-end': `url(#qy-arrow-${mapIndex})`}));
      });
    };
    map.querySelectorAll('[data-node]').forEach(node => node.addEventListener('click', () => {
      map.querySelectorAll('[data-node]').forEach(n => n.setAttribute('aria-pressed', String(n === node)));
      map.parentElement.querySelector('.qy-node-detail').textContent = node.dataset.detail;
    }));
    observeSize(map, draw);
  });
  const tabs = [...root.querySelectorAll('[role="tab"][data-case]')];
  const selectCase = tab => {
    tabs.forEach(button => { button.setAttribute('aria-selected', String(button === tab)); button.tabIndex = button === tab ? 0 : -1; });
    root.querySelectorAll('[data-case-panel]').forEach(panel => { panel.hidden = panel.dataset.casePanel !== tab.dataset.case; });
    root.querySelectorAll('[data-proof-panel]').forEach(panel => { panel.hidden = panel.dataset.proofPanel !== tab.dataset.case; });
  };
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => selectCase(tab));
    tab.addEventListener('keydown', event => {
      if (!['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) return;
      event.preventDefault();
      const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : (index + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
      selectCase(tabs[next]); tabs[next].focus({preventScroll:true});
    });
  });
  root.querySelectorAll('[data-case-panel]').forEach(panel => {
    panel.querySelectorAll('[data-mode]').forEach(button => button.addEventListener('click', () => {
      panel.querySelectorAll('[data-mode]').forEach(b => b.setAttribute('aria-pressed', String(b === button)));
      panel.querySelectorAll('[data-mode-panel]').forEach(p => { p.hidden = p.dataset.modePanel !== button.dataset.mode; });
    }));
  });
  root.querySelectorAll('[data-chart]').forEach(chart => {
    const data = JSON.parse(chart.dataset.chart);
    const surface = chart.querySelector('.qy-chart-canvas');
    const input = chart.querySelector('input[type="range"]');
    const svg = svgElement('svg', {'aria-hidden': 'true'});
    surface.append(svg);
    const draw = () => {
      const width = surface.clientWidth, height = 240;
      if (!width) return;
      svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
      svg.replaceChildren();
      const left = 48, right = width - 12, top = 14, bottom = height - 29;
      const values = data.series.flatMap(series => series.values);
      const min = Math.min(...values), max = Math.max(...values);
      const pad = (max - min || Math.max(Math.abs(min) * .01, 1)) * .12;
      const low = min - pad, high = max + pad;
      const x = i => left + i / Math.max(1, data.labels.length - 1) * (right - left);
      const y = value => bottom - (value - low) / (high - low) * (bottom - top);
      for (let i = 0; i < 4; i++) {
        const value = low + (high - low) * i / 3;
        svg.append(svgElement('line', {x1: left, x2: right, y1: y(value), y2: y(value), class: 'qy-chart-grid'}));
        svg.append(svgElement('text', {x: left - 9, y: y(value) + 4, 'text-anchor': 'end', class: 'qy-axis-label'}, value.toFixed(high-low > 5 ? 0 : 2)));
      }
      [0, data.labels.length - 1].forEach(i => svg.append(svgElement('text', {x: x(i), y: height - 4, 'text-anchor': i ? 'end' : 'start', class: 'qy-axis-label'}, data.labels[i])));
      data.series.forEach((series, index) => {
        const d = series.values.map((value, i) => `${i ? 'L' : 'M'}${x(i)},${y(value)}`).join(' ');
        svg.append(svgElement('path', {d, class: `qy-plot-line key-${index}`, 'stroke-dasharray': index === 1 ? '6 5' : 'none'}));
      });
      const point = Number(input.value);
      svg.append(svgElement('line', {x1: x(point), x2: x(point), y1: top, y2: bottom, class: 'qy-chart-cursor'}));
      data.series.forEach((series, i) => {
        svg.append(svgElement('circle', {cx: x(point), cy: y(series.values[point]), r: 4, class: `qy-plot-point key-${i}`}));
        chart.querySelector(`[data-series-value="${i}"]`).textContent = series.values[point].toFixed(2);
      });
      chart.querySelector('[data-chart-date]').textContent = data.labels[point];
      input.setAttribute('aria-valuetext', `${data.labels[point]}，${data.series.map(series => `${series.name} ${series.values[point].toFixed(2)}`).join('，')}`);
    };
    input.addEventListener('input', draw);
    surface.addEventListener('pointermove', event => {
      if (event.pointerType === 'touch') return;
      const rect = surface.getBoundingClientRect();
      input.value = Math.round(Math.max(0,Math.min(1,(event.clientX - rect.left - 48) / (rect.width - 60))) * (data.labels.length - 1));
      draw();
    });
    observeSize(surface, draw);
  });
  root.querySelectorAll('.qy-gallery').forEach(gallery => {
    const track = gallery.querySelector('.qy-gallery-track');
    const slides = [...track.querySelectorAll('.qg-image')];
    const previous = gallery.querySelector('[data-action="previous"]');
    const next = gallery.querySelector('[data-action="next"]');
    const play = gallery.querySelector('[data-action="play"]');
    const counter = gallery.querySelector('[data-counter]');
    let active = 0;
    let timer;
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
    const update = () => {
      active = slides.reduce((best, slide, index) => Math.abs(slide.offsetLeft - slides[0].offsetLeft - track.scrollLeft) < Math.abs(slides[best].offsetLeft - slides[0].offsetLeft - track.scrollLeft) ? index : best, 0);
      previous.disabled = active === 0;
      next.disabled = active === slides.length - 1;
      counter.textContent = `${String(active + 1).padStart(2,'0')} / ${String(slides.length).padStart(2,'0')}   ${gallery.dataset.source}`;
    };
    const go = index => track.scrollTo({ left: slides[index].offsetLeft - slides[0].offsetLeft, behavior: reduced.matches ? 'instant' : 'smooth' });
    const stop = () => { clearInterval(timer); timer = undefined; play.textContent = '自动播放'; play.setAttribute('aria-pressed','false'); };
    previous.addEventListener('click', () => { stop(); go(Math.max(0,active-1)); });
    next.addEventListener('click', () => { stop(); go(Math.min(slides.length-1,active+1)); });
    play.addEventListener('click', () => {
      if (timer) { stop(); return; }
      play.textContent = '暂停播放'; play.setAttribute('aria-pressed','true');
      timer = setInterval(() => {
        if (!root.isConnected || document.hidden || dialog.open) { stop(); return; }
        go((active+1)%slides.length);
      }, 5000);
    });
    track.addEventListener('scroll', update, {passive:true});
    track.addEventListener('pointerdown', stop, {passive:true});
    track.addEventListener('keydown', event => {
      if (event.target !== track || !['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) return;
      event.preventDefault(); stop();
      go(event.key === 'Home' ? 0 : event.key === 'End' ? slides.length-1 : Math.max(0,Math.min(slides.length-1,active+(event.key==='ArrowRight'?1:-1))));
    });
    new IntersectionObserver(entries => { if (!entries[0].isIntersecting) stop(); }, {threshold: .1}).observe(gallery);
    update();
    cleanups.push(stop);
  });
  window.addEventListener('pagehide', () => cleanups.forEach(stop => stop()), {once:true});
  // Streamlit inserts the gallery after the browser's initial anchor jump.
  if (location.hash === '#team-gallery') {
    requestAnimationFrame(() => root.querySelector('#team-gallery')?.scrollIntoView({block:'start',behavior:'instant'}));
  }
})();
