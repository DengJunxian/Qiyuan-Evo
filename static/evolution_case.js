/* UI-only fictional case navigation. No requests, model calls or archive mutations. */
(() => {
  const root = document.querySelector('.qy-evolution-case');
  if (!root || root.dataset.ready) return;
  root.dataset.ready = 'true';
  // Keep the current chapter through Streamlit appearance rerenders. A full
  // navigation creates a new document, so explicit chapter links still win.
  const rememberedChapter = window.__qyEvolutionCaseChapter;
  const tabs = [...root.querySelectorAll('[data-chapter]')];
  const scenes = [...root.querySelectorAll('[data-scene]')];
  const previous = root.querySelector('[data-case-action="previous"]');
  const next = root.querySelector('[data-case-action="next"]');
  const counter = root.querySelector('[data-case-counter]');
  const ns = 'http://www.w3.org/2000/svg';
  const element = (tag, attrs, text) => {
    const node = document.createElementNS(ns, tag);
    Object.entries(attrs || {}).forEach(([key,value]) => node.setAttribute(key,value));
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const renders = [];
  const observers = [];
  const watch = (container, draw) => {
    const observer = new ResizeObserver(() => {
      if (!root.isConnected) { observer.disconnect(); return; }
      if (container.clientWidth) draw();
    });
    observer.observe(container); observers.push(observer); renders.push(draw);
  };
  root.querySelectorAll('[data-case-edges]').forEach((graph, index) => {
    const svg = element('svg', {'aria-hidden':'true'});
    graph.prepend(svg);
    const edges = JSON.parse(graph.dataset.caseEdges);
    const draw = () => {
      if (!graph.clientWidth) return;
      svg.setAttribute('viewBox', `0 0 ${graph.clientWidth} ${graph.clientHeight}`);
      svg.replaceChildren();
      const defs = element('defs');
      const marker = element('marker',{id:`case-arrow-${index}`,viewBox:'0 0 8 8',refX:7,refY:4,markerWidth:5,markerHeight:5,orient:'auto'});
      marker.append(element('path',{d:'M1 1 L7 4 L1 7',fill:'none',stroke:'currentColor','stroke-width':1.5}));
      defs.append(marker);svg.append(defs);
      const bounds=graph.getBoundingClientRect();
      const roles=new Map([...graph.querySelectorAll('[data-role]')].map(node=>[node.dataset.role,node]));
      edges.forEach(([from,to]) => {
        const a=roles.get(from).getBoundingClientRect(), b=roles.get(to).getBoundingClientRect();
        const ax=a.left+a.width/2-bounds.left, ay=a.top+a.height/2-bounds.top;
        const bx=b.left+b.width/2-bounds.left, by=b.top+b.height/2-bounds.top;
        const dx=bx-ax, dy=by-ay;
        const at=Math.min((a.width/2+3)/(Math.abs(dx)||.001),(a.height/2+3)/(Math.abs(dy)||.001));
        const bt=Math.min((b.width/2+8)/(Math.abs(dx)||.001),(b.height/2+8)/(Math.abs(dy)||.001));
        const x1=ax+dx*at,y1=ay+dy*at,x2=bx-dx*bt,y2=by-dy*bt;
        const d=Math.abs(dx)>Math.abs(dy) ? `M${x1},${y1} C${(x1+x2)/2},${y1} ${(x1+x2)/2},${y2} ${x2},${y2}` : `M${x1},${y1} C${x1},${(y1+y2)/2} ${x2},${(y1+y2)/2} ${x2},${y2}`;
        svg.append(element('path',{d,'marker-end':`url(#case-arrow-${index})`}));
      });
    };
    watch(graph,draw);
  });
  root.querySelectorAll('[data-prices]').forEach(chart => {
    const series=Object.entries(JSON.parse(chart.dataset.prices));
    const svg=element('svg',{'aria-hidden':'true'});chart.append(svg);
    const draw=()=>{
      const width=chart.clientWidth;if(!width)return;
      const height=220,left=34,right=width-12,top=12,bottom=192;
      svg.setAttribute('viewBox',`0 0 ${width} ${height}`);svg.replaceChildren();
      const x=i=>left+i/10*(right-left), y=p=>bottom-(p-90)/15*(bottom-top);
      [90,95,100,105].forEach(p=>{svg.append(element('line',{x1:left,x2:right,y1:y(p),y2:y(p),class:'case-grid'}));svg.append(element('text',{x:left-7,y:y(p)+3,'text-anchor':'end'},p));});
      [0,5,10].forEach(i=>svg.append(element('text',{x:x(i),y:214,'text-anchor':i===0?'start':i===10?'end':'middle'},`第 ${i} 步`)));
      series.forEach(([name,prices],index)=>svg.append(element('path',{d:prices.map((p,i)=>`${i?'L':'M'}${x(i)},${y(p)}`).join(' '),class:`price-path price-${index}`})));
    };watch(chart,draw);
  });
  const select = (chapter, updateUrl = true) => {
    const value = Math.max(1,Math.min(8,Number(chapter)));
    root.dataset.selected=String(value);
    window.__qyEvolutionCaseChapter=value;
    tabs.forEach(tab=>{const active=Number(tab.dataset.chapter)===value;tab.setAttribute('aria-selected',String(active));tab.tabIndex=active?0:-1;});
    scenes.forEach(scene=>{scene.hidden=Number(scene.dataset.scene)!==value;});
    previous.disabled=value===1;next.disabled=value===8;counter.textContent=`${String(value).padStart(2,'0')} / 08`;
    if(updateUrl){const url=new URL(location.href);url.searchParams.set('chapter',String(value));history.replaceState(null,'',url);}
    requestAnimationFrame(()=>renders.forEach(draw=>draw()));
  };
  tabs.forEach((tab,index)=>{
    tab.addEventListener('click',()=>select(tab.dataset.chapter));
    tab.addEventListener('keydown',event=>{
      if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;
      event.preventDefault();const i=event.key==='Home'?0:event.key==='End'?7:(index+(event.key==='ArrowRight'?1:-1)+8)%8;
      select(i+1);tabs[i].focus({preventScroll:true});
    });
  });
  previous.addEventListener('click',()=>select(Number(root.dataset.selected)-1));
  next.addEventListener('click',()=>select(Number(root.dataset.selected)+1));
  select(rememberedChapter ?? root.dataset.selected,rememberedChapter !== undefined);
  window.addEventListener('pagehide',()=>observers.forEach(observer=>observer.disconnect()),{once:true});
})();
