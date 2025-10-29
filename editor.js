(function () {
  // ---------- Utilities ----------
  const $ = (id) => document.getElementById(id);
  const has = (id) => !!$(id);

  // ---------- Canvas setup ----------
  const canvas = $('edit-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  // Off-screen PAINT layer
  const paint = document.createElement('canvas');
  paint.width = canvas.width;
  paint.height = canvas.height;
  const pctx = paint.getContext('2d', { willReadFrequently: true });

  const baseURL = window.__CURRENT_OUTPUT_URL || '';

  // ---------- State ----------
  const state = {
    tool: 'brush',                // 'brush'|'erase'|'text'|'pan'|'select'
    color: (has('color') ? $('color').value : '#ffffff'),
    size:  (has('size')  ? parseInt($('size').value,10) : 24),
    drawing: false,
    panning: false,
    startX: 0,
    startY: 0,
    zoom: 1,
    offsetX: 0,
    offsetY: 0,

    // selection
    restrictToSelection: has('restrictSel') ? $('restrictSel').checked : false,
    sel: null,
    activeHandle: null,

    // vector objects (text)
    objects: [],
    selectedIndex: -1,

    // history of PAINT layer
    history: [],
    historyPtr: -1,

    // eyedropper
    picking: false,

    // NEW: optional colored eraser (null = transparent erase)
    eraseColor: null,
  };

  const textStyle = {
    family: (has('fontFamily') ? $('fontFamily').value : 'Arial, Helvetica, sans-serif'),
    size:   (has('fontSize') ? parseInt($('fontSize').value, 10) : 28),
    color:  (has('textColor') ? $('textColor').value : '#111111'),
    bold:   (has('fontBold') ? $('fontBold').checked : false),
    italic: (has('fontItalic') ? $('fontItalic').checked : false),
  };

  let baseImg = null;

  // ---------- Wire toolbox ----------
  const setTool = (t) => { state.tool = t; state.picking = false; };

  has('tool-brush') && $('tool-brush').addEventListener('click', () => setTool('brush'));
  has('tool-erase') && $('tool-erase').addEventListener('click', () => setTool('erase'));
  has('tool-text')  && $('tool-text').addEventListener('click',  () => setTool('text'));
  has('tool-move')  && $('tool-move').addEventListener('click',  () => setTool('pan'));
  has('selectBtn')  && $('selectBtn').addEventListener('click', () => setTool('select'));
  has('clearSelBtn')&& $('clearSelBtn').addEventListener('click', () => { state.sel = null; state.activeHandle = null; draw(); });

  has('color') && $('color').addEventListener('input', e => { state.color = e.target.value; draw(); });
  has('size')  && $('size').addEventListener('input',  e => { state.size = parseInt(e.target.value, 10) || 24; });

  has('restrictSel') && $('restrictSel').addEventListener('change', e => {
    state.restrictToSelection = e.target.checked;
    draw();
  });

  // Eyedropper
  has('pick') && $('pick').addEventListener('click', () => {
    state.picking = !state.picking;
    canvas.style.cursor = state.picking ? 'crosshair' : '';
  });

  // text style controls
  has('fontFamily') && $('fontFamily').addEventListener('change', e => { textStyle.family = e.target.value; draw(); });
  has('fontSize') && $('fontSize').addEventListener('input', e => { textStyle.size = Math.max(8, parseInt(e.target.value, 10) || 28); draw(); });
  has('textColor') && $('textColor').addEventListener('input', e => { textStyle.color = e.target.value; draw(); });
  has('fontBold') && $('fontBold').addEventListener('change', e => { textStyle.bold = e.target.checked; draw(); });
  has('fontItalic') && $('fontItalic').addEventListener('change', e => { textStyle.italic = e.target.checked; draw(); });

  // zoom
  const setZoom = (z) => {
    state.zoom = Math.max(0.2, Math.min(4, z));
    has('zoom-label') && ($('zoom-label').textContent = `${Math.round(state.zoom*100)}%`);
    draw();
  };
  has('zoom-in')  && $('zoom-in').addEventListener('click',  () => setZoom(state.zoom + 0.1));
  has('zoom-out') && $('zoom-out').addEventListener('click', () => setZoom(state.zoom - 0.1));

  // ---------- History over PAINT ----------
  const pushHistory = () => {
    try {
      const snapshot = paint.toDataURL('image/png', 1.0);
      state.history.splice(state.historyPtr + 1);
      state.history.push(snapshot);
      state.historyPtr = state.history.length - 1;
    } catch {}
  };

  const restoreFromDataURL = (dataURL, onload) => {
    const img = new Image();
    img.onload = () => {
      pctx.setTransform(1,0,0,1,0,0);
      pctx.clearRect(0,0,paint.width,paint.height);
      pctx.drawImage(img, 0, 0, paint.width, paint.height);
      onload && onload();
    };
    img.src = dataURL;
  };

  has('undo') && $('undo').addEventListener('click', () => {
    if (state.historyPtr > 0) {
      state.historyPtr--;
      restoreFromDataURL(state.history[state.historyPtr], () => draw());
    }
  });

  has('redo') && $('redo').addEventListener('click', () => {
    if (state.historyPtr + 1 < state.history.length) {
      state.historyPtr++;
      restoreFromDataURL(state.history[state.historyPtr], () => draw());
    }
  });

  has('clear') && $('clear').addEventListener('click', () => {
    pushHistory();
    pctx.save();
    pctx.setTransform(1,0,0,1,0,0);
    pctx.clearRect(0,0,paint.width,paint.height);
    pctx.restore();
    state.objects = [];
    state.sel = null; state.selectedIndex = -1; state.activeHandle = null;
    draw();
  });

  // ---------- Download / Save ----------
  const bakeVectorObjects = () => {
    pctx.save();
    pctx.setTransform(1,0,0,1,0,0);
    state.objects.forEach(obj => {
      if (obj.type === 'text') {
        const weight = obj.bold ? '700' : '400';
        const italic = obj.italic ? 'italic' : 'normal';
        pctx.font = `${italic} ${weight} ${obj.fontSize}px ${obj.fontFamily}`;
        pctx.fillStyle = obj.color;
        pctx.textBaseline = 'top';
        const pad = 4;

        const wrap = (str, maxW) => {
          const words = str.split(/\s+/);
          let lines = [], line = '';
          words.forEach(w => {
            const test = (line ? line + ' ' : '') + w;
            if (pctx.measureText(test).width <= maxW) line = test;
            else { if (line) lines.push(line); line = w; }
          });
          if (line) lines.push(line);
          return lines;
        };

        const lines = wrap(obj.text, obj.w - pad*2);
        let y = obj.y + pad;
        lines.forEach(ln => {
          pctx.fillText(ln, obj.x + pad, y);
          y += obj.fontSize * 1.3;
        });
      }
    });
    pctx.restore();
  };

  has('download') && $('download').addEventListener('click', () => {
    bakeVectorObjects();
    const a = document.createElement('a');
    a.href = paint.toDataURL('image/png');
    a.download = 'edit.png';
    a.click();
  });

  if (has('save-form') && has('save-img-field')) {
    $('save-form').addEventListener('submit', () => {
      bakeVectorObjects();
      $('save-img-field').value = paint.toDataURL('image/png', 1.0);
    });
  }

  // ---------- Geometry helpers ----------
  const screenToCanvas = (sx, sy) => {
    const rect = canvas.getBoundingClientRect();
    const x = (sx - rect.left) / state.zoom - state.offsetX;
    const y = (sy - rect.top)  / state.zoom - state.offsetY;
    return {x,y};
  };

  const handleRects = (box) => {
    const s = 8;
    const midX = box.x + box.w/2, midY = box.y + box.h/2;
    return [
      {name:'tl', x: box.x - s,        y: box.y - s,        w: s, h: s},
      {name:'tr', x: box.x+box.w,      y: box.y - s,        w: s, h: s},
      {name:'bl', x: box.x - s,        y: box.y+box.h,      w: s, h: s},
      {name:'br', x: box.x+box.w,      y: box.y+box.h,      w: s, h: s},
      {name:'tm', x: midX - s/2,       y: box.y - s,        w: s, h: s},
      {name:'bm', x: midX - s/2,       y: box.y+box.h,      w: s, h: s},
      {name:'ml', x: box.x - s,        y: midY - s/2,       w: s, h: s},
      {name:'mr', x: box.x+box.w,      y: midY - s/2,       w: s, h: s},
    ];
  };

  const isPointInRect = (px,py,r) => px>=r.x && py>=r.y && px<=r.x+r.w && py<=r.y+r.h;

  // ---------- Vector text drawing (screen) ----------
  const setCanvasFont = (obj) => {
    const weight = obj.bold ? '700' : '400';
    const italic = obj.italic ? 'italic' : 'normal';
    ctx.font = `${italic} ${weight} ${obj.fontSize}px ${obj.fontFamily}`;
    ctx.fillStyle = obj.color;
    ctx.textBaseline = 'top';
  };

  const wrapText = (str, maxW, obj) => {
    const words = str.split(/\s+/);
    const lines = [];
    let line = '';
    setCanvasFont(obj);
    words.forEach(w => {
      const test = (line ? line + ' ' : '') + w;
      if (ctx.measureText(test).width <= maxW) line = test;
      else { if (line) lines.push(line); line = w; }
    });
    if (line) lines.push(line);
    return lines;
  };

  const drawTextObject = (obj, selected=false) => {
    setCanvasFont(obj);
    const pad = 4;
    const lines = wrapText(obj.text, obj.w - pad*2, obj);
    let y = obj.y + pad;
    lines.forEach(ln => {
      ctx.fillText(ln, obj.x + pad, y);
      y += obj.fontSize * 1.3;
    });
    if (selected) {
      ctx.save();
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 1;
      ctx.setLineDash([4,3]);
      ctx.strokeRect(obj.x, obj.y, obj.w, obj.h);
      ctx.setLineDash([]);
      handleRects(obj).forEach(r => {
        ctx.fillStyle = '#10b981';
        ctx.fillRect(r.x, r.y, r.w, r.h);
      });
      ctx.restore();
    }
  };

  // ---------- Render pipeline ----------
  const drawBaseImage = () => {
    ctx.drawImage(paint, 0, 0, canvas.width, canvas.height);
  };

  const drawSelection = () => {
    if (!state.sel) return;
    const {x,y,w,h} = state.sel;
    ctx.save();
    ctx.strokeStyle = '#1d4ed8';
    ctx.setLineDash([6,3]);
    ctx.lineWidth = 1.5;
    ctx.strokeRect(x,y,w,h);
    ctx.setLineDash([]);
    const hs = handleRects(state.sel);
    ctx.fillStyle = '#1d4ed8';
    hs.forEach(r => ctx.fillRect(r.x, r.y, r.w, r.h));
    ctx.restore();
  };

  function draw() {
    ctx.save();
    ctx.setTransform(state.zoom,0,0,state.zoom,
                     state.offsetX * state.zoom, state.offsetY * state.zoom);
    ctx.clearRect(-10000,-10000,20000,20000);

    drawBaseImage();
    state.objects.forEach((obj, idx) => {
      if (obj.type === 'text') drawTextObject(obj, idx === state.selectedIndex);
    });
    if (state.sel && state.tool === 'select') drawSelection();

    ctx.restore();
  }

  // ---------- Hit testing ----------
  const hitObject = (x,y) => {
    for (let i=state.objects.length-1; i>=0; i--) {
      const o = state.objects[i];
      if (x>=o.x && y>=o.y && x<=o.x+o.w && y<=o.y+o.h) return i;
    }
    return -1;
  };

  // helper to clip strokes to selection
  const maybeClipToSelection = () => {
    if (state.restrictToSelection && state.sel) {
      pctx.beginPath();
      pctx.rect(state.sel.x, state.sel.y, state.sel.w, state.sel.h);
      pctx.clip();
    }
  };

  // ---------- Mouse handling ----------
  let draggingObj = false;
  let resizingObj = false;
  let dragDX=0, dragDY=0;
  let resizeHandle=null;
  let prevHForFontScale = 0;

  canvas.addEventListener('mousedown', (e) => {
    const {x,y} = screenToCanvas(e.clientX, e.clientY);

    // Eyedropper sample (updates brush + text; and if eraser active, sets eraseColor)
    if (state.picking) {
      const px = Math.floor(x), py = Math.floor(y);
      if (px>=0 && py>=0 && px<paint.width && py<paint.height) {
        const imgData = pctx.getImageData(px, py, 1, 1).data;
        const hex = '#' + [imgData[0],imgData[1],imgData[2]]
          .map(v => v.toString(16).padStart(2,'0')).join('');
        state.color = hex;
        textStyle.color = hex;
        has('color') && ($('color').value = hex);
        has('textColor') && ($('textColor').value = hex);

        // NEW: if eraser is the active tool, also set eraseColor
        if (state.tool === 'erase') {
          state.eraseColor = hex; // colored eraser mode
        }

        state.picking = false;
        canvas.style.cursor = '';
        draw();
      }
      return;
    }

    if (state.tool === 'pan') {
      state.panning = true;
      state.startX = e.clientX;
      state.startY = e.clientY;
      return;
    }

    // click on vector text obj?
    const idx = hitObject(x,y);
    if (idx >= 0) {
      state.selectedIndex = idx;
      const obj = state.objects[idx];
      const h = handleRects(obj).find(h => isPointInRect(x,y,h));
      if (h) {
        resizingObj = true;
        resizeHandle = h.name;
        prevHForFontScale = obj.h;
      } else {
        draggingObj = true;
        dragDX = x - obj.x;
        dragDY = y - obj.y;
      }
      draw();
      return;
    } else {
      state.selectedIndex = -1;
    }

    if (state.tool === 'select') {
      state.sel = {x, y, w: 0, h: 0};
      state.activeHandle = null;
      state.drawing = true;
      draw();
      return;
    }

    if (state.tool === 'text') {
      const newObj = {
        type: 'text',
        text: 'Edit me',
        x, y,
        w: 260, h: Math.max(40, textStyle.size*1.6),
        fontFamily: textStyle.family,
        fontSize:  textStyle.size,
        color:     textStyle.color,
        bold:      textStyle.bold,
        italic:    textStyle.italic,
      };
      state.objects.push(newObj);
      state.selectedIndex = state.objects.length - 1;
      draw();
      return;
    }

    if (state.tool === 'brush' || state.tool === 'erase') {
      pushHistory();
      state.drawing = true;
      pctx.save();
      maybeClipToSelection();
      pctx.lineJoin = 'round';
      pctx.lineCap  = 'round';

      if (state.tool === 'erase') {
        if (state.eraseColor) {
          // colored eraser: paint over with picked color
          pctx.globalCompositeOperation = 'source-over';
          pctx.strokeStyle = state.eraseColor;
        } else {
          // true erase to transparent
          pctx.globalCompositeOperation = 'destination-out';
          pctx.strokeStyle = 'rgba(0,0,0,1)';
        }
      } else {
        pctx.globalCompositeOperation = 'source-over';
        pctx.strokeStyle = state.color;
      }

      pctx.lineWidth = state.size;
      pctx.beginPath();
      pctx.moveTo(x, y);
      return;
    }
  });

  canvas.addEventListener('mousemove', (e) => {
    const {x,y} = screenToCanvas(e.clientX, e.clientY);

    if (state.panning) {
      const dx = (e.clientX - state.startX) / state.zoom;
      const dy = (e.clientY - state.startY) / state.zoom;
      state.offsetX += dx; state.offsetY += dy;
      state.startX = e.clientX; state.startY = e.clientY;
      draw();
      return;
    }

    if (resizingObj && state.selectedIndex >= 0) {
      const obj = state.objects[state.selectedIndex];
      const minW = 40, minH = 24;
      const ox = obj.x, oy = obj.y, ow = obj.w, oh = obj.h;

      switch (resizeHandle) {
        case 'tl': obj.x = Math.min(x, ox+ow-minW); obj.y = Math.min(y, oy+oh-minH);
                   obj.w = ow + (ox - obj.x); obj.h = oh + (oy - obj.y); break;
        case 'tr': obj.y = Math.min(y, oy+oh-minH);
                   obj.w = Math.max(minW, x - ox); obj.h = oh + (oy - obj.y); break;
        case 'bl': obj.x = Math.min(x, ox+ow-minW);
                   obj.w = ow + (ox - obj.x); obj.h = Math.max(minH, y - oy); break;
        case 'br': obj.w = Math.max(minW, x - ox); obj.h = Math.max(minH, y - oy); break;
        case 'tm': obj.y = Math.min(y, oy+oh-minH);
                   obj.h = oh + (oy - obj.y); break;
        case 'bm': obj.h = Math.max(minH, y - oy); break;
        case 'ml': obj.x = Math.min(x, ox+ow-minW);
                   obj.w = ow + (ox - obj.x); break;
        case 'mr': obj.w = Math.max(minW, x - ox); break;
      }

      if (prevHForFontScale > 0) {
        const scale = obj.h / prevHForFontScale;
        obj.fontSize = Math.max(8, Math.round(obj.fontSize * scale));
        prevHForFontScale = obj.h;
      }
      draw();
      return;
    }

    if (draggingObj && state.selectedIndex >= 0) {
      const obj = state.objects[state.selectedIndex];
      obj.x = x - dragDX;
      obj.y = y - dragDY;
      draw();
      return;
    }

    if (state.drawing && (state.tool==='brush' || state.tool==='erase')) {
      pctx.lineTo(x,y);
      pctx.stroke();
      draw();
      return;
    }

    if (state.drawing && state.tool === 'select' && state.sel) {
      state.sel.w = x - state.sel.x;
      state.sel.h = y - state.sel.y;
      draw();
      return;
    }
  });

  canvas.addEventListener('mouseup', () => {
    if (state.drawing && (state.tool==='brush' || state.tool==='erase')) {
      pctx.closePath();
      pctx.restore();
      draw();
    }
    state.drawing = false;
    state.panning = false;
    draggingObj = false;
    resizingObj = false;
    resizeHandle = null;
  });

  canvas.addEventListener('dblclick', (e) => {
    const {x,y} = screenToCanvas(e.clientX, e.clientY);
    const idx = hitObject(x,y);
    if (idx>=0 && state.objects[idx].type==='text') {
      const obj = state.objects[idx];
      const val = prompt('Edit text:', obj.text);
      if (typeof val === 'string') { obj.text = val; draw(); }
    }
  });

  // ---------- Base image load into PAINT ----------
  const loadBase = () => {
    if (!baseURL) { draw(); return; }
    baseImg = new Image();
    baseImg.crossOrigin = 'anonymous';
    baseImg.onload = () => {
      pctx.setTransform(1,0,0,1,0,0);
      pctx.clearRect(0,0,paint.width,paint.height);
      pctx.drawImage(baseImg, 0, 0, paint.width, paint.height);
      draw();
      pushHistory();
    };
    baseImg.src = baseURL;
  };

  // ---------- Init ----------
  loadBase();
  draw();
})();
