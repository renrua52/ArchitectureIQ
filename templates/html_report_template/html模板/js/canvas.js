/* ============================================================
   canvas.js —【引擎】Canvas 动画工具箱
   ------------------------------------------------------------
   做交互/动画用的通用函数, 与具体内容无关。
   两种写法:
     A. 高层(推荐): animate('cvId', slideIndex, (s) => { ...用 s.ctx 画... })
        注册后由统一 RAF 循环驱动, 自动只在该幕 active 时绘制。
        回调参数 s = { ctx, w, h, t(秒), dt(秒), frame }。
     B. 低层: 自己 setupHiDPI + requestAnimationFrame(见 OU 老演示)。

   颜色 colors 自动读取 css/theme.css 的变量, 切主题会跟着变。
   绘图捷径: clearBg / grid / arrow / makePlot。
   ============================================================ */

/* ---------- 随机数 / 数学 ---------- */
// 标准正态随机数 (Box–Muller)
function randn() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const lerp = (a, b, t) => a + (b - a) * t;

/* ---------- 高分屏 canvas ---------- */
// 返回 {ctx, w, h} (w/h 为 CSS 像素尺寸)
function setupHiDPI(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  return { ctx, w: rect.width, h: rect.height };
}

/* ---------- 主题感知调色板 ---------- */
// 从 CSS 变量读色, 切主题后调 refreshCanvasColors() 即可同步
function readCanvasColors() {
  const cs = getComputedStyle(document.documentElement);
  const g = (n, fb) => (cs.getPropertyValue(n).trim() || fb);
  return {
    bg: g('--canvas-bg', '#0a0e12'),
    grid: g('--border', '#1c2229'),
    dim: g('--dim', '#5a6068'),
    accent: g('--accent', '#d4a574'),
    live: g('--live', '#a8d8b9'),
    dead: g('--dead', '#d8a8a8'),
    blue: g('--blue', '#8db4d8'),
    purple: g('--purple', '#b9a8d8'),
    text: g('--text', '#e8e4d8'),
  };
}
let colors = readCanvasColors();
function refreshCanvasColors() { colors = readCanvasColors(); return colors; }
window.refreshCanvasColors = refreshCanvasColors;

/* ---------- 动画注册器 + 统一 RAF 循环 ---------- */
const __anims = [];
let __lastTs = 0;
function __tick(ts) {
  const dt = __lastTs ? (ts - __lastTs) / 1000 : 0;
  __lastTs = ts;
  for (const a of __anims) {
    const s = window.slides && window.slides[a.slideIndex];
    if (!s || !s.classList.contains('active')) continue;   // 不在当前幕就跳过, 省 CPU
    a.state.t = ts / 1000; a.state.dt = dt; a.state.frame++;
    a.draw(a.state);
  }
  requestAnimationFrame(__tick);
}
requestAnimationFrame(__tick);

// 注册一个 canvas 动画。draw(state) 每帧调用; state = {ctx,w,h,t,dt,frame, ...你存的}
function animate(canvasId, slideIndex, draw) {
  const cv = document.getElementById(canvasId);
  if (!cv) return null;
  const base = setupHiDPI(cv);
  const state = { ...base, t: 0, dt: 0, frame: 0 };
  __anims.push({ slideIndex, draw, state });
  return state;   // 返回 state, 可往上挂自定义字段(粒子数组/参数等)
}

/* ---------- 绘图捷径 ---------- */
// 用主题底色清屏
function clearBg(s) { s.ctx.fillStyle = colors.bg; s.ctx.fillRect(0, 0, s.w, s.h); }

// 画网格 (在矩形 box={x,y,w,h} 内, nx×ny 格)
function grid(ctx, box, nx = 4, ny = 4, color = colors.grid) {
  ctx.strokeStyle = color; ctx.lineWidth = 1;
  for (let i = 0; i <= nx; i++) {
    const x = box.x + i * box.w / nx;
    ctx.beginPath(); ctx.moveTo(x, box.y); ctx.lineTo(x, box.y + box.h); ctx.stroke();
  }
  for (let j = 0; j <= ny; j++) {
    const y = box.y + j * box.h / ny;
    ctx.beginPath(); ctx.moveTo(box.x, y); ctx.lineTo(box.x + box.w, y); ctx.stroke();
  }
}

// 带箭头的线段
function arrow(ctx, x1, y1, x2, y2, color = colors.accent, head = 8) {
  ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 1.8;
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  const a = Math.atan2(y2 - y1, x2 - x1);
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - head * Math.cos(a - 0.4), y2 - head * Math.sin(a - 0.4));
  ctx.lineTo(x2 - head * Math.cos(a + 0.4), y2 - head * Math.sin(a + 0.4));
  ctx.closePath(); ctx.fill();
}

/* ---------- 数据坐标系绘图器 ----------
   在像素矩形 box={x,y,w,h} 里建立数据坐标 xRange=[x0,x1], yRange=[y0,y1] (y 向上)。
   返回一组在数据坐标下作图的方法:
     p.sx(v)/p.sy(v)            数据→像素
     p.frame()                  画外框
     p.grid(nx,ny)              画网格
     p.fn(f, color)             画函数曲线 y=f(x)
     p.line(points, color)      画折线 points=[[x,y],...]
     p.bars(values, color)      画柱状图 (x 按索引均分)
     p.dot(x,y,color,r)         画点
     p.text(x,y,str,color)      在数据坐标处写字
*/
function makePlot(ctx, box, xRange, yRange) {
  const [x0, x1] = xRange, [y0, y1] = yRange;
  const sx = v => box.x + (v - x0) / (x1 - x0) * box.w;
  const sy = v => box.y + box.h - (v - y0) / (y1 - y0) * box.h;
  return {
    sx, sy,
    frame(color = colors.dim) {
      ctx.strokeStyle = color; ctx.lineWidth = 1;
      ctx.strokeRect(box.x, box.y, box.w, box.h);
    },
    grid(nx = 4, ny = 4, color = colors.grid) { grid(ctx, box, nx, ny, color); },
    fn(f, color = colors.accent, width = 2, steps = 240) {
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath();
      for (let i = 0; i <= steps; i++) {
        const x = x0 + (x1 - x0) * i / steps, y = f(x);
        const px = sx(x), py = clamp(sy(y), box.y - 2, box.y + box.h + 2);
        i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
      }
      ctx.stroke();
    },
    line(points, color = colors.accent, width = 2) {
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath();
      points.forEach(([x, y], i) => { const px = sx(x), py = sy(y); i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py); });
      ctx.stroke();
    },
    bars(values, color = colors.accent, gap = 0.25) {
      const bw = box.w / values.length;
      ctx.fillStyle = color;
      values.forEach((v, i) => {
        const h = (v - y0) / (y1 - y0) * box.h;
        ctx.fillRect(box.x + i * bw + bw * gap / 2, box.y + box.h - h, bw * (1 - gap), h);
      });
    },
    dot(x, y, color = colors.accent, r = 3) {
      ctx.fillStyle = color; ctx.beginPath(); ctx.arc(sx(x), sy(y), r, 0, 2 * Math.PI); ctx.fill();
    },
    text(x, y, str, color = colors.dim, font = '11px JetBrains Mono') {
      ctx.fillStyle = color; ctx.font = font; ctx.fillText(str, sx(x), sy(y));
    },
  };
}
