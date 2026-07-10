# HTML 演示模板

可复用幻灯片模板。深色衬线主题、MathJax 公式、键盘/圆点/目录导航、Canvas 动画框架、5 套主题、演讲者备注、PDF 导出。骨架按职责拆成独立文件，浏览器直接打开即可全屏演示，无需构建。

## 两个入口

| 文件 | 用途 |
|---|---|
| **`template.html`** | 精简起手式 — 复制它开始做新演示 |
| **`showcase.html`** | 完整组件画廊 — 来这里看效果、抄片段 |

## 文件结构

```
html模板/
├── template.html        起手式: 内容幕 + 全部功能接好线
├── showcase.html        画廊: 每种组件/主题/动画工具的示范
├── css/
│   ├── theme.css        【设计】默认配色/字体/形状 token
│   ├── themes.css       【设计】备选主题皮肤 (light/academic/warm/projector)
│   ├── base.css         【设计】reset + 排版 (标题/正文/代码块/kbd/mark…)
│   ├── components.css   【设计】内容组件 (见下表)
│   └── navigation.css   【导航】演示外壳 + 进度条/备注/帮助/打印
├── js/
│   ├── engine.js        【引擎】翻页 + fragment + 全屏 + 备注 + 主题循环
│   └── canvas.js        【引擎】动画工具箱 (registry + plot helpers + 主题感知)
└── README.md
```

## 用法

1. 复制整个 `html模板/` 文件夹改名（css/js 用相对路径，要一起带走）。
2. 在 `template.html` 里，每一「幕」是一个 `<div class="slide" data-title="目录名">`，第一幕加 `active`。
3. 删掉不需要的示范幕、照结构填内容。导航/目录/圆点由 `engine.js` 自动生成。
4. 动画写在 `template.html` 底部 inline `<script>`，用 `animate()` 注册（见下）。

> CSS `<link>` 顺序即层叠顺序：theme → themes → base → components → navigation，勿乱序。
> JS 顺序：`canvas.js` → `engine.js` → 本页动画。

## 快捷键

| 键 | 作用 |
|---|---|
| `←` `→` `空格` `PgUp/Dn` | 上一步 / 下一步（先点 fragment，再翻幕） |
| `0`–`9` | 跳到对应幕 |
| `Home` / `End` | 首幕 / 末幕 |
| `F` | 全屏 |
| `S` | 演讲者备注面板 |
| `T` | 循环切换主题 |
| `?` | 快捷键帮助浮层 |
| `⌘P` | 导出 PDF（所有幕纵向平铺、fragment 全展开） |

## 主题 (5 套)

在 `<html>` 上设属性，一行切换；不写=默认深色。也可演示时按 `T` 循环。

```html
<html lang="zh-Hans" data-theme="academic">
```

`light`(浅色简洁) · `academic`(暖白学术蓝) · `warm`(暖棕深色) · `projector`(纯黑高对比)。
canvas 颜色会自动跟着主题变——`canvas.js` 直接读 CSS 变量，**不再需要手动同步两处**。改色只动 `theme.css` / `themes.css`。

## 组件速查 (showcase.html 每种都有示范幕)

| 组件 | 类名 |
|---|---|
| 提示框 | `.callout` + `live`/`dead`/`blue` |
| 统计大数字 | `.stats` > `.stat` (`.stat-num`/`.stat-label`/`.stat-sub`) |
| 进度条 | `.bar-row` (`.bar` > `.bar-fill`) |
| 卡片网格 | `.cards` > `.card` (`.card-icon`/`.card-title`) |
| 徽章 | `.badge` + `live`/`dead`/`blue`/`purple` |
| 对比/利弊 | `.compare` > `.compare-col.pro` / `.con` |
| 时间线 | `.timeline` > `.tl-item` (`.tl-time`/`.tl-title`/`.tl-body`) |
| 横向流程 | `.flow` > `.flow-step` + `.flow-sep` |
| 键值定义 | `dl.kv` > `dt`/`dd` |
| 引用 | `.quote` + `.quote-cite` |
| 图片图注 | `.figure` > `img` + `.figure-cap` |
| 分步推导 | `.derivation` > `.step-card` + `.arrow`，tag 三色 |
| 多栏 | `.two-col` / `.three-col` |
| 表格 | `table` / `table.zebra` |
| 代码块 | `<pre><code>` |
| 逐条点出 | 任意元素加 `.fragment` |
| 演讲者备注 | 幕内 `<div class="notes">` |

## 写动画

推荐用 `canvas.js` 的高层 API，注册后由统一 RAF 循环驱动，**自动只在该幕 active 时绘制**：

```js
animate('canvasId', 幕序号, (s) => {
  clearBg(s);                                  // 用主题底色清屏
  const box = { x: 40, y: 20, w: s.w-60, h: s.h-50 };
  const p = makePlot(s.ctx, box, [0,1], [-1,1]);   // 数据坐标系
  p.grid(); p.frame();
  p.fn(x => Math.sin(6*x + s.t), colors.accent);   // s.t 秒, s.dt 帧间隔
});
```

回调参数 `s = { ctx, w, h, t(秒), dt(秒), frame }`，可往 `s` 上挂自定义状态。
`makePlot` 提供 `fn / line / bars / dot / grid / frame / text`，全在数据坐标下作图。
其它工具：`randn()` 高斯噪声、`clamp` / `lerp`、`arrow()`、主题感知的 `colors`。
也可沿用低层写法（`setupHiDPI` + 自己的 `requestAnimationFrame`），见 OU 老演示。
