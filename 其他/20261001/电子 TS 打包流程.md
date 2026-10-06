# 电子 TS 打包流程

## 🧑‍💻 User

electron是如何打包TypeScript 写的app的？

---
## 😀 默认助手 (7)

# Electron 打包 TypeScript app 的原理与流程

## 核心前提：TypeScript 不会被 Electron 直接运行

这是理解整个流程的关键：**Electron 本身不识别 TypeScript**。

Electron 由两部分组成：

| 进程 | 运行环境 | 能执行的语言 |
|------|---------|-------------|
| 主进程 (main) | Node.js | JavaScript |
| 渲染进程 (renderer) | Chromium | JavaScript / HTML / CSS |
| 预加载 (preload) | Node.js 沙箱 | JavaScript |

所以 `.ts` 文件**必须先编译成 `.js`**，Electron 运行时真正加载的永远是 JS。这就引出了两个**完全独立**的阶段：

```
TypeScript 源码  ──[编译/打包]──►  JavaScript 产物  ──[打包分发]──►  .exe / .dmg / .AppImage
     │                                  │                          │
  src/main.ts                     dist/main.js              myapp-Setup.exe
  src/renderer.ts                 dist/renderer.js
```

> **常见误区**：很多人把 "编译" 和 "打包" 混为一谈。编译是 `tsc`/`esbuild` 的事，打包是 `electron-builder`/`electron-forge` 的事，两者解耦。

---

## 阶段一：TypeScript → JavaScript（编译）

### 方案 A：纯 `tsc`（最原始、零魔法）

```json
// tsconfig.json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "CommonJS",
    "moduleResolution": "node",
    "outDir": "dist",
    "rootDir": "src",
    "strict": true,
    "sourceMap": true
  },
  "include": ["src/**/*.ts"]
}
```

```json
// package.json —— 关键是指向编译产物，而不是 .ts
{
  "main": "dist/main.js",
  "scripts": {
    "build": "tsc",
    "start": "npm run build && electron ."
  }
}
```

缺点：主进程可以用 CJS，**但渲染进程通常是 ESM/bundler 体系**，一个 `tsc` 很难同时满足两边，且无法处理 `import './style.css'`、图片等资源。

### 方案 B：现代 Bundler（推荐）

主流工具链会同时处理 TS 编译 + 模块打包 + 资源导入：

- **`electron-vite`**（最流行，基于 Vite）
- **Electron Forge + Vite/Webpack 插件**
- **esbuild / SWC**（极快，适合自定义脚本）

它们的核心思路是**为 main / preload / renderer 三份代码分别建构建配置**，因为三者目标环境不同：

```javascript
// electron.vite.config.ts
import { defineConfig, externalizeDepsPlugin } from 'electron-vite'

export default defineConfig({
  main: {
    plugins: [externalizeDepsPlugin()]  // 不打包 node_modules 里的原生依赖
  },
  preload: {
    plugins: [externalizeDepsPlugin()]
  },
  renderer: {
    // 这里是浏览器环境，可以像普通前端项目一样配置
  }
})
```

```json
{
  "main": "out/main/index.js",
  "scripts": {
    "dev": "electron-vite dev",
    "build": "electron-vite build"
  }
}
```

---

## 阶段二：JavaScript → 可执行安装包（打包分发）

编译完成后，`dist/` 或 `out/` 里是纯粹的 JS。这时才交给打包工具。三大主流工具：

<details>
<summary><b>1. electron-builder（最常用）</b></summary>

优点：配置声明式、跨平台一键出安装包、自动处理签名与自动更新。

```json
// package.json
{
  "main": "out/main/index.js",
  "scripts": {
    "build": "electron-vite build",
    "dist": "npm run build && electron-builder"
  },
  "build": {
    "appId": "com.example.myapp",
    "files": ["out/**/*", "package.json"],   // 只打包编译产物，排除 src/*.ts
    "directories": { "output": "release" },
    "win": { "target": "nsis" },
    "mac": { "target": "dmg" },
    "linux": { "target": "AppImage" }
  }
}
```

它会：
1. 下载对应平台的 Electron 预编译二进制
2. 把你的 `out/` + `package.json` 塞进 `resources/app.asar`
3. 生成安装器（NSIS/DMG/AppImage）

</details>

<details>
<summary><b>2. Electron Forge（官方推荐）</b></summary>

官方维护，插件化，配合 `@electron-forge/plugin-vite` 可打通开发/打包。

```javascript
// forge.config.js
module.exports = {
  packagerConfig: { asar: true },
  makers: [
    { name: '@electron-forge/maker-squirrel' },  // Windows
    { name: '@electron-forge/maker-dmg' },       // macOS
    { name: '@electron-forge/maker-deb' }        // Linux
  ],
  plugins: [{ name: '@electron-forge/plugin-vite' }]
}
```

```
npx electron-forge import   # 自动把现有项目改造成 Forge 结构
npm run make                # 出安装包
```

</details>

<details>
<summary><b>3. electron-packager（最基础，已被 Forge 集成）</b></summary>

只负责把 Electron 二进制 + 你的代码文件夹合并，**不生成安装器**，适合快速测试。

```
electron-packager . MyApp --platform=win32 --arch=x64 --out=release
```

</details>

---

## 完整典型流程（以 electron-vite + electron-builder 为例）

```text
my-app/
├── src/
│   ├── main/index.ts        # 主进程 TS
│   ├── preload/index.ts     # 预加载 TS
│   └── renderer/            # 前端 TS/TSX + HTML + CSS
├── electron.vite.config.ts
├── tsconfig.json
└── package.json

# 开发：Vite 提供主进程热重载 + 渲染进程 HMR
npm run dev

# 构建：TS → JS（三份产物）
npm run build          # 产出 out/main/*.js, out/preload/*.js, out/renderer/*

# 分发：JS → 安装包
npm run dist           # 产出 release/MyApp Setup.exe
```

`package.json` 中 **`main` 字段永远指向编译后的 JS**（如 `out/main/index.js`），绝不指向 `.ts`——这是新手最常见的报错来源。

---

## 关键难点与最佳实践

<details>
<summary><b>ESM vs CJS 的坑</b></summary>

- 主进程/预加载：Electron 对 ESM 支持有限（较新版本才逐步支持），通常产出 **CommonJS**。
- 渲染进程：现代前端普遍是 **ESM**。
- 结果：需要**两套 tsconfig / 构建目标**，不能一刀切。`electron-vite` 已内置处理。

</details>

<details>
<summary><b>原生模块（node-gyp / .node）</b></summary>

原生模块（如 `better-sqlite3`、`sharp`）编译后是平台相关的二进制：
- 打包时必须用 `externalizeDepsPlugin` / `--external` 排除它们，**不能 bundle**。
- 用 `electron-rebuild` 针对 Electron 的 ABI 重新编译。

</details>

<details>
<summary><b>`__dirname` 与路径</b></summary>

编译后 JS 在 `out/` 里，相对路径会变。资源加载应基于 `app.getAppPath()` 或使用打包工具提供的路径注入，不要写死相对源码的路径。

</details>

<details>
<summary><b>类型定义</b></summary>

安装 `electron` 包即自带 `electron.d.ts`，直接 `import { app, BrowserWindow } from 'electron'` 即可获得完整类型提示。

</details>

---

## 一句话总结

**TypeScript 写的 Electron app，本质是「先用 bundler（electron-vite/esbuild）把 TS 编译成三份 JS 产物（main / preload / renderer），再用打包工具（electron-builder/forge）把 JS 产物 + Electron 运行时封成安装包」。** 编译和打包是两条流水线，TS 本身在最终产物里已经不存在了。

如果你告诉我具体用的是哪套工具链（或者贴出 `package.json`），我可以给出针对性的配置和排错建议。
