# Trace 子页：前后端部署 / Trace page deployment

公开入口：<https://pris-cv.github.io/DataEvolver/traces/?lang=zh>

这是 Project Page 的**公开、只读 trace 展示服务**，不是私有 Harness 的训练控制台。
Trace API 包含经过整理的两条历史轨迹、六张原图、轮次评价、实际修改与证据 JSON。
前端另有一组三物体 Qwen / FLUX 配置对比（六张原图），作为独立静态证据展示，
不混入逐轮 trace API，也不把跨模型对比描述成单模型的自进化。
不读取任意实验目录，不启动渲染、VLM 或训练，不提供停止、恢复或删除接口。

## 1. 安装与本地开发

要求 Node.js 22+（CI / Docker 使用 Node 24）和 npm。无第三方 npm 运行依赖。
前后端、npm 依赖、构建产物与部署模板均隔离在 `apps/project-page/`，
不依赖 Harness、训练环境或实验目录。先进入该目录执行：

```sh
cd apps/project-page
npm ci
npm run dev
```

打开 `http://127.0.0.1:4173/traces/?lang=zh`。开发服务同时提供前端和只读 API。
修改前端文件后刷新浏览器；Node 的 watch 模式重启服务端代码。

```sh
npm test         # API、图片哈希、路径边界、CORS、数据回退与构建测试
npm run build    # 生成 dist/，保留首页和 /traces/ 子页
npm run preview  # 静态构建预览；默认 127.0.0.1:8787，不提供 API
```

使用浏览器 HTTP 服务打开，不再保证直接双击 `file://` 的 ES module 页面。
`build` 只重建固定的 `dist/` 目录，不修改原图、JSON 或实验源数据。

## 2. Project Page 前端发布（npm）

默认不配置 API：页面独立使用随站点发布的归档快照，无后端也能完整查看。

1. 将代码正常提交到 `main`。
2. GitHub Actions 的 `Deploy GitHub Pages` 工作流执行 `npm ci → npm test → npm run build`。
3. 上传 `apps/project-page/dist/`，保留项目首页并发布 `/DataEvolver/traces/`。

Pull request 只做测试与构建，并提供可下载的 `project-page-preview` artifact，
不会覆盖线上页面；合并到 `main` 后自动部署。Harness 改动不会触发该工作流。

也可安装并登录 GitHub CLI 后运行：

```sh
npm run deploy
```

此命令仅触发仓库 **main 已提交版本**的 Pages 工作流，不会自动提交、推送或发布本地修改。
发布状态：<https://github.com/PRIS-CV/DataEvolver/actions/workflows/deploy-pages.yml>

GitHub Pages 是静态站点托管，不会运行 Node 后端：
<https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages>

## 3. 独立后端 / 同机前后端

后端代码：`apps/project-page/backend/server.mjs`。默认仅监听 loopback。
Linux 示例把代码和 npm 缓存放在 `/aaaidata`，不占用 `/home` 的实验空间：

```sh
cd /aaaidata/dataevolver-project/apps/project-page
npm ci --cache /aaaidata/dataevolver-project/apps/project-page/.npm
TRACE_API_BASE_URL=/ npm run build
npm start
```

`npm start` 同时提供构建后的前端和只读 API，打开
`http://127.0.0.1:8787/traces/?lang=zh`；`/` 表示前端连接同源 API。
不要与正在运行的 `preview` 共用端口，可设置 `PORT=8788`。
只需要后端时执行 `npm run start:api`。

环境变量：

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| `HOST` | `127.0.0.1` | 服务监听地址；不要直接将私有 Harness 暴露到公网 |
| `PORT` | `8787`，开发 `4173` | 本只读服务端口 |
| `TRACE_DATA_DIR` | 应用内的 `frontend/traces` | 已整理公开包，不是原始实验根目录 |
| `TRACE_ALLOWED_ORIGINS` | 空 | 允许跨域访问的完整 origin，逗号分隔，不允许 `*` |
| `TRACE_API_BASE_URL` | 空 | **构建时**前端 API 配置；`/` 为同源，或独立 HTTPS origin |

`TRACE_API_BASE_URL` 不接受账号密码、query、fragment 或子路径。
本地调试允许 `http://127.0.0.1:端口`；公网连接必须 HTTPS。
不要在公开构建变量中放密钥。浏览器 URL 参数不能覆盖 API 地址。

### 将 GitHub Pages 连接独立 API

1. 部署本只读后端，使用你拥有的域名与 TLS，通过反向代理连接 `127.0.0.1:8787`。
2. 后端配置 `TRACE_ALLOWED_ORIGINS=https://pris-cv.github.io`。origin 不含 `/DataEvolver`。
3. 仓库 Actions variable 设置 `TRACE_API_BASE_URL=https://你的API域名`，重新发布前端。
4. 页面显示“服务已连接 · 只读归档”。这不是训练运行或实时调度状态。

API 无法连接、超时或返回错误数据时，页面明确显示“归档副本”，不会假装服务在线。
重试通过刷新页面进行。后端的归档是在启动时读取的快照；更新已审核包后需重启服务。
API 图像启动时验证原始 SHA256，之后从内存服务同一批字节，避免混入被修改的文件。

### API

| 请求 | 返回 |
| --- | --- |
| `GET /api/health` | 只读状态、归档 revision、案例数 |
| `GET /api/traces` | 完整公开 manifest |
| `GET /api/traces/grounding` | 接地案例 |
| `GET /api/traces/count/rounds/3` | 指定轮次 |
| `GET /api/assets/count-r3.png` | 已登记且哈希核验的原图 |

支持 HEAD；CORS 预检仅允许 GET/HEAD。所有修改方法返回 405；任意文件路径、
路径穿越、越界符号链接、隐藏文件和未登记的图片均不可通过 API 读取。
接口不需要登录，因为内容与公开 Pages 归档相同；**不要接入未脱敏私有记录**。

## 4. 部署配置

### Docker Compose

```sh
docker compose -f deploy/compose.yaml up -d --build
docker compose -f deploy/compose.yaml logs --tail=50
```

镜像用 npm 构建前端，运行时使用非 root 用户、只读文件系统；宿主机仅映射
`127.0.0.1:8787`。不挂载实验根目录。构建上下文只包含 `apps/project-page/`，
不会把 Harness、模型或实验目录发送给 Docker。

### systemd 与 Nginx

- `dataevolver-traces.service`：示例单位文件。先创建专用非特权用户，核对 Node 路径、
  `/aaaidata/dataevolver-project/apps/project-page` 及只读权限，再由管理员安装启用。不要直接覆盖现有 Harness 服务。
- `nginx.conf.example`：加入已有 HTTPS server block；TLS 域名/证书由部署者提供。
  可以只代理 `/api/`，也可同时服务 `dist/`。

这些是部署模板，不代表已在你的服务器安装 systemd、Docker 或公开监听端口。

## 5. 证据更新与设计边界

- 保留 `frontend/traces/build_showcase.py` 导出器与 Python 证据核验；npm build 不重新生成实验数据。
- 当前 `v1` 合约仅接入 `grounding` 和 `count`，不是自动扫描任意 Harness runs。
  新案例类型需同时审核导出合约、字段渲染与 UI 事实文案；不能只替换分数。
- 整理过程不调用模型，不改变原始图片，不删除失败轮次，不把局部恢复描述成全面达标。
- 设计沿用首页的暖炭色 / 奶油色、铜色强调、衬线标题、1200px 内容宽度；
  参考用户 DESIGN.md 的编辑式排版，不复制 Claude 标志或使用其授权字体。
- EN / 中文、浅色 / 深色、移动布局、键盘操作与 reduced-motion 保留。

## English quick start

Run `npm ci && npm run dev` for the frontend plus the read-only archive API.
Use `npm run build` to produce `dist/`; GitHub Pages deploys that directory through
the repository workflow. `npm run preview` is static-only. For a self-hosted full
service, build with `TRACE_API_BASE_URL=/` and run `npm start`. To connect Pages to
an external API, set an HTTPS origin at build time and configure the exact Pages
origin in `TRACE_ALLOWED_ORIGINS`. No private experiment controller is exposed.
