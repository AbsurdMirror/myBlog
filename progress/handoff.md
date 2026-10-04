# myBlog 项目交接：继续完成入库与 GitHub Pages 部署

> 给下一位执行者：请直接接手已有工程，先核对远端再提交真实文件。不要重新讨论方案、重写正文、重画图片或反复做 1×1 测试图。用户已经明确授权本项目的提交、合并和部署。只有五张原图与 MD 真正进入仓库、对应新网页实际上线，才算完成。

整理日期：2026-10-04。配套文件：`myBlog_local_repository.zip`。本交接文本同时位于包内 `myBlog/progress/handoff.md`。

## 1. 最终要交付什么

目标仓库：https://github.com/AbsurdMirror/myBlog

用户只要下面三类**已验证可用的链接**，不要把 CI、认证、上传方案重新交给用户选择：

1. 已部署的图文阅读页面。按当前目录预计为 `https://absurdmirror.github.io/myBlog/models/deepseek-v4.1-flash/02-layer0.html`；这是目标地址，交接时没有验证新站已经上线。
2. `main` 分支中的 Markdown 文件：`docs/models/deepseek-v4.1-flash/02-layer0.md`。
3. `main` 分支中的五张原图各自的 GitHub 文件链接；目录 `docs/models/deepseek-v4.1-flash/assets/`。

用户允许使用连接器提交、原生 Git/gh、CI 构建或本地构建后上传，选择实际可靠的路径即可。不要再要求逐步确认已经授权的提交/合并/发布；不得泄露认证信息或为完成任务绕过平台安全限制。

## 2. 远端真实状态——已在这次打包时只读复查

| 对象 | 实际状态 |
|---|---|
| 旧站备份 `backup/hexo-before-rebuild-20261003` | `439738a6b8ba88ab7a6b4d6c1df8ba4a267c940d` |
| 线上分支 `gh-pages` | 同上，仍是旧 Hexo 提交 |
| 源码 `main` | `9b2c7202a98d0d4238b0da986c859bbd01cb67ab` |
| 工作分支 `work/deepseek-v41-layer0` | `efcc3d3717832a605f2e0039bcfa88b238f9f96f` |
| PR #1 | open、draft、未合并；base 为 main |
| 工作分支的正式图片目录 | 只有 `figures.json`，五张 PNG 尚未入库 |
| 新站发布 | 尚无成功上线证据；不能把旧博客首页当作交付 |

审阅 PR：https://github.com/AbsurdMirror/myBlog/pull/1

远端状态原始查询入口、核查时刻见包内 `progress/remote-observed.json`。接手后必须再刷新；若分支已有新提交，先比较并整合，不能强推覆盖。默认分支在上一轮元数据查询中仍为 `gh-pages`，实际 Pages Source 尚需通过站点设置或受支持的 API 再核查。

PR 旧描述里的“请勿直接合并”“等待确认”和“MkDocs 未构建”是早期状态：用户随后明确授权直接完成。应先补齐真实资源、通过校验并更新 PR 描述，不是继续空等确认。模型事实尚未锁定上游版本等问题仍需如实注明。

## 3. 压缩包包含什么，以及不包含什么

```text
myBlog/
├── README.md                        # 当前包的正确入口
├── AUTHORING.md                     # 系列组织、符号和更新规则
├── publication.json                 # 哪些页面参与正式/预览构建
├── docs/
│   ├── index.md
│   ├── legacy/index.md
│   └── models/deepseek-v4.1-flash/
│       ├── index.md
│       ├── 02-layer0.md              # 正文已写，不需要从聊天重建
│       └── assets/
│           ├── figures.json         # 五张图尺寸、SHA256、已知标注问题
│           └── 01_...png … 05_...png
├── scripts/                         # 校验、静态构建、预览、备选发布器
├── theme/                           # 页面样式、MathML展示、图片查看器
├── .github/workflows/               # 检查及构建artifact；部署还需接通
├── progress/                        # 交接、来源、远端快照、验证结果
└── site/                            # 已生成的本地图文站点，非源码
```

**重要：这是完整的当前本地工作文件包，不是远端仓库的全历史 clone。** 没有 `.git/`，也没有旧站 `archive/hexo-site/`。这两者应从现有远端 clone 获得并保留。不要因为本包不包含旧归档就把它从远端删掉；不要 `git init` 新历史后覆盖原仓库。

五张原图在 `docs/` 与 `site/` 各有一份：前者为源资源，后者为构建拷贝，字节相同。双份是为了可以立刻查看本地网站，不是两个不同图版本。`site/` 在 `.gitignore` 中，不能混入 main 正文源。

没有打包访问令牌、登录态、字体文件或完整私人对话。包内 `PACKAGE_MANIFEST.json` 提供除清单自身以外各文件的大小与 SHA256。

### 五张固定采用图

| 文件 | 像素尺寸 | 字节数 | SHA256 |
|---|---:|---:|---|
| `01_layer0_overview.png` | 1422 × 1106 | 1,372,471 | `8781a1771d01dc119bc5d11c166a3b4e4d185920fb2ee01609705fa2c42c2cd4` |
| `02_mhc_mixes_raw.png` | 1086 × 1448 | 1,883,417 | `389a9cba4f46d3629d41987f7ef573a72326b3ae4bdcc3fc528e2d5b59927caf` |
| `03_mhc_coefficient_generation.png` | 1024 × 1536 | 2,862,228 | `6bb128eedd969ca3c0fd384fcc0ed448f3bc4b10e3e682f67dc802cd56b85e7f` |
| `04_mhc_hc_pre.png` | 1448 × 1086 | 1,768,649 | `e1d6d3cb3852e05bbd4345c7ded155397f4d83bc1fde38ca5e2822a4132d5bb5` |
| `05_mhc_hc_post.png` | 1448 × 1086 | 2,069,709 | `457a6e636beffc5d3cd6cccb8d05b62b10c2597598d9e72c26f94e7bf12c0c84` |

对应主题：Layer 0 总图、原始 mixes 生成、pre/post/comb 后处理、hc_pre 汇聚、hc_post 残差融合。**“系数生成”是两张，五张总数不要再算错。**

## 4. 接手后的最短路径

### A. 先确认实际可执行通道

先检查是否有受支持的 GitHub 写入工具，或已经联网、已登录 GitHub 的代码执行环境。连接器与终端的联网、认证是两套路径，不能混为一谈。上一会话的终端无法出网，不代表新环境也一样，更不代表用户没有授权。

若终端确实能联网并已有认证，可 clone 现有仓库；若只提供连接器，按已公开的函数契约读写 Git blob/tree/commit/ref。不要把失败的工具关键词搜索当作没有权限的证明，也不要无休止地重复测试小图。

### B. 在现有工作分支上叠加本地文件

原生 Git 可用时，示例：

```sh
git clone https://github.com/AbsurdMirror/myBlog.git myBlog-remote
cd myBlog-remote
git fetch origin
git switch work/deepseek-v41-layer0
# 核对 HEAD 与交接快照；若不同先处理差异。
```

然后把 ZIP 的 `myBlog/` 文件复制到 clone 的对应位置。**覆盖有意修改的同名文件，但不使用全局 `--delete`；保留 `.git/` 和 `archive/hexo-site/`。** `site/`、清单/交接用的根目录校验文件可以只留本地。

清理已知的传输实验文件（先检查路径存在及其内容，不能按目录大范围删除）：

- `.github/workflows/asset-transfer.yml`
- `scripts/fixtures/binary-upload-probe.png`
- `.bootstrap/` 若存在，先审阅是否仅为旧实验，再决定清理。

这些已经识别的清理项见 `progress/remote-base.json`。提交前检查 diff，确保五张图实际加入，不是只增加引用或 Base64 文本。

### C. 构建与检查

当前主路线使用 **Python + PyYAML + Pandoc** 生成 HTML/MathML，不依赖 MkDocs。`mkdocs.yml` 和相关 hook 是保留的另一入口，不必为了首次上线重做框架。

在已有上述依赖的环境：

```sh
python scripts/check.py --published-only
python scripts/build_static.py --require-legacy
python scripts/check_site.py
python -m http.server 8000 --directory site
```

只有拿到远端的真实 `archive/hexo-site/` 后才使用 `--require-legacy`。单独解压本包预览时可去掉该选项，但不能由此声称已验证旧文章路径。

检查入口：`http://localhost:8000/models/deepseek-v4.1-flash/02-layer0.html`。

### D. 提交、合并、发布

正式 PNG/正文和维护文件进入工作分支，读回五图 SHA256，检查 CI 后把 PR #1 改为非草稿并合并到 main；不要把“blob 创建成功”当作文件已经进入分支。

**本包 `.github/workflows/publish.yml` 目前只是“Build publication bundle”，只上传 artifact，不部署 Pages。** 必须补实际部署环节。两条路线任选其一，并保持一种后续可重复发布方式：

- 已有 `gh-pages` 的分支发布正常可用：将 `site/` 内容提交到该分支，保留备份和旧文章资源。若用 workflow 凭据推送发布分支，必须确认它确实触发了 Pages 构建；不能仅看 push 返回成功。必要时改用 Pages 专用部署 action。
- 改用 GitHub Actions Pages：先确认仓库 Pages 的构建来源设置、必要的发布权限和 environment，再接通 artifact/deploy 流程。不要同时维护两套互相覆盖的发布流水线。

若有管理仓库设置的真实权限，最终把默认分支改为 main，方便直接看源码。不要用 `permissions.admin=true` 的读取结果替代实际设置 API 的授权检验。

首次切换完成后再验证旧 URL、五张图、CSS/JS、MathML 与手机布局，最后给链接。

## 5. 现成发布脚本的适用条件与局限

`scripts/publish_remote.py` 提供另一条 REST API 路线，可参考，不要盲目认为它已经端到端验证成功：

```sh
python scripts/publish_remote.py              # 仅本地检查，无远端写入
python scripts/publish_remote.py --execute    # 真正远端写入，需先审阅和准备
```

- 原图由程序读取字节并 Base64 编码，经 blob/tree/commit/ref 入库，有哈希读回检查。**不要手工粘贴几 MB Base64、缩小原图或用占位图绕过检查。**
- 需要终端的 `GH_TOKEN` / `GITHUB_TOKEN` 或已登录的 `gh`；它不会自动继承聊天连接器的认证。凭据不得写入仓库或聊天。
- 脚本检查工作分支 expected SHA；远端改变时需要先合并与审阅，再更新基准，不能单纯改 SHA 以绕过检查。
- 脚本不会自动重建 `site/`，必须先构建并验证，避免上传旧生成物。
- 脚本预期 Pages 为 `gh-pages` 根目录的分支发布。配置不同会停止，且可能已经先完成源码提交/合并。失败后先查真实状态，不要从头重复执行。
- 脚本在所选源码目录中提交变更，但不包含本包所有根目录辅助文件；有意新增的根目录文件需要明确纳入。
- 脚本没有替你完整处理 PR 的 draft 状态、旧说明、默认分支切换，也不是端到端 UI 测试器。
- 当前只做过本地 dry-run 和结构检查，正式远端发布没有成功验证。新执行者可修正或换用 Git/CI，不必执着于该脚本。

## 6. 本地验证与剩余内容风险

### 本次打包实际检查

检查结果保存在 `progress/handoff-validation.json`：四篇构建页的本地引用有效；五张源图及站点拷贝与图清单 SHA256 一致；PNG 可解码；生成页面有 75 个 MathML 节点。检查通过仅证明文件和渲染结构，不证明模型学术结论全部正确。

此前记录已做桌面 1440px、手机 390px 和图片查看器测试。本次重新构建及文件检查通过，但浏览器重测被当前环境策略阻止（`ERR_BLOCKED_BY_ADMINISTRATOR`），未修改浏览器策略；旧测试只能作为历史记录。不要把本地测试等同于 Pages 已上线，新环境必须再次以线上 URL 验收。

### 尚未完全核实但已透明记录

- 图 2：投影同时需要 `Xf` 与归一化因子 `r`，原图旁路线不完整，正文已说明。
- 图 3：epsilon 应位于行 Softmax 之后相加，原图括号有歧义；正文明确写出递推。
- 图 5：右半是固定一个输出 j、遍历四个输入 i；`post_j` 广播形状 `[B*S,1]`，图中标注有歧义。
- 用户认可原图版式，不能擅自改图；优先保留正文纠正注释。技术内容修正与网站部署是两个验收维度，不把原图认可说成数学细节全部核验。
- 上游候选 revision `2cba9e42aa026125f3ed06c6d98c1db82f7ca027` 没有成功读回验证；`progress/sources.json` 明确记录 false。正文依照前面对官方 main 的核查整理。本次打包没有重新查证模型。下一执行者不要把候选 SHA 写成已锁定依据。

旧 `progress/*.md` 和 PR body 可能仍有早期状态。最新用户授权、本次只读快照与本文件优先；完成部署后更新进度，保留事实核查限定。

## 7. 文档组织和后续更新原则

一个模型一套系列，建议“模型全貌 → 基准层 → 差异层 → 特殊模块 → 完整推理 → 存储与计算”。当前只写 Layer 0，不要为了凑目录虚构后续章节。

第 0 层正文顺序：八模块总览 → mHC（两张系数生成＋hc_pre＋hc_post）→ RMSNorm → SWA → MoE → 输入输出与参数汇总。RMSNorm/SWA/MoE 的专用图还未正式完成，后续再由用户指导。

- 中文、图文并重；正文解释连线、公式和易错点，不机械复述图片。
- `B` 是 batch，`S` 是本次 seq，`S_his` 是参与评分的历史总长度。独立 token 运算写 `B*S`，数学注意力用完整序列与 mask，不先混入窗口 gather/tile 实现。
- 区分可学习权重、动态系数、固定常数。mHC 一次生成 24 个动态值，只有 pre 提前给下一个子层；post/comb 用于本子层出口。
- 以 MD 和原图为源；网页为渲染物。窄正文、宽图、放大/拖动、手机适配。发布知乎时做版式导出，不维护第二套独立正文。
- 图片稳定命名，修改留 Git 历史；在用户指导下逐步更新，不重新生成已确认的五张图。
- 仓库公开，分支/PR/草稿都不是私人空间。不要上传完整聊天、密钥、系统诊断原始敏感数据。

## 8. 上次失败原因与不要再踩的坑

上一执行环境终端对 GitHub、Google 等的 DNS 和外网直连都失败；这不是用户的电脑，也不是用户没给仓库 push 权限。独立联网工具/连接器曾读取成功，连接器也曾完成 MD 与小型 PNG 测试提交。工具发现列表在不同轮次表现不一致，其根因没有确认。

因此新环境只做必要的真实通道检查；可写时直接处理正式文件，别再次循环“申请权限 → 测试小图 → 宣告成功 → 重新打包”。不用凭空承诺后台工作。遇到平台明确禁止的操作不绕过；遇到普通构建问题则依据日志修复。

## 9. 完成标准与最终回复

- [ ] 五张原图进入 main，逐张读回哈希匹配；MD 引用可用。
- [ ] main 包含最新正文、构建/检查/部署配置与维护经验；不夹带 `site/` 或凭据。
- [ ] PR #1 的 draft 与过时说明处理完毕，合并结果真实可查。
- [ ] 旧站快照、备份和 Git 历史保留；旧文 URL 不丢失或有有效跳转。
- [ ] GitHub Pages 新页面实际可访问；五图、公式、目录、图片查看器、手机布局正常。
- [ ] 记录源提交、发布提交/部署编号、实测 URL 和验证时刻。
- [ ] 最终直接给阅读页面链接、main 中 MD 链接和五张图文件链接。

目标链接格式（**必须实际部署并验证之后再说已完成**）：

- 阅读：`https://absurdmirror.github.io/myBlog/models/deepseek-v4.1-flash/02-layer0.html`
- 正文：`https://github.com/AbsurdMirror/myBlog/blob/main/docs/models/deepseek-v4.1-flash/02-layer0.md`
- 图片：`https://github.com/AbsurdMirror/myBlog/blob/main/docs/models/deepseek-v4.1-flash/assets/<文件名>.png`

不要把本地压缩包、旧首页、只创建了 blob、只生成 artifact 或测试图成功当作这个任务完成。
