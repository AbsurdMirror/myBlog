# 图解技术笔记 · 本地交接工作副本

Markdown 和原始配图是唯一正文源；网页由它们构建。当前任务是在已有仓库 `AbsurdMirror/myBlog` 上完成正式图片入库、正文合并与 GitHub Pages 发布，不是重新设计网站。

**先读 [交接说明](progress/handoff.md)。**

## 此压缩包是什么

这是最新本地工作文件快照，含正文、五张原图、构建/发布脚本、站点样式和已生成的 `site/`。

- **没有 `.git/`**，不能把解压目录当作已经关联远端、可直接 push 的 Git clone。
- **不包含旧站 `archive/hexo-site/`**；它已保存在远端源码分支。请在新环境 clone 现有工作分支后覆盖本包文件，保留原 `.git/`、旧站归档和历史。不要 `git init` 后强推，也不要全目录镜像删除。
- `site/` 为本地生成物，打包仅方便预览与部署，不应提交到 `main`。
- 当前有本地文件尚未提交。`status: published` 是构建白名单标记，不代表页面已经上线。

## 主要入口

- [Layer 0 正文](docs/models/deepseek-v4.1-flash/02-layer0.md)
- [模型系列入口](docs/models/deepseek-v4.1-flash/index.md)
- [写作规范](AUTHORING.md)
- [五图清单和 SHA256](docs/models/deepseek-v4.1-flash/assets/figures.json)
- [本次读回的远端状态](progress/remote-observed.json)

## 本地检查与构建

需要 Python 3.11+、PyYAML、Pandoc。现有主构建路线无需安装 MkDocs。

```sh
python scripts/check.py --published-only
python scripts/build_static.py
python scripts/check_site.py
python -m http.server 8000 --directory site
```

浏览器访问 `http://localhost:8000/models/deepseek-v4.1-flash/02-layer0.html`。

从远端 clone 并覆盖本包后，正式构建应使用：

```sh
python scripts/build_static.py --require-legacy
python scripts/check_site.py
```

`--require-legacy` 检查旧站快照是否存在；不要建一个空文件夹蒙混过关。可单独生成内嵌原图的 HTML：

```sh
python scripts/preview.py --output .preview/layer0.html
```

## 发布与未完成事项

`.github/workflows/publish.yml` 在 main 更新后构建和校验站点，提交至现有 gh-pages 分支并显式请求 Pages 构建。保留旧文章资源与发布分支历史；以实际 Pages 页面验收发布结果。

`scripts/publish_remote.py` 是备选的直接 API 发布脚本。默认仅做本地 dry-run；显式 `--execute` 才写远端。它需要执行环境能联网并已有认证；不继承聊天连接器的凭据。执行前读脚本、刷新远端分支并核对 Pages 设置。脚本不是已验证成功的一键部署方案，细节见交接说明。

旧站备份必须保留。五张原图不得重画或替换为缩略图。仓库公开，不提交令牌、个人笔记或整段聊天。
