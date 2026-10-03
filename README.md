# 图解技术笔记

Markdown＋原始配图是唯一内容源。模型系列先建立数学计算与 shape，再讨论执行实现。

## 维护与分支

- `main`：确认后的源码；`work/deepseek-v41-layer0`：当前样章草稿。
- `backup/hexo-before-rebuild-20261003`：旧站备份；`gh-pages`：旧线上站点，未改动。
- `archive/hexo-site/` 直接保留重构前整棵 Git tree，不重写历史。

[写作规则](AUTHORING.md) · [样章](docs/models/deepseek-v4.1-flash/02-layer0.md) · [进度](progress/deepseek-v4.1-flash.md)

## 预览

依赖 Python 3.11+、Pandoc 2.17+。常规站点另外安装 `requirements.txt` 中的 MkDocs。正文经 Pandoc 输出原生 MathML，无外部字体、公式 CDN 或图片图床。

```sh
python scripts/check.py
python scripts/preview.py --output .preview/layer0.html
# 直接用浏览器打开 .preview/layer0.html，单文件包含原图
python -m pip install -r requirements.txt
python scripts/stage.py --preview
mkdocs build --strict
mkdocs serve
```

`--preview` 收录草稿；不加该选项只拷贝 status=published 的页面及实际引用的资源到 .build/docs，草稿不进入发布物。静态站点和离线预览共用 CSS、查看器和 Pandoc 的正文渲染。

## 发布

发布工作流仅允许 main 手动触发，并要求输入 PUBLISH；本轮不触发。需要先把 Pages Source 切换为 GitHub Actions，并完成用户确认与完整检查。当前默认分支仍是 gh-pages，尚未更改；新分支通过明确 URL 访问。

## 图片入库状态

工作分支已有图清单及相对路径；五张原始 PNG 在当前交付的完整源码包与 HTML 内。当前 GitHub 连接没有接受本地图片文件的上传动作，远端 PNG 尚未完成。将源码包的 docs/models/deepseek-v4.1-flash/assets/ 五张 PNG 原样写入对应目录后运行校验；SHA256 必须与 figures.json 一致，不以缩略图或占位图替代。

该仓库公开，工作分支不是私人草稿空间。禁止提交密钥、私人笔记或完整对话。
