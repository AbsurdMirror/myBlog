# 图解技术笔记

Markdown 与采用版配图是正文源，网页由 Python、Pandoc 和 MathML 构建。

- [在线阅读](https://absurdmirror.github.io/myBlog/)
- [DeepSeek V4.1 Layer 0](https://absurdmirror.github.io/myBlog/models/deepseek-v4.1-flash/02-layer0.html)
- [DeepSeek V4.1 Layer 1](https://absurdmirror.github.io/myBlog/models/deepseek-v4.1-flash/03-layer1.html)
- [Layer 0 正文](docs/models/deepseek-v4.1-flash/02-layer0.md)
- [Layer 1 正文](docs/models/deepseek-v4.1-flash/03-layer1.md)
- [写作规范](AUTHORING.md)
- [发布验收记录](progress/online-verification.json)

## 构建与发布

需要 Python 3.11+、PyYAML 和 Pandoc。

```sh
python scripts/build_static.py --require-legacy
python scripts/check_site.py
```

main 更新后，发布 workflow 构建和校验站点，将结果提交到 gh-pages，并显式请求 Pages 构建。旧文章资源保留在 archive/hexo-site，发布分支历史和旧站备份均保留。

site/ 是生成物，不提交到 main。十四张采用版配图由清单记录 SHA256；图中已核实的勘误或限定同时记录在清单与相邻正文。Layer 0、Layer 1 全文的固定 revision、来源范围和已知标注限定见正文及 progress/sources.json。

progress/handoff.md 与 remote-observed.json 是接手时的历史记录；当前上线状态以 online-verification.json 为准。
