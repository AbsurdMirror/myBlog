# 图解技术笔记

Markdown 与原始配图是正文源，网页由 Python、Pandoc 和 MathML 构建。

- [在线阅读](https://absurdmirror.github.io/myBlog/)
- [DeepSeek V4.1 Layer 0](https://absurdmirror.github.io/myBlog/models/deepseek-v4.1-flash/02-layer0.html)
- [正文](docs/models/deepseek-v4.1-flash/02-layer0.md)
- [写作规范](AUTHORING.md)
- [发布验收记录](progress/online-verification.json)

## 构建与发布

需要 Python 3.11+、PyYAML 和 Pandoc。

```sh
python scripts/build_static.py --require-legacy
python scripts/check_site.py
```

main 更新后，发布 workflow 构建和校验站点，将结果提交到 gh-pages，并显式请求 Pages 构建。旧文章资源保留在 archive/hexo-site，发布分支历史和旧站备份均保留。

site/ 是生成物，不提交到 main。九张采用版原图保持原始字节，清单记录 SHA256。模型事实的来源限定和已知标注歧义见正文及 progress/sources.json。

progress/handoff.md 与 remote-observed.json 是接手时的历史记录；当前上线状态以 online-verification.json 为准。
