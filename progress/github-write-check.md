# GitHub 写入通道验证

本次用户重新授权后，已恢复文件创建、Git blob / tree / commit、分支引用更新及 PR 合并接口。

验证使用 `scripts/fixtures/binary-upload-probe.png`：一个 1×1 的透明 PNG（70 bytes），通过 `create_blob(encoding=base64)` 写入真实二进制内容，不是将 Base64 字符串作为 PNG 文本保存。返回 blob SHA 与本地计算一致：`145a07dbcb5cf07a4a8560492854177f3d6ce292`。

该文件仅用于检验图片二进制上传与工作分支提交能力，不是五张模型配图之一，不纳入网站，不替代尚未上传的原始配图。本次测试不修改线上 gh-pages。
