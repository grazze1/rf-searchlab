"""Record installed dependency versions and declared license metadata."""

from importlib.metadata import distributions
from pathlib import Path

root = Path(__file__).resolve().parents[1]
rows = []
for distribution in distributions():
    metadata = distribution.metadata
    name = metadata.get("Name", "")
    if name.lower().replace("_", "-") == "rf-searchlab":
        continue
    license_text = metadata.get("License-Expression") or metadata.get("License") or "Not declared in metadata"
    license_text = " ".join(license_text.split()).replace("|", "/")[:140]
    rows.append((name, distribution.version, license_text))
text = "# 依赖清单\n\n本表来自发布验证环境的已安装包元数据，包含运行与开发依赖。完整条款以各依赖发行包为准，不改变项目源码授权。\n\n| 包 | 版本 | 声明的许可证/元数据 |\n|---|---|---|\n"
text += "".join(
    f"| {name} | {version} | {license_text} |\n"
    for name, version, license_text in sorted(rows, key=lambda row: row[0].lower())
)
(root / "docs/DEPENDENCIES.md").write_text(text, encoding="utf-8")
