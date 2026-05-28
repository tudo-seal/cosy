"""Generate the examples pages and navigation."""

import re
from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()
nav["Introduction to the Examples"] = "introduction.md"

root = Path(__file__).parent.parent
src = root / "examples"


for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("examples", doc_path)

    parts = tuple(module_path.parts)
    with open(path) as f:
        override_name = re.findall(r"##(.*?)##", f.readline())
        contents = f.read()

    if parts[-1] == "__init__":
        parts = parts[:-1]

    if override_name:
        nav[override_name] = doc_path.as_posix()
    else:
        nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write("---\nhide:\n  - toc\n---\n")
        fd.write(f"```\n{contents}\n```\n")

    # mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

with mkdocs_gen_files.open("examples/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
