import os
import re
from datetime import date
from pathlib import Path

WIKI_PATH = Path(os.getenv("WIKI_PATH", "./wiki"))


def _ensure_dirs():
    for d in ["models", "departments", "concepts", "interpretations"]:
        (WIKI_PATH / d).mkdir(parents=True, exist_ok=True)


# ── 읽기 ──────────────────────────────────────────────────────────────────────

def read_index() -> str:
    p = WIKI_PATH / "index.md"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def read_page(category: str, name: str) -> str:
    """category: models | departments | concepts | interpretations"""
    p = WIKI_PATH / category / f"{name}.md"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def read_all_model_names() -> list[str]:
    d = WIKI_PATH / "models"
    if not d.exists():
        return []
    return [f.stem for f in d.glob("*.md")]


def search_wiki(query: str) -> str:
    """query 키워드가 포함된 wiki 페이지들을 반환 (최대 3개)"""
    results = []
    keywords = query.lower().split()

    for md_file in WIKI_PATH.rglob("*.md"):
        if md_file.name in ("index.md", "log.md"):
            continue
        content = md_file.read_text(encoding="utf-8")
        score = sum(1 for kw in keywords if kw in content.lower())
        if score > 0:
            results.append((score, md_file.stem, content))

    results.sort(key=lambda x: x[0], reverse=True)
    if not results:
        return ""

    combined = []
    for _, name, content in results[:3]:
        combined.append(f"### [{name}]\n{content[:800]}")
    return "\n\n".join(combined)


# ── 쓰기 ──────────────────────────────────────────────────────────────────────

def write_model_page(model_name: str, department: str, project: str,
                     description: str, task_type: str, disease: str,
                     required_data: list[str], result_type: str) -> None:
    _ensure_dirs()
    required_str = ", ".join(required_data)
    content = f"""# {model_name}

## 기본 정보
- **진료과:** {department}
- **프로젝트:** {project}
- **task_type:** {task_type}
- **required_data:** [{required_str}]
- **result_type:** {result_type}

## 설명
{description}

## 파이프라인
(등록 후 업데이트)

## 임상 해석 패턴
(누적 예정)

## 관련 개념
{disease}
"""
    p = WIKI_PATH / "models" / f"{model_name}.md"
    p.write_text(content, encoding="utf-8")


def update_department_page(department: str, model_name: str, project: str) -> None:
    _ensure_dirs()
    p = WIKI_PATH / "departments" / f"{department}.md"
    if p.exists():
        content = p.read_text(encoding="utf-8")
        model_line = f"- [[{model_name}]] - {project}"
        if model_line not in content:
            content += f"\n{model_line}\n"
    else:
        content = f"# {department}\n\n## 모델 목록\n- [[{model_name}]] - {project}\n"
    p.write_text(content, encoding="utf-8")


def update_index(model_name: str, department: str, project: str) -> None:
    index_path = WIKI_PATH / "index.md"
    content = index_path.read_text(encoding="utf-8") if index_path.exists() else "# MARS AI Agent Wiki Index\n\n## Models\n\n## Departments\n\n## Concepts\n\n## Interpretations\n"

    model_line = f"- [[{model_name}]] - {department}/{project}"
    dept_line = f"- [[{department}]] - {project}"

    # Models 섹션에 추가
    if model_line not in content:
        content = _insert_under_section(content, "## Models", model_line)

    # Departments 섹션에 추가
    if dept_line not in content:
        content = _insert_under_section(content, "## Departments", dept_line)

    index_path.write_text(content, encoding="utf-8")


def remove_model_from_index(model_name: str) -> None:
    index_path = WIKI_PATH / "index.md"
    if not index_path.exists():
        return
    content = index_path.read_text(encoding="utf-8")
    lines = [l for l in content.splitlines() if f"[[{model_name}]]" not in l]
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def delete_model_page(model_name: str) -> bool:
    p = WIKI_PATH / "models" / f"{model_name}.md"
    if p.exists():
        p.unlink()
        return True
    return False


def append_log(action: str, description: str) -> None:
    log_path = WIKI_PATH / "log.md"
    today = date.today().isoformat()
    entry = f"## [{today}] {action} | {description}\n"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry)


def append_interpretation(model_name: str, interpretation: str) -> None:
    """좋은 해석 결과를 해당 모델 wiki 페이지에 누적"""
    p = WIKI_PATH / "models" / f"{model_name}.md"
    if not p.exists():
        return
    content = p.read_text(encoding="utf-8")
    today = date.today().isoformat()
    pattern_entry = f"\n### [{today}] 해석 패턴\n{interpretation}\n"
    content = content.replace("## 임상 해석 패턴\n(누적 예정)", f"## 임상 해석 패턴{pattern_entry}")
    if "## 임상 해석 패턴" not in content:
        content += f"\n## 임상 해석 패턴{pattern_entry}"
    p.write_text(content, encoding="utf-8")


# ── 내부 유틸 ─────────────────────────────────────────────────────────────────

def _insert_under_section(content: str, section_header: str, line: str) -> str:
    """section_header 바로 아래에 line 삽입"""
    lines = content.splitlines()
    result = []
    inserted = False
    for i, l in enumerate(lines):
        result.append(l)
        if not inserted and l.strip() == section_header:
            result.append(line)
            inserted = True
    if not inserted:
        result.append(section_header)
        result.append(line)
    return "\n".join(result) + "\n"
