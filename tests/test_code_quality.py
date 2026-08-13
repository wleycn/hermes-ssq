"""代码质量静态检查: 调用链深度 ≤3 层 + Google docstring 强制。

规则(用户 2026-08-13 约定, 写入 multi-agent-systems-architect skill):
1. 函数调用链深度 ≤3 层(A→B→C), 优先 2 层——新代码零豁免, 存量渐进重构。
2. Google docstring: 有参数必须有 Args:, 有返回必须有 Returns:。
3. 存量文件(大文件/历史代码)列入豁免清单, 未来重构后移除; 新文件必须零豁免达标。

本测试用 AST 静态扫描, 不 import 任何业务模块(避免执行副作用)。
"""
import ast
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 豁免清单: (文件名, 原因) —— 存量代码, 渐进重构后移除
# 规则: 新交付/新修改文件不允许出现在豁免清单里(零豁免)
HARD_MODULES = ["wheel.py", "ml/popularity.py", "ml/decode.py", "ml/spectral.py"]

# 存量文件: 允许超限但必须报告(渐进重构目标), 不出现在豁免清单则测试失败
ALLOWLIST_OVER_DEPTH = {
    "evaluate.py": ["write_report->4层", "main->5层"],
    "select_numbers.py": ["main->4层"],
}

# 私有辅助函数(下划线开头)不强制 docstring, 但调用链仍计入
DOCSTRING_EXEMPT_PREFIX = ("_",)


def _parse(path: Path) -> ast.Module:
    """解析 Python 文件为 AST。

    Args:
        path: 目标 .py 文件路径。

    Returns:
        解析后的 AST 模块节点。

    Raises:
        SyntaxError: 文件语法错误(测试直接失败)。
    """
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _module_funcs(tree: ast.Module) -> dict:
    """提取模块内全部函数定义。

    Args:
        tree: 已解析的 AST 模块。

    Returns:
        函数名 -> FunctionDef 节点 的映射。
    """
    return {
        n.name: n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _call_graph(tree: ast.Module) -> dict:
    """构建模块内函数调用图。

    Args:
        tree: 已解析的 AST 模块。

    Returns:
        函数名 -> 被调用的模块内函数名集合。
    """
    funcs = _module_funcs(tree)
    graph = {name: set() for name in funcs}
    for name, node in funcs.items():
        for n in ast.walk(node):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                callee = n.func.id
                if callee in funcs and callee != name:
                    graph[name].add(callee)
    return graph


def _longest_chain(graph: dict, start: str, max_depth: int = 10) -> list:
    """迭代 DFS 求从 start 出发的最长调用链(带环检测)。

    Args:
        graph: 调用图(函数名 -> 被调函数集合)。
        start: 起点函数名。
        max_depth: 搜索深度上限, 防环导致无限循环。

    Returns:
        最长链的函数名列表(环存在时返回检测到的链)。
    """
    best: list = [start]
    stack: list = [(start, [start])]
    while stack:
        node, path = stack.pop()
        if len(path) > len(best):
            best = path
        if len(path) >= max_depth:
            continue
        for callee in graph.get(node, set()):
            if callee in path:
                continue  # 环: 跳过避免死循环
            stack.append((callee, path + [callee]))
    return best


def _max_chain_depth(graph: dict, public_names: list) -> tuple:
    """计算全部公开函数的最长调用链深度。

    规则细化(2026-08-13 与用户约定): 调用链 ≤3 层, 其中**纯工具叶子层**
    不计入层级——从链尾持续剥离两类函数:
      1) 下划线前缀(模块内部实现细节, 如 _clamp01、_rule_*);
      2) 模块内出度=0 的纯叶子(不再调用模块内任何函数, 如 fisher_g_pvalue)。
    理由: 工具函数与内部实现不构成"迷宫", 硬性压缩会破坏合理模块边界;
    非叶子公开中间层仍受 ≤3 层硬约束。

    Args:
        graph: 调用图。
        public_names: 公开函数名列表(非下划线开头)。

    Returns:
        (最大深度, 超标函数名列表[(name, depth, chain)])。
    """
    over: list = []
    max_depth = 0
    for name in public_names:
        chain = _longest_chain(graph, name)
        trimmed = list(chain)
        while trimmed:
            last = trimmed[-1]
            is_private = last.startswith("_")
            is_pure_leaf = not graph.get(last, set())
            if is_private or is_pure_leaf:
                trimmed.pop()
            else:
                break
        depth = len(trimmed)
        max_depth = max(max_depth, depth)
        if depth > 3:
            over.append((name, depth, trimmed))
    return max_depth, over


def _google_docstring_issues(tree: ast.Module) -> list:
    """检查 Google docstring 合规性(有参必有 Args, 有返回必有 Returns)。

    Args:
        tree: 已解析的 AST 模块。

    Returns:
        违规描述列表, 空 = 合规。
    """
    issues: list = []
    funcs = _module_funcs(tree)
    for name, node in funcs.items():
        if name.startswith(DOCSTRING_EXEMPT_PREFIX):
            continue
        doc = ast.get_docstring(node) or ""
        args = [a.arg for a in node.args.args if a.arg not in ("self", "cls")]
        has_return = any(
            isinstance(n, ast.Return) and n.value is not None
            for n in ast.walk(node)
        )
        if args and "Args:" not in doc:
            issues.append(f"{name}: 有 {len(args)} 个参数但缺 Args 段")
        if has_return and "Returns:" not in doc:
            issues.append(f"{name}: 有返回值但缺 Returns 段")
    return issues


def _scan_file(path: Path) -> dict:
    """扫描单个文件的静态检查结果。

    Args:
        path: 目标 .py 文件路径。

    Returns:
        {depth, over_depth, doc_issues} 汇总。
    """
    tree = _parse(path)
    funcs = _module_funcs(tree)
    graph = _call_graph(tree)
    public = [n for n in funcs if not n.startswith("_")]
    max_depth, over = _max_chain_depth(graph, public)
    return {
        "depth": max_depth,
        "over_depth": over,
        "doc_issues": _google_docstring_issues(tree),
    }


TARGETS = [
    PROJECT_ROOT / "wheel.py",
    PROJECT_ROOT / "ml/popularity.py",
    PROJECT_ROOT / "ml/decode.py",
    PROJECT_ROOT / "ml/spectral.py",
    PROJECT_ROOT / "ml/spectral_red.py",
]


@pytest.mark.parametrize("path", TARGETS)
def test_call_depth_leq_3(path: Path):
    """硬门禁: 新交付模块公开函数调用链 ≤3 层。"""
    result = _scan_file(path)
    over = [f"{name}->{depth}层({' -> '.join(chain)})" for name, depth, chain in result["over_depth"]]
    assert not over, f"{path.name}: 调用链超过 3 层: {over}"


@pytest.mark.parametrize("path", TARGETS)
def test_google_docstring(path: Path):
    """硬门禁: 新交付模块公开函数必须 Google docstring 合规。"""
    result = _scan_file(path)
    assert not result["doc_issues"], f"{path.name}: {result['doc_issues']}"


def test_all_scan_targets_parse():
    """全部目标文件必须可被 AST 解析(语法健康)。"""
    for path in TARGETS:
        assert path.exists(), f"缺少文件 {path}"
        _parse(path)  # SyntaxError 会直接让测试失败


def test_static_scan_report():
    """存量文件扫描报告: 超限项必须在豁免清单内(渐进重构台账)。"""
    legacy = [PROJECT_ROOT / "evaluate.py", PROJECT_ROOT / "select_numbers.py"]
    for path in legacy:
        if not path.exists():
            continue
        result = _scan_file(path)
        over = [f"{name}->{depth}层" for name, depth, _ in result["over_depth"]]
        allow = ALLOWLIST_OVER_DEPTH.get(path.name, [])
        unknown = [o for o in over if o not in allow]
        assert not unknown, (
            f"{path.name}: 调用链超限但不在豁免清单: {unknown} "
            f"(存量代码请渐进重构后移除豁免)"
        )
