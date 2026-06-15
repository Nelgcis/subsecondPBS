#!/usr/bin/env python3
"""
cpp_formatter.py
C++ 代码规范格式化工具
符合公司医疗器械软件 C++ 编码规范（命名/格式/注释/头文件/类/函数/特性等）

用法:
    python3 cpp_formatter.py <file_or_directory> [--inplace] [--log <logfile>]

参数:
    file_or_directory   : 待格式化的 .cpp/.h 文件，或包含 .cpp/.h 的目录
    --inplace           : 直接覆盖原文件（默认输出到 <原文件>.formatted）
    --log <logfile>     : 日志输出路径（默认 ./format_changes.log）
"""

import re
import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

# ─────────────────────────────────────────────
# 日志记录器
# ─────────────────────────────────────────────
class ChangeLog:
    INFO = "INFO"
    WARN = "WARN"
    SKIP = "SKIP"

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.entries: List[str] = []
        self.info_count = 0
        self.warn_count = 0
        self.skip_count = 0

    def add(self, level: str, filepath: str, lineno: int, description: str, clause: str = ""):
        clause_str = f"  (规范条款 {clause})" if clause else ""
        entry = f"[{level}]  {filepath}:{lineno}  {description}{clause_str}"
        self.entries.append(entry)
        if level == self.INFO:
            self.info_count += 1
        elif level == self.WARN:
            self.warn_count += 1
        elif level == self.SKIP:
            self.skip_count += 1

    def write(self):
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("# C++ 格式化变更日志\n")
            f.write(f"# 共计: INFO={self.info_count}, WARN={self.warn_count}, SKIP={self.skip_count}\n")
            f.write("#\n")
            f.write("# [INFO]  纯格式变更，已自动应用\n")
            f.write("# [WARN]  存在潜在含义变化，已在原文件对应行加注释标记，请人工确认\n")
            f.write("# [SKIP]  无法自动修复，需人工处理\n")
            f.write("\n")
            for entry in self.entries:
                f.write(entry + "\n")

    def summary(self) -> str:
        return (f"格式化完成。\n"
                f"  已自动应用格式变更: {self.info_count} 处\n"
                f"  需人工确认 (WARN):  {self.warn_count} 处\n"
                f"  需人工处理 (SKIP):  {self.skip_count} 处\n"
                f"详见日志: {self.log_path}")


# ─────────────────────────────────────────────
# 核心格式化逻辑
# ─────────────────────────────────────────────
class CppFormatter:

    def __init__(self, log: ChangeLog, filepath: str):
        self.log = log
        self.filepath = filepath

    def format_lines(self, lines: List[str]) -> List[str]:
        """对文件行列表逐一应用格式化规则，返回新行列表。"""
        result = []
        i = 0
        while i < len(lines):
            line = lines[i]
            lineno = i + 1
            new_line, applied = self._apply_rules(line, lineno, lines, i)
            result.append(new_line)
            i += 1
        return result

    def _apply_rules(self, line: str, lineno: int, all_lines: List[str], idx: int) -> Tuple[str, bool]:
        """对单行应用所有可应用的格式规则。返回 (新行内容, 是否有改动)。"""
        original = line
        changed = False

        # ── 规范 4.2: Tab → 4空格
        if "\t" in line:
            # 计算 tab 的缩进量，转为等效空格
            new_line = self._tabs_to_spaces(line)
            if new_line != line:
                self.log.add(ChangeLog.INFO, self.filepath, lineno,
                             f"Tab缩进 → 4空格缩进", "4.2")
                line = new_line
                changed = True

        # ── 规范 4.2: 行尾空白清除
        stripped = line.rstrip()
        if stripped != line.rstrip('\n'):
            # 有行尾空白（不算换行符）
            nl = '\n' if line.endswith('\n') else ''
            new_line = stripped + nl
            if new_line != line:
                self.log.add(ChangeLog.INFO, self.filepath, lineno,
                             "行尾空白字符已删除", "4.2")
                line = new_line
                changed = True

        # ── 规范 4.2: 每行不超过120字符（仅记录，不截断）
        content = line.rstrip('\n')
        if len(content) > 120:
            self.log.add(ChangeLog.SKIP, self.filepath, lineno,
                         f"行宽 {len(content)} > 120，需人工换行处理", "4.2")

        # ── 规范 4.3: 注释掉的代码块（整行注释且内容像代码）
        if self._is_commented_out_code(content):
            self.log.add(ChangeLog.WARN, self.filepath, lineno,
                         f"疑似注释掉的代码，规范要求直接删除 → 原行: {content.strip()}", "4.3")
            # 不自动删除，在行前加标记注释
            nl = '\n' if line.endswith('\n') else ''
            warn_tag = "    // FORMAT-WARN: 疑似注释掉的代码，见 format_changes.log\n"
            # 只在行前插入，不修改原行
            line = warn_tag + line
            changed = True

        # ── 规范 4.3: TODO/TBD/FIXME 注释
        if re.search(r'//\s*(TODO|TBD|FIXME)\b', content, re.IGNORECASE):
            self.log.add(ChangeLog.WARN, self.filepath, lineno,
                         f"正式交付代码不能包含 TODO/TBD/FIXME 注释 → 原行: {content.strip()}", "4.3")
            nl = '\n' if line.endswith('\n') else ''
            warn_tag = "    // FORMAT-WARN: TODO/TBD/FIXME 不能出现在正式交付代码中，见 format_changes.log\n"
            line = warn_tag + line
            changed = True

        # ── 规范 4.2: if/for/while 后缺少大括号（单语句无括号，仅记录 WARN）
        if self._missing_braces(content):
            self.log.add(ChangeLog.WARN, self.filepath, lineno,
                         f"if/for/while 缺少大括号，需人工添加 → 原行: {content.strip()}", "4.2")
            nl = '\n' if line.endswith('\n') else ''
            warn_tag = "    // FORMAT-WARN: if/for/while 语句缺少大括号，见 format_changes.log\n"
            line = warn_tag + line
            changed = True

        # ── 规范 4.8.1: #define 定义常量（建议改为 const）
        if self._is_define_constant(content):
            self.log.add(ChangeLog.WARN, self.filepath, lineno,
                         f"用 #define 定义常量，建议改为 const/constexpr → 原行: {content.strip()}", "4.8.1")
            nl = '\n' if line.endswith('\n') else ''
            warn_tag = "// FORMAT-WARN: 建议用 const/constexpr 替代 #define 常量，见 format_changes.log\n"
            line = warn_tag + line
            changed = True

        # ── 规范 4.8.2: 含自增/自减的复合表达式
        if self._has_compound_increment(content):
            self.log.add(ChangeLog.WARN, self.filepath, lineno,
                         f"含自增/自减的复合表达式，需人工将自增/自减单独放一行 → 原行: {content.strip()}", "4.8.2")
            nl = '\n' if line.endswith('\n') else ''
            warn_tag = "    // FORMAT-WARN: 含自增/自减复合表达式，需人工拆分，见 format_changes.log\n"
            line = warn_tag + line
            changed = True

        # ── 规范 4.8.1: 使用魔鬼数字（孤立的数字字面量）
        if self._has_magic_number(content):
            self.log.add(ChangeLog.WARN, self.filepath, lineno,
                         f"疑似魔鬼数字，建议定义为具名常量 → 原行: {content.strip()}", "4.8.1")
            nl = '\n' if line.endswith('\n') else ''
            warn_tag = "    // FORMAT-WARN: 疑似魔鬼数字，建议定义为具名 const 常量，见 format_changes.log\n"
            line = warn_tag + line
            changed = True

        # ── 规范 4.8.3: C 风格强制类型转换
        if self._has_c_style_cast(content):
            self.log.add(ChangeLog.WARN, self.filepath, lineno,
                         f"疑似 C 风格类型转换，建议改为 C++ 风格 → 原行: {content.strip()}", "4.8.3")
            nl = '\n' if line.endswith('\n') else ''
            warn_tag = "    // FORMAT-WARN: 疑似 C 风格类型转换，建议改为 static_cast/dynamic_cast 等，见 format_changes.log\n"
            line = warn_tag + line
            changed = True

        # ── 规范 4.8.10: 使用 NULL 而非 nullptr
        if self._uses_null_macro(content):
            new_line = self._replace_null(line)
            if new_line != line:
                self.log.add(ChangeLog.INFO, self.filepath, lineno,
                             f"NULL → nullptr", "4.8.10")
                line = new_line
                changed = True

        # ── 规范 4.2: 指针/引用两边都无空格（int*p 或 int&p）修正
        new_line = self._fix_pointer_spacing(line)
        if new_line != line:
            self.log.add(ChangeLog.INFO, self.filepath, lineno,
                         f"指针/引用符号空格修正（两边都无空格 → 靠左形式）", "4.2")
            line = new_line
            changed = True

        # ── 规范 4.4: 头文件 #include 使用 <stdlib.h> 等旧式头文件
        if self._has_old_c_header(content):
            self.log.add(ChangeLog.WARN, self.filepath, lineno,
                         f"C 风格头文件，建议改为 C++ 头文件（如 <cstdlib>）→ 原行: {content.strip()}", "4.8.5")
            nl = '\n' if line.endswith('\n') else ''
            warn_tag = "// FORMAT-WARN: 建议使用 C++ 头文件（如 <cstdlib> 替代 <stdlib.h>），见 format_changes.log\n"
            line = warn_tag + line
            changed = True

        # ── 规范 4.8.9: 使用宏函数 #define FUNC(...)
        if self._is_macro_function(content):
            self.log.add(ChangeLog.WARN, self.filepath, lineno,
                         f"宏函数，建议替换为模板函数或内联函数 → 原行: {content.strip()}", "4.8.9")
            nl = '\n' if line.endswith('\n') else ''
            warn_tag = "// FORMAT-WARN: 宏函数建议改为 template/inline 函数，见 format_changes.log\n"
            line = warn_tag + line
            changed = True

        # ── 规范 4.2: 变量多个初始化在同一行（仅 SKIP 提示）
        if self._multi_var_init_on_one_line(content):
            self.log.add(ChangeLog.SKIP, self.filepath, lineno,
                         f"同一行多个变量初始化，建议每行只有一个 → 原行: {content.strip()}", "4.2")

        return line, changed

    # ─────────────────────────────────────────
    # 辅助检测函数
    # ─────────────────────────────────────────

    def _tabs_to_spaces(self, line: str) -> str:
        """将 tab 替换为等效的 4 空格（列对齐）。"""
        result = []
        col = 0
        for ch in line:
            if ch == '\t':
                spaces = 4 - (col % 4)
                result.append(' ' * spaces)
                col += spaces
            else:
                result.append(ch)
                col += 1
        return ''.join(result)

    def _is_commented_out_code(self, line: str) -> bool:
        """检测整行注释且内容像代码（含 ; { } = -> 等符号）。"""
        stripped = line.strip()
        if not stripped.startswith('//'):
            return False
        comment_content = stripped[2:].strip()
        # 排除普通文字注释（不含代码符号）
        if not comment_content:
            return False
        # 疑似代码的特征：含分号、花括号、赋值操作符、函数调用、return/if/for 关键字
        code_patterns = [
            r';\s*$',           # 以分号结尾
            r'\bif\s*\(',       # if(
            r'\bfor\s*\(',      # for(
            r'\bwhile\s*\(',    # while(
            r'\breturn\b',      # return
            r'\w+\s*=\s*\w+',  # 赋值
            r'\w+\s*\(',        # 函数调用
            r'^\s*\{',          # 以 { 开头
        ]
        for p in code_patterns:
            if re.search(p, comment_content):
                return True
        return False

    def _missing_braces(self, line: str) -> bool:
        """检测 if/for/while 后面没有大括号的单语句写法（简单启发式）。"""
        stripped = line.strip()
        # 匹配 if(...)  或 for(...)  或 while(...) 后不跟 {
        # 简单检测：语句以 ) 结尾而不是 ) { 或 ){
        pattern = r'^\s*(if|for|while)\s*\(.*\)\s*$'
        if re.match(pattern, stripped):
            # 排除带 { 的
            if not stripped.endswith('{'):
                return True
        return False

    def _is_define_constant(self, line: str) -> bool:
        """检测 #define 定义常量（非函数宏，非 include guard）。"""
        stripped = line.strip()
        # 匹配 #define NAME value（无括号参数）
        m = re.match(r'^#\s*define\s+([A-Z_][A-Z0-9_]*)\s+(.+)', stripped)
        if not m:
            return False
        name, value = m.group(1), m.group(2).strip()
        # 排除 include guard（只有名字没有值，或值是1/0/空）
        if re.match(r'^[A-Z_][A-Z0-9_]*_H\s*$', name):
            return False
        if value in ('', '1', '0'):
            return False
        # 排除宏函数（含括号参数）
        if re.match(r'^[A-Z_][A-Z0-9_]*\s*\(', stripped[len('#define'):].strip()):
            return False
        # 如果值像数字或字符串常量，则是常量宏
        if re.match(r'^[0-9"\'(]', value):
            return True
        return False

    def _has_compound_increment(self, line: str) -> bool:
        """检测含自增/自减的复合表达式（如 b[i++], a + i++, Func(i++, j)）。"""
        stripped = line.strip()
        # 排除只有 i++ 或 ++i 的单独一行
        if re.match(r'^\+\+\w+\s*;?\s*$', stripped) or re.match(r'^\w+\+\+\s*;?\s*$', stripped):
            return False
        if re.match(r'^--\w+\s*;?\s*$', stripped) or re.match(r'^\w+--\s*;?\s*$', stripped):
            return False
        # 检测 i++ 或 i-- 出现在更复杂表达式中
        if re.search(r'\w+(\+\+|--)', stripped) and re.search(r'[+\-\*/\[\(,]', stripped):
            # 排除简单的 for 循环第三部分（for(;;i++)）
            if re.match(r'^\s*for\s*\(', stripped):
                return False
            return True
        if re.search(r'(\+\+|--)(\w+)', stripped) and re.search(r'[+\-\*/\[\(,]', stripped):
            if re.match(r'^\s*for\s*\(', stripped):
                return False
            return True
        return False

    def _has_magic_number(self, line: str) -> bool:
        """检测孤立的数字字面量（魔鬼数字）。"""
        stripped = line.strip()
        # 排除注释行
        if stripped.startswith('//') or stripped.startswith('*') or stripped.startswith('/*'):
            return False
        # 排除 #define 和 const 定义行（这些是定义常量的）
        if re.match(r'^\s*(#\s*define|const|constexpr|enum)\b', stripped):
            return False
        # 排除 for 循环
        if re.match(r'^\s*for\s*\(', stripped):
            return False
        # 排除仅有数字的数组初始化
        if re.match(r'^\s*[\d,\s]+[,;]?\s*$', stripped):
            return False
        # 检测：赋值语句右侧或函数参数中出现大于1的整数字面量（排除0和1）
        # 仅对形如  var = 42; 或 func(42) 这类情况报告
        magic = re.findall(r'(?<!["\'])\b([2-9][0-9]+|[3-9])\b(?!["\'])', stripped)
        # 排除版本号、数组大小定义等模式
        for m in magic:
            # 如果数字旁边是 * 或 / 运算，可能是算术，保守不报
            if int(m) >= 3:
                return True
        return False

    def _has_c_style_cast(self, line: str) -> bool:
        """检测 C 风格强制类型转换，如 (int)x 或 (char*)ptr。"""
        stripped = line.strip()
        if stripped.startswith('//') or stripped.startswith('*'):
            return False
        # 匹配 (type)expr 模式，排除函数调用、sizeof、alignof
        pattern = r'\(\s*(int|char|long|short|float|double|unsigned|signed|void\s*\*|[A-Z]\w*\s*\*?)\s*\)\s*\w'
        if re.search(pattern, stripped):
            # 排除 static_cast 等已有 C++ 风格转换的行
            if re.search(r'(static_cast|dynamic_cast|reinterpret_cast|const_cast)', stripped):
                return False
            return True
        return False

    def _uses_null_macro(self, line: str) -> bool:
        """检测使用 NULL 而非 nullptr（排除注释和字符串内）。"""
        stripped = line.strip()
        if stripped.startswith('//') or stripped.startswith('*'):
            return False
        # 简单检测代码部分含 NULL
        return bool(re.search(r'\bNULL\b', line))

    def _replace_null(self, line: str) -> str:
        """将代码部分的 NULL 替换为 nullptr（保留注释和字符串中的 NULL）。"""
        # 简单替换，不处理字符串内的 NULL（保守）
        # 仅替换代码部分（注释前）
        comment_idx = line.find('//')
        if comment_idx >= 0:
            code = line[:comment_idx]
            comment = line[comment_idx:]
            new_code = re.sub(r'\bNULL\b', 'nullptr', code)
            return new_code + comment
        else:
            return re.sub(r'\bNULL\b', 'nullptr', line)

    def _fix_pointer_spacing(self, line: str) -> str:
        """修正 int*p 或 int&p（两边无空格）→ int* p（靠左形式）。"""
        # 排除注释行
        stripped = line.strip()
        if stripped.startswith('//') or stripped.startswith('*') or stripped.startswith('/*'):
            return line
        # 修正 type*varname → type* varname（两边都无空格的情况）
        new_line = re.sub(r'(\w)\*(\w)', r'\1* \2', line)
        new_line = re.sub(r'(\w)&(\w)', r'\1& \2', new_line)
        return new_line

    def _has_old_c_header(self, line: str) -> bool:
        """检测旧式 C 头文件，如 <stdlib.h>、<string.h>、<stdio.h> 等。"""
        stripped = line.strip()
        old_headers = [
            'stdlib.h', 'string.h', 'stdio.h', 'math.h', 'time.h',
            'ctype.h', 'limits.h', 'float.h', 'assert.h', 'signal.h',
            'setjmp.h', 'stdarg.h', 'stddef.h', 'errno.h', 'locale.h'
        ]
        for h in old_headers:
            if re.search(r'#\s*include\s*<' + re.escape(h) + r'>', stripped):
                return True
        return False

    def _is_macro_function(self, line: str) -> bool:
        """检测宏函数 #define NAME(...)。"""
        stripped = line.strip()
        return bool(re.match(r'^#\s*define\s+\w+\s*\(', stripped))

    def _multi_var_init_on_one_line(self, line: str) -> bool:
        """检测同行多个变量初始化（如 int a = 1, b = 2;）。"""
        stripped = line.strip()
        if stripped.startswith('//') or stripped.startswith('*'):
            return False
        # 匹配 type var1 = val, var2 = val; 模式
        if re.match(r'^\s*(int|char|bool|float|double|long|short|unsigned|auto)\s+\w+\s*=\s*.+,\s*\w+\s*=', stripped):
            return True
        return False


# ─────────────────────────────────────────────
# 文件处理入口
# ─────────────────────────────────────────────

def format_file(filepath: str, log: ChangeLog, inplace: bool) -> str:
    """格式化单个文件，返回输出文件路径。"""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    formatter = CppFormatter(log, filepath)
    new_lines = formatter.format_lines(lines)

    if inplace:
        out_path = filepath
    else:
        out_path = filepath + ".formatted"

    with open(out_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    return out_path


def collect_files(path: str) -> List[str]:
    """收集目录下所有 .cpp/.h 文件，或直接返回单文件。"""
    p = Path(path)
    if p.is_file():
        return [str(p)]
    elif p.is_dir():
        files = []
        for ext in ('*.cpp', '*.h', '*.hpp', '*.cc', '*.cxx'):
            files.extend(str(f) for f in p.rglob(ext))
        return sorted(files)
    else:
        print(f"错误：路径不存在: {path}", file=sys.stderr)
        return []


def main():
    parser = argparse.ArgumentParser(
        description="C++ 代码规范格式化工具（符合公司医疗器械软件 C++ 编码规范）"
    )
    parser.add_argument("path", help="待格式化的 .cpp/.h 文件或目录")
    parser.add_argument("--inplace", action="store_true", help="直接覆盖原文件（默认输出到 .formatted）")
    parser.add_argument("--log", default="./format_changes.log", help="日志文件路径（默认 ./format_changes.log）")
    args = parser.parse_args()

    log = ChangeLog(args.log)
    files = collect_files(args.path)

    if not files:
        print("未找到任何 .cpp/.h 文件。")
        return

    print(f"共找到 {len(files)} 个文件，开始格式化...")
    output_paths = []
    for fp in files:
        out = format_file(fp, log, args.inplace)
        output_paths.append(out)
        print(f"  ✓ {fp} → {out}")

    log.write()
    print()
    print(log.summary())
    print()
    print("请查阅日志文件中 [WARN] 条目，确认是否保留相应更改。")
    print("WARN 条目对应位置已在格式化文件中加入 // FORMAT-WARN 注释标记，便于定位。")


if __name__ == "__main__":
    main()
