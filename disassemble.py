"""
该脚本用于自动创建索引编辑器项目结构和文件内容
从Claude的输出中提取所有代码块并创建相应的文件
"""

import os
import re
import sys

# 要排除的目录名称 (精确匹配)
EXCLUDE_DIRS = {
    '.git',
    '__pycache__',
    '.vscode',
    '.idea',
    'venv',
    '.env',
    'node_modules',
    'build',
    'dist',
    'checkpoints',
    'data',
    # 添加其他你想要排除的目录
}

# 要排除的文件名称 (精确匹配)
EXCLUDE_FILES = {
    '.DS_Store',
    'assemble.py',
    'disassemble.py',
    'response.txt',
    'project_prompt.txt',
    'README.md',
    # 添加其他你想要排除的文件
}

# 要包含的文件扩展名 (如果列表不为空，则只包含这些扩展名的文件)
# 如果为空列表 `[]` 或 `None`，则包含所有未被 EXCLUDE_FILES 排除的文件
INCLUDE_EXTENSIONS = [
    '.py',
    '.md',
    '.txt',
    '.json',
    '.yaml',
    '.yml',
    '.sh',
    # 添加其他你想要包含的文件扩展名
]
#INCLUDE_EXTENSIONS = [] # 取消注释此行以包含所有文件类型

# 项目根目录 (默认是当前脚本所在的目录的父目录，或者就是当前工作目录)
# 通常你会在项目根目录下运行这个脚本
ROOT_DIR = os.getcwd()

# 文件内容提取正则表达式
head = "#" * 3
FILE_PATTERN = "\n" + head + r" (.*?)[\r\n]+[.\r\n]*```.*?\n(.*?)```"


def extract_files_from_content(content):
    """从内容字符串中提取文件路径和代码内容"""
    matches = re.findall(FILE_PATTERN, content, re.DOTALL)
    files = []

    for file_path, code in matches:
        if file_path.find(".") < 0:
            continue
        # 清理文件路径
        length = 0
        for fp in file_path.strip().split(" "):
            if "." in fp and len(fp) > length:
                fp = fp.strip("`")
                reg = re.findall(r'([A-Za-z0-9_]+\.[A-Za-z0-9]+)', fp)
                if reg:
                    length = len(fp)
                    file_name = reg[0]
                    file_path = fp[:fp.find(file_name) + len(file_name)]
                    # file_path = "merge_model/" + file_path

        # 将requirements.txt单独处理
        if file_path == "requirements.txt":
            files.append((file_path, code))
        else:
            files.append((file_path, code))

    return files


def create_project_structure(files):
    """创建项目目录结构并写入文件内容"""
    # 创建项目根目录
    os.makedirs(ROOT_DIR, exist_ok=True)
    print(f"创建项目根目录: {ROOT_DIR}")

    # 创建文件并写入内容
    for file_path, content in files:
        # 构建完整文件路径
        full_path = os.path.join(ROOT_DIR, file_path)

        # 确保目录存在
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        # 写入文件内容
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"创建文件: {full_path}")


def read_claude_response():
    """读取Claude回答内容"""
    print("请粘贴Claude的完整回答，然后按Ctrl+D (Unix/Linux/Mac)或Ctrl+Z (Windows)结束输入:")
    content = sys.stdin.read()
    return content


def generate_project_structure(root_dir, exclude_dirs, exclude_files, include_extensions):
    """生成项目目录结构的字符串表示"""
    tree_lines = []
    ROOT_DIR_name = os.path.basename(root_dir)
    tree_lines.append(f"{ROOT_DIR_name}/")

    # 使用集合进行快速查找
    exclude_dirs_set = set(exclude_dirs)
    exclude_files_set = set(exclude_files)
    include_extensions_set = set(include_extensions) if include_extensions else None

    # 获取脚本自身的文件名，以便在遍历时排除它
    try:
        script_name = os.path.basename(sys.argv[0])
        exclude_files_set.add(script_name)
    except IndexError:
        # 如果无法获取脚本名称（例如在交互式环境中运行），则忽略
        pass

    for root, dirs, files in os.walk(root_dir, topdown=True):
        # 过滤掉需要排除的目录
        dirs[:] = [d for d in dirs if d not in exclude_dirs_set]

        relative_root = os.path.relpath(root, root_dir)
        if relative_root == ".":
            level = -1  # 根目录
        else:
            level = relative_root.count(os.sep)

        indent = '    ' * level  # 使用4个空格作为缩进

        # 添加目录到树结构
        if level >= 0:  # 不重复添加根目录
            dir_name = os.path.basename(root)
            tree_lines.append(f"{indent}├── {dir_name}/")
            indent += '    '  # 下一级文件/目录的连接线

        # 过滤并排序文件
        valid_files = []
        for f in files:
            if f in exclude_files_set:
                continue
            if include_extensions_set:
                _, ext = os.path.splitext(f)
                if ext.lower() not in include_extensions_set:
                    continue
            valid_files.append(f)

        valid_files.sort()  # 保证文件顺序一致

        # 添加文件到树结构
        num_files = len(valid_files)
        for i, f in enumerate(valid_files):
            prefix = '└── ' if i == num_files - 1 else '├── '
            tree_lines.append(f"{indent}{prefix}{f}")

    return "\n".join(tree_lines)


def main():
    # 读取Claude的回答
    # content = read_claude_response()
    with open("response.txt", encoding='utf-8') as fd:
        content = fd.read()

    # 提取文件
    files = extract_files_from_content(content)
    print(f"从回答中提取了 {len(files)} 个文件")

    # 创建项目结构
    create_project_structure(files)

    print(f"\n项目创建完成! 项目位置: {os.path.abspath(ROOT_DIR)}")
    print("目录结构:")
    # 1. 生成项目结构树
    project_structure = generate_project_structure(
        ROOT_DIR,
        EXCLUDE_DIRS,
        EXCLUDE_FILES,
        INCLUDE_EXTENSIONS
    )
    print(project_structure)


if __name__ == "__main__":
    main()
