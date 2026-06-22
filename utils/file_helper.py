import os


def get_secure_path(project_root: str, target_dir: str) -> str:
    """
    安全解析并拼接路径，严格防止目录穿越
    参数:
    - project_root: config.yaml 中配置的安全根目录
    - target_dir: Agent 传入的目标子目录或文件路径
    """
    abs_root = os.path.abspath(project_root)
    if not target_dir:
        return abs_root
    # 兼容处理：如果传入的已经是绝对路径
    if os.path.isabs(target_dir):
        abs_target = os.path.normpath(target_dir)
    else:
        abs_target = os.path.abspath(os.path.join(abs_root, target_dir))
    # 安全校验：目标路径的绝对格式必须以项目根目录为前缀
    if not abs_target.startswith(abs_root):
        raise PermissionError(f"安全拦截: 拒绝访问超出项目根目录的工作区 ({target_dir})")
    return abs_target