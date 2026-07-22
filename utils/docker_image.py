"""
沙箱镜像 tag 计算（W3-6）

将 requirements.txt 内容的 md5 前 8 位混入镜像 tag：
依赖文件变更 → tag 变化 → warmup 检测不到同名镜像 → 自动触发重建，
替代原先"依赖变了仍复用 latest、需手动 docker rmi"的脆弱流程。
"""
import hashlib
import os


def compute_requirements_hash(req_path: str) -> str:
    """requirements 文件内容的 md5 前 8 位。"""
    with open(req_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:8]


def resolve_sandbox_image(sandbox_cfg: dict) -> str:
    """
    计算沙箱目标镜像引用。
    - 配置了有效 requirements_path：<镜像名>:<requirements md5 前8位>
    - 未配置或文件不存在：<镜像名>:latest（与历史行为一致，基础镜像直打 tag）
    """
    image_id = sandbox_cfg.get("name", "autotune-env")
    req_path = sandbox_cfg.get("requirements_path", "")
    if req_path and os.path.isfile(req_path):
        return f"{image_id}:{compute_requirements_hash(req_path)}"
    return f"{image_id}:latest"
