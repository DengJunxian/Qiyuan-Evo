import sys
import os
import subprocess
import importlib.util


MIN_PYTHON = (3, 11)
REQUIRED_UI_MODULES = {
    "streamlit": "streamlit",
    "pandas": "pandas",
    "numpy": "numpy",
    "plotly": "plotly",
    "yaml": "pyyaml",
    "openai": "openai",
    "akshare": "akshare",
}


def check_python_version() -> bool:
    """Fail fast with a readable message when the runtime is too old."""

    if sys.version_info >= MIN_PYTHON:
        return True
    current = ".".join(str(part) for part in sys.version_info[:3])
    required = ".".join(str(part) for part in MIN_PYTHON)
    print(f"[ERROR] 当前 Python 版本为 {current}，项目需要 Python {required} 及以上。")
    print("[HINT] 建议创建新环境后安装依赖：")
    print("       python3.11 -m venv .venv")
    print("       source .venv/bin/activate")
    print("       python -m pip install -r requirements.txt")
    return False


def check_required_modules() -> bool:
    """Check front-end dependencies before spawning Streamlit."""

    missing = [
        package_name
        for module_name, package_name in REQUIRED_UI_MODULES.items()
        if importlib.util.find_spec(module_name) is None
    ]
    if not missing:
        return True
    print("[ERROR] 缺少启动界面所需依赖：")
    for package_name in missing:
        print(f"       - {package_name}")
    print("[HINT] 请在 Python 3.11+ 虚拟环境中执行：")
    print("       python -m pip install -r requirements.txt")
    return False


def check_environment():
    """检查运行环境依赖"""
    # 局部导入以避免提前加载
    import os
    import importlib
    import config
    
    cfg = config.GLOBAL_CONFIG
    print(f"[*] 初始化 {cfg.PROJECT_NAME} (v{cfg.VERSION})")
    
        # 检查接口密钥
    if not cfg.DEEPSEEK_API_KEY:
        print("\n[!] 警告: 未检测到 DEEPSEEK_API_KEY 环境变量。")
        non_interactive = (not sys.stdin) or (not sys.stdin.isatty())
        if non_interactive:
            print("[*] 检测到非交互终端，自动跳过 API Key 输入并进入离线可演示模式。")
            key = ""
        else:
            key = input("请输入 DeepSeek API Key (回车跳过使用默认/Mock): ").strip()
        if key:
            # 设置环境变量
            os.environ["DEEPSEEK_API_KEY"] = key
            
            # 重新加载 配置
            importlib.reload(config)
            cfg = config.GLOBAL_CONFIG
            print("[*] API Key 已配置并重新加载 GLOBAL_CONFIG")
        else:
            print("[!] 继续运行，部分AI功能将不可用。")
    return cfg

def run_ui():
    """启动 Streamlit 界面"""
    print("[*] 正在启动可视化控制台...")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "app.py")
    
    if not os.path.exists(file_path):
        print(f"[ERROR] 找不到文件: {file_path}")
        return

    try:
        env = os.environ.copy()
        cmd = [sys.executable, "-m", "streamlit", "run", file_path]
        print(f"Executing: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, env=env)
    except Exception as e:
        print(f"[!] 启动失败: {e}")

if __name__ == "__main__":
    if not check_python_version():
        sys.exit(1)
    if not check_required_modules():
        sys.exit(1)
    GLOBAL_CONFIG = check_environment()
    run_ui()
