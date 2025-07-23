"""
因子存储模块

提供统一的因子测试结果存储和管理功能
"""

import sys
from pathlib import Path

# 添加项目根目录到sys.path，使utils成为可导入的模块
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# from utils.file_utils import update_json_file
#
# __all__ = ['update_json_file']
