import sys
from pathlib import Path

# 调试：打印路径信息
current_file = Path(__file__)
project_root = current_file.parent.parent.parent.parent.parent

# print(f"当前文件: {current_file}")
# print(f"项目根目录: {project_root}")
# print(f"utils目录是否存在: {(project_root / 'utils').exists()}")
# print(f"file_utils.py是否存在: {(project_root / 'utils' / 'file_utils.py').exists()}")

sys.path.insert(0, str(project_root))

from utils.file_utils import update_json_file

from utils.file_utils import update_json_file


# 现在可以导入utils模块
def add_single_factor_test_result(filepath, new_data):

    update_json_file(filepath, new_data)
