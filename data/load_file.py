import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import yaml

IS_DEBUG_MODE = True


def _load_config(config_path: str) -> Dict[str, Any]:
    confirm_production_mode(IS_DEBUG_MODE)
    """加载配置文件"""
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"配置文件加载成功: {config_path}")
    else:
        raise RuntimeError("未找到config文件")

    # 根据debug模式 修改内容
    # 在这里，根据总开关来决定你的过滤器配置
    if IS_DEBUG_MODE:
        print("【信息】当前处于debug模式，部分过滤器已暂停（为了更快调试代码而已！。")
        stock_pool_profiles = config['stock_pool_profiles']
        for pool in stock_pool_profiles:
            pool = pool[next(iter(pool))]
            pool['filters']['remove_new_stocks'] = False
            pool['filters']['remove_st'] = False
            pool['filters']['adapt_tradeable_matrix_by_suspend_resume'] = False

    else:
        print("【信息】当前处于生产模式，所有过滤器已启用。")
        stock_pool_profiles = config['stock_pool_profiles']
        for name, pool in stock_pool_profiles.items():
            pool = pool[next(iter(pool))]
            pool['filters']['remove_new_stocks'] = True
            pool['filters']['remove_st'] = True
            pool['filters']['adapt_tradeable_matrix_by_suspend_resume'] = True


    return config

def confirm_production_mode(is_debug_mode: bool, task_name: str = "批量因子测试"):
    """
    一个安全确认函数，防止在调试模式下运行生产级任务。
    """
    if is_debug_mode:
        warning_message = f"""
#################################################################
#                                                               #
#   警告! 警告! 警告!  (WARNING! WARNING! WARNING!)             #
#                                                               #
#   您正准备以【调试模式】(DEBUG MODE)运行 '{task_name}'！        #
#                                                               #
#   在此模式下，ST股、停牌股、新股等关键过滤器已被禁用！            #
#   产出的结果将是【失真】且【不可信】的，仅供快速代码调试使用！    #
#                                                               #
#   如要运行正式测试，请在主脚本中设置 IS_DEBUG_MODE = False      #
#                                                               #
#################################################################
"""
        print(warning_message)
        print("程序将在5秒后继续，以便您有时间终止。")
        seconds = 1
        print("如果您确认要在调试模式下继续，请等待...")
        try:
            for i in range(seconds, 0, -1):
                print(f"...{i}")
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n操作已由用户终止。")
            exit() # 直接退出程序
        print("继续执行调试模式任务...")