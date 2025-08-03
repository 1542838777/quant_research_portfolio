import datetime
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import yaml

from quant_lib.config.logger_config import log_warning

# RUN_MODE = 'for_fast_test'
RUN_MODE = ('for_relly_but_one_pool','实测环境-但是只用了沪深300股票池')


def check_backtest_periods(start_date, end_date):
    if pd.to_datetime(end_date) - pd.to_datetime(start_date) < datetime.timedelta(days=110):
        raise ValueError("回测时间太短")


def _load_local_config(config_path: str) -> Dict[str, Any]:
    confirm_production_mode(RUN_MODE)
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
    if RUN_MODE[0] =='for_relly_but_one_pool':
        log_warning(f"【信息】当前处于{RUN_MODE}模式，desp:{RUN_MODE[1]}。")
        stock_pool_profiles = config['stock_pool_profiles']
        first_pool_profile_name = next(iter(stock_pool_profiles))
        stock_pool_profile = stock_pool_profiles[first_pool_profile_name]

        stock_pool_profile['filters']['remove_new_stocks'] = True
        stock_pool_profile['filters']['remove_st'] = True
        stock_pool_profile['filters']['adapt_tradeable_matrix_by_suspend_resume'] = True
        config['stock_pool_profiles']={first_pool_profile_name: stock_pool_profile}

    else:
        print("【信息】当前处于生产模式，所有过滤器已启用。")
        check_backtest_periods(config['backtest']['start_date'], config['backtest']['end_date'])
        stock_pool_profiles = config['stock_pool_profiles']
        for pool_name,pool_config  in stock_pool_profiles.items():
            pool_config['filters']['remove_new_stocks'] = True
            pool_config['filters']['remove_st'] = True
            pool_config['filters']['adapt_tradeable_matrix_by_suspend_resume'] = True

    return config
class stock_pool_profile():
    pool_name: str
    index_filter_profile:Dict[str, object]
class index_filter_profile():
    enable:bool
    index_code:str


def self_define(stock_pool_profile_list):
    #实现这里的逻辑 todo
    return  config


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
            exit()  # 直接退出程序
        print("继续执行调试模式任务...")
if __name__ == '__main__':
    config = _load_local_config('config.yaml')
    config.get('forward_periods', [1, 5, 10, 20])