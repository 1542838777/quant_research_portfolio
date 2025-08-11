import datetime
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import yaml

from projects._03_factor_selection.config.config_file.local_config_file_definition import \
    pool_for_massive_test_CSI800_profile, pool_for_massive_test_MICROSTRUCTURE_profile, generate_dynamic_config, \
    CSI300_most_basic_profile, CSI300_none_FFF_most_basic_profile, CSI300_more_filter_profile, \
    CSI1000_more_filter_profile, CSI500_none_FFF_most_basic_profile
from quant_lib.config.logger_config import log_warning
fast_periods = ('20250424','20250710')
fast_periods_2 = ('20240301','20250710')
self_periods = ('20220101','20250710')
longest_periods = ('20190328','20250710')


massive_test_mode = {
    'mode': 'massive_test',
    'pools': {
        **pool_for_massive_test_CSI800_profile,
        **pool_for_massive_test_MICROSTRUCTURE_profile
    },
    'period': longest_periods,
    'desc': '海量测试环境-用了沪深800 和 全A 股票池 （这是最真实的环境'
}

CSI300_most_basic_mode = {
    'mode': 'CSI300_most_basic_profile',
    'pools': {
        **CSI300_most_basic_profile
    },
    'period':self_periods,
    'desc': '但是只用了沪深300股票池（）只有普适性过滤，除此之外，没有任何过滤'
}

fast_mode = {
    'mode': 'fast',
    'pools': {
        **CSI300_none_FFF_most_basic_profile
    },
    'period':fast_periods,
    'desc': '但是只用了沪深300股票池（） ，没有任何过滤 fast'
}


fast_mode_2 = {
    'mode': 'fast',
    'pools': {
        **CSI300_none_FFF_most_basic_profile
    },
    'period':fast_periods_2,
    'desc': '但是只用了沪深300股票池（） ，没有任何过滤 fast'
}

fast_mode_two_pools = {
    'mode': 'fast',
    'pools': {
        **CSI300_none_FFF_most_basic_profile,
        **CSI500_none_FFF_most_basic_profile
    },
    'period':fast_periods,
    'desc': '但是只用了沪深300股票池（） ，没有任何过滤 fast'
}

CSI300_more_filter_mode = {
    'mode': 'CSI300_most_basic_profile',
    'pools': {
        **CSI300_more_filter_profile
    },
    'period':self_periods,
    'desc': '但是只用了沪深300股票池（）普适性过滤+流动率过滤'
}

东北证券_CSI300_more_filter_mode = {
    'mode': 'CSI300_most_basic_profile',
    'pools': {
        **CSI300_more_filter_profile
    },
    'period':self_periods,
    'desc': '但是只用了沪深300股票池（）普适性过滤+流动率过滤'
}

东北证券_CSI1000_more_filter_mode = {
    'mode': '东北证券_CSI1000_more_filter_mode',
    'pools': {
        **CSI1000_more_filter_profile
    },
    'period':self_periods,
    'desc': 'CSI1000（）普适性过滤+流动率过滤'
}
CSI300_FFF_most_basic_mode = {
    'mode': 'CSI300_FFF_most_basic_mode',
    'pools': {
        **CSI300_none_FFF_most_basic_profile
    },
    'period':self_periods,
    'desc': '但是只用了沪深300股票池（）无普适性过滤，，没有任何过滤'
}

def check_backtest_periods(start_date, end_date):
    if pd.to_datetime(end_date) - pd.to_datetime(start_date) < datetime.timedelta(days=110):
        raise ValueError("回测时间太短")

















trans_pram =东北证券_CSI1000_more_filter_mode
is_debug = True
















def _load_local_config(config_path: str) -> Dict[str, Any]:
    # confirm_production_mode(massive_test_mode)
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
    log_warning(f"【信息】当前处于 {trans_pram['mode']} 模式，desp: {trans_pram['desc']}。")

    target_factors_for_evaluation_fields = config['target_factors_for_evaluation']['fields']
    start, end = trans_pram['period']

    dynamic_config = generate_dynamic_config(
        start_date=  start,end_date=end,
        target_factors = target_factors_for_evaluation_fields,
        pool_profiles =  trans_pram['pools']  # 直接取用 dict
    )
    config['backtest']['start_date'] = start
    config['backtest']['end_date'] =end

    config['stock_pool_profiles']=dynamic_config['stock_pool_profiles']
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
            print("\\n操作已由用户终止。")
            exit()  # 直接退出程序
        print("继续执行调试模式任务...")
if __name__ == '__main__':
    config = _load_local_config('D:\\lqs\\codeAbout\\py\\Quantitative\\quant_research_portfolio\\projects\\_03_factor_selection\\factory\\config.yaml')
