import json

import pandas as pd
from mako.ext.babelplugin import extract

from projects._03_factor_selection.factor_manager.factor_manager import FactorManager
from quant_lib.config.constant_config import LOCAL_PARQUET_DATA_DIR
from pathlib import Path

from quant_lib.tushare.data.downloader import delete_suffix_index


def extract_monotonicity_spearman(json):
    ret = {}
    for one_period_name,one_period_data in json.items():
        val = round(one_period_data['monotonicity_spearman'], 1)
        # if abs(val) < 0.7:
        #     continue
        need_data = {'val':val}
        ret.update({one_period_name:need_data})
    
    return ret

#提取 单调系数
def load_re():
    RESULTS_PATH = 'D:\\lqs\\codeAbout\\py\\Quantitative\\quant_research_portfolio\\projects\\_03_factor_selection\\workspace\\result'

    base_path = Path(RESULTS_PATH) / '000906'
    ret = []
    for factor_dir in base_path.iterdir():
        path = factor_dir /'o2c/20190328_20250710/summary_stats.json'
        d1 = json.load(open(path, 'r', encoding='utf-8'))

        cur_ret = {
            'name':factor_dir.name,
            # 'ms_raw':extract_monotonicity_spearman(d1['quantile_backtest_raw']),
            'ms_processed':extract_monotonicity_spearman(d1['quantile_backtest_processed'])
        }
        ret.append(cur_ret)
    return ret
if __name__ == '__main__':
    load_re()
