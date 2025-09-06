from data.local_data_load import load_sw_l_n_codes
from projects._03_factor_selection.config_manager.base_config import config_yaml_path
from projects._03_factor_selection.config_manager.function_load.load_config_file import _load_local_config_functional
from quant_lib.config.constant_config import LOCAL_PARQUET_DATA_DIR
import pandas as pd
import numpy as np
import pandas as pd
def t_lwzy():
    config = _load_local_config_functional(config_yaml_path)
    names =[ curDict['name'] for curDict in  config['factor_definition']]
    local_df = pd.read_csv(r'D:\lqs\codeAbout\py\Quantitative\quant_research_portfolio\projects\_03_factor_selection\factor_manager\selector\o2o_v3.csv')
    ok_columns= local_df['factor_name'].tolist()
    miss_names =set(names)-set(ok_columns)
    print(miss_names)
if __name__ == '__main__':
    t_lwzy()