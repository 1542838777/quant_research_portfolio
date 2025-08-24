import json


#ic_series_processed_60d.parquet
# def build_targer_name(core_eveluation_type,is_raw_factor, forward_period):
#
#     if core_eveluation_type == 'ic':
#         if is_raw_factor:
#             return f'ic_raw_{forward_period}d'
#         else:
#             return f'ic_processed_{forward_period}d'
#     elif core_eveluation_type == 'tmb':
#         if is_raw_factor:
#             return f'tmb_raw_{forward_period}d'
#         else:
#             return f'tmb_processed_{forward_period}d'
#     elif core_eveluation_type == 'monotonicity':
#     pass


def load_summary_stats(param):
    d1 = json.load(open(param, 'r', encoding='utf-8'))
    return d1
def load_ic_stats(json:json=None,is_raw_factor:bool=False):
    subfix='_processed'
    if is_raw_factor:
        subfix='_raw'
    ic_stas =json.get(f'ic_analysis{subfix}')
    return ic_stas

class ResultLoadManager:
    main_work_path='D:\\lqs\\codeAbout\\py\\Quantitative\\quant_research_portfolio\\projects\\_03_factor_selection\\workspace\\result'
    #D:\lqs\codeAbout\py\Quantitative\quant_research_portfolio\projects\_03_factor_selection\workspace\result\
    # 000906\volatility_90d\c2c\20190328_20231231
    #多periods
    def get_ic_stats_from_local(self,  stock_pool_name,factor_name,calcu_type='c2c',version='20190328_20231231',core_eveluation_type='ic',is_raw_factor:bool=False):
        path = self.main_work_path + f'/{stock_pool_name}/{factor_name}/{calcu_type}/{version}'
        ret = load_summary_stats(path+'/summary_stats.json')
        ic_stas = load_ic_stats(ret,is_raw_factor)
        return ic_stas
if __name__ == '__main__':
    ResultLoadManager().get_ic_stats_from_local('000906','volatility_90d','c2c','20190328_20231231','ic',False)