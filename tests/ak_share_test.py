import akshare as ak


def index_stock_cons_csindex_df(index_code='000300'):
    index_stock_cons_csindex_df = ak.index_csindex_all(symbol=index_code)
    print(index_stock_cons_csindex_df)
if __name__ == '__main__':
    index_stock_cons_csindex_df()