from typing import Tuple

import pandas as pd

from utils.setup import Config


def get_trade_and_book_by_stock_and_time_id(stock_id: int, time_id: int = None,
                                            datatype: str = 'train') -> Tuple[pd.DataFrame, pd.DataFrame]:
    book_example = pd.read_parquet(f'{Config.data_directory}book_{datatype}.parquet/stock_id={stock_id}')
    trade_example = pd.read_parquet(f'{Config.data_directory}trade_{datatype}.parquet/stock_id={stock_id}')
    if time_id:
        book_example = book_example[book_example['time_id'] == time_id]
        trade_example = trade_example[trade_example['time_id'] == time_id]
    book_example.loc[:, 'stock_id'] = stock_id
    trade_example.loc[:, 'stock_id'] = stock_id
    return book_example, trade_example
