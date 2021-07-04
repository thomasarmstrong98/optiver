from typing import Callable, Tuple, Union

import pandas as pd

from utils.setup import Config


def get_trade_and_book_by_stock_and_time_id(stock_id: int, time_id: int = None,
                                            datatype: str = 'train') -> Tuple[pd.DataFrame, pd.DataFrame]:
    book_example = pd.read_parquet( Config.data_directory / f"book_{datatype}.parquet" / f"stock_id={stock_id}")
    trade_example = pd.read_parquet( Config.data_directory / f"trade_{datatype}.parquet" / f"stock_id={stock_id}")
    if time_id:
        book_example = book_example[book_example['time_id'] == time_id]
        trade_example = trade_example[trade_example['time_id'] == time_id]
    book_example.loc[:, 'stock_id'] = stock_id
    trade_example.loc[:, 'stock_id'] = stock_id
    return book_example, trade_example

def load_training_target_data(stock_id: Union[int, None] = None, time_id: Union[int, None] = None) -> pd.DataFrame:
    train = pd.read_csv(Config.data_directory / "train.csv")

    if stock_id is not None:
        train = train[train["stock_id"] == stock_id]
    if time_id is not None:
        train = train[train["time_id"] == time_id]
    return train

def load_training_data() -> pd.DataFrame:
    # load in the target data for the training set so we can see which stock/time pairs we have
    stock_list = load_training_target_data()["stock_id"].unique()

    book_data = []
    trade_data = []


    for id in stock_list:
        book, trade = get_trade_and_book_by_stock_and_time_id(id)
        book_data.append(book)
        trade_data.append(trade)
        
    trade_data = pd.concat(trade_data)
    book_data = pd.concat(book_data)

    return book_data, trade_data

def load_and_apply_training_data(apply_to_book_data: Callable, apply_to_trade_data: Callable) -> pd.DataFrame:
    """
    Loads in trade and book data per stock and applies the supplied functions to those.
    """
    stock_list = load_training_target_data()["stock_id"].unique()

    book_data = []
    trade_data = []


    for id in stock_list:
        book, trade = get_trade_and_book_by_stock_and_time_id(id)
        book_data.append(apply_to_book_data(book))
        trade_data.append(apply_to_trade_data(trade))
        
    trade_data = pd.concat(trade_data)
    book_data = pd.concat(book_data)

    return book_data, trade_data