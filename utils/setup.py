from pathlib import Path


class Config:
    data_directory = Path(__file__).parent.resolve() / '..' / 'input' / 'optiver-realized-volatility-prediction'
    random_seed = 420
    random_state = random_seed
