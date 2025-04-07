import pandas as pd
import polars as pl
import sportsdataverse as sdv
import ssl
import urllib.request

# Never do this in production
ssl._create_default_https_context = ssl._create_unverified_context

# Get the data from the sportsdataverse api, and convert it to pandas
class DataApi(): 
    def __init__(self, seasons_list: list[int]) -> None:
        super().__init__()
        self.list = seasons_list
    def run(self) -> pd.DataFrame:
        df = pd.DataFrame()
        try:
            polarsDf = sdv.load_nba_team_boxscore(seasons = self.list, return_as_pandas= False)
            df =  polarsDf.to_pandas()
        except Exception as e: 
            print(f"Data was not loaded in properly {e}")
        return df
# Prepare the data to be processed
def process_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns= df.filter(like= "opponent").columns)
    df = df.drop(columns= df.columns[0])
    df = df.drop(columns = df.columns[3:16])
    drop_columns = ['game_date', 'flagrant_fouls', 'total_technical_fouls', 'season_type', 'team_turnovers', 'total_turnovers', 'fouls']
    df = df.drop(columns = drop_columns)
    df = df.dropna()
    df['fast_break_points'] = df['fast_break_points'].astype(int)
    df['turnover_points'] = df['turnover_points'].astype(int)
    df['largest_lead'] = df['largest_lead'].astype(int)
    df['points_in_paint'] = df['points_in_paint'].astype(int)
    return df

    

