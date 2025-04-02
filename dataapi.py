import pandas as pd
import polars as pl
import sportsdataverse as sdv

class DataApi(): 
    def __init__(self, seasons_list:list[int]) -> None:
        super().__init__()
        self.list = seasons_list
    def run(self) -> pd.DataFrame:
        df = pd.DataFrame()
        try:
            polarsDf = sdv.load_nba_pbp(seasons = self.list, return_as_pandas= True)
            df =  polarsDf.to_pandas()
        except: 
            print("Data was not loaded in properly")
        return df 


    

