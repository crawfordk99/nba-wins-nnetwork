from dataapi import *
def main():
    data_instance = DataApi([2017, 2018, 2019, 2020, 2021, 2022])
    seasons_data: pd.DataFrame = data_instance.run()


if __name__ == "main":
    main()