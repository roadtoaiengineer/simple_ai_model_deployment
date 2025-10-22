import pandas as pd 


class Preprocessor:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.parse_csv()

    def parse_csv(self):
        self.df = pd.read_csv(self.csv_path)

    def clean_df(self):
        # self.df['name'] = self.df['name'].apply(lambda x: x.split(' ')[0])
        self.df.drop(columns=['name','city','points'], inplace=True)

        return self.df
