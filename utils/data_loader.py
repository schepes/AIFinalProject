import pandas as pd

class DataLoader:
    def __init__(self, base_path):
        self.base_path = base_path

    def load_data(self):
        # Reading the data
        df1 = pd.read_csv(f'{self.base_path}/train_data.tsv', sep='\t')
        df2 = pd.read_csv(f'{self.base_path}/test_data.tsv', sep='\t')
        df3 = pd.read_csv(f'{self.base_path}/Validation_data.tsv', sep='\t')

        # Adding column names
        columns = ['index', 'id', 'label', 'statement', 'subject', 'speaker', 'JobTitle', 'State', 'Party', 'BTC', 'FC', 'HT', 'MT', 'POF', 'context', 'justification']
        df1.columns = columns
        df2.columns = columns
        df3.columns = columns

        # Merging datasets
        df = pd.concat([df1, df2, df3], axis=0)

        # Additional preprocessing steps (if any) should be added here

        return df
