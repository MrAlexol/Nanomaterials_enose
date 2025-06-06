import pandas as pd
import chardet

class FileParser:
    def __init__(self):
        self.csv_delimiter = ';'
        self.csv_decimal = ','
        self.feature_names = list()
        self.target_name = ''

    def set_file_params(self, sep: str, decimal: str) -> None:
        self.csv_delimiter = sep
        self.csv_decimal = decimal

    def set_columns(self, x: list[str], y: str) -> None:
        self.feature_names = x
        self.target_name = y
        
    def load_single_csv(self, file_path) -> pd.DataFrame:
        try:
            df = self.read_csv_auto_encoding(file_path)
        except Exception as e:
            print("Ошибка:", e)

        # Преобразуем все числовые признаки
        for col in self.feature_names:
            if col in df.columns:
                df[col] = df[col].str.replace(',', '.').astype(float)

        # Привести целевой признак к строковому виду
        if self.target_name in df.columns:
            df[self.target_name] = df[self.target_name].astype(str).str.strip()

        return df


    def load_multiple_csvs(self, file_paths: list[str]) -> pd.DataFrame:
        all_dfs = []
        for path in file_paths:
            df = self.load_single_csv(path)
            all_dfs.append(df)
        combined = pd.concat(all_dfs, ignore_index=True)
        return combined

    def read_csv_auto_encoding(self, file_path):
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            detected = chardet.detect(raw_data)
            encoding = detected['encoding']

        try:
            df = pd.read_csv(file_path, encoding=encoding, sep=self.csv_delimiter, decimal=self.csv_decimal, dtype=str)
            return df
        except Exception as e:
            raise ValueError(f"Не удалось прочитать CSV-файл. Кодировка: {encoding}. Ошибка: {e}")


        
