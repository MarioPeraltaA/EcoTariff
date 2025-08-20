"""Read and process electrical tariffs of national utilities.

Load historical tariffs data since year 2013 and
sort out its content in a more compact and simple
format such as .json so it can be handled by any LLM easily
with entries like::

    utilities = {
        "CompanyName": {
            "Year": {
                "TariffType": {
                    "Energy": [],
                    "Power": [],
                    "Others": []
                }
            }
        }
    }

Where the lists data are float such lists have length
of twelve due to each month of the year.

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json


class Chewer:
    """Cope with format and parsing."""

    def __init__(self, excel_path='./data/tarifas.xlsx'):
        """From creepy data structure to fancy one."""
        self.excel_path = excel_path

    def split_blocks(self, df: pd.DataFrame) -> dict[str, list[float]]:
        """Break block features down."""
        features = {}
        for (_, field) in df.iterrows():
            field = field.replace(int(0), np.nan)
            bool_serie = field.apply(
                (lambda x: False if isinstance(x, str)
                 else (bool(x) if pd.notna(x) else False))
            )
            if bool_serie[1:].any():
                features[field.iloc[0]] = list(field.iloc[1:].values)
        return features

    def split_tariffs(self, df: pd.DataFrame) -> dict[str, dict]:
        """Classify tariffs blocks.

        blocks = {
            "TariffType": {
                "Energy": [],
                "Power": [],
                "Other": []
            }
        }

        """
        def get_tariffs_code_index(df):
            t_indices = []
            for (i, row) in df.iterrows():
                val = row.iloc[0]
                if isinstance(val, str) and 'T-' == val[:2]:
                    t_indices.append(i)
            return t_indices

        t_indices = get_tariffs_code_index(df)
        t_indices.append(len(df) + 1)
        tariffs = {}
        for (start, end) in zip(t_indices, t_indices[1:]):
            tariff_type = df.iloc[(start, 0)]
            tariff_labels.append(
                (df.iloc[(start, 0)], df.iloc[(start - 1, 0)])
            )
            block_df = df.iloc[start + 1:end - 1]
            block_dict = self.split_blocks(block_df)
            tariffs[tariff_type] = block_dict
        return tariffs

    def load_excel(self) -> tuple[dict, dict]:
        """Load and process data."""
        global tariff_labels
        data = pd.read_excel(
            self.excel_path, engine='openpyxl', sheet_name=None
        )
        sheets = list(data.keys())
        utilities_set = set()
        years = set()
        for sheet in sheets:
            if ' ' in sheet:
                (company, year) = sheet.split(' ')
                try:
                    year = int(year)
                except ValueError:
                    continue
                else:
                    years.add(year)
                    utilities_set.add(company)

        utilities_list = list(utilities_set)
        years = list(years)
        years.sort()
        utilities_dict = {}
        tariff_labels = []
        for company in utilities_list:
            years_dict = {}
            for year in years:
                sheet = f'{company} {year}'
                df = data[sheet]
                blocks = self.split_tariffs(df)
                years_dict[year] = blocks
            utilities_dict[company] = years_dict
        labels_dict = self.check_tariff_labels(tariff_labels)
        return (utilities_dict, labels_dict)

    def check_tariff_labels(
            self,
            labels: list[tuple[str, str]]
    ) -> dict[str, list]:
        """Verify one code per block description."""
        labels_dict = {}
        for (code, meaning) in labels:
            if code in labels_dict:
                labels_dict[code].append(meaning)
            else:
                labels_dict[code] = [meaning]

        for (label, description) in labels_dict.items():
            uni_style_descrip = {d.strip().lower() for d in description}
            try:
                if len(uni_style_descrip) > 1:
                    raise ValueError('RepeatedLabel:\n')
            except ValueError as e:
                logg = (
                    f"{e} Same tariff code <{label}> has more "
                    f"than one meaning\n\t{uni_style_descrip}"
                )
                print(logg)
            finally:
                labels_dict[label] = list(uni_style_descrip)
        return labels_dict

    def convert_xlsx_to_json(
            self,
            output_path: str = './data/tariffs.json'
    ):
        """Write out a json file."""
        (data, _) = self.load_excel(self.excel_path)
        with open(output_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=4, ensure_ascii=False)

    def chew_data(self) -> pd.DataFrame:
        """Turn nested into flat structure."""
        (nested_data, block_codes) = self.load_excel()
        flatten_data = {
            'Empresa': [],
            'Año': [],
            'Tarifa ID': [],
            'Nombre bloque': [],
            'Concepto': [],
            'Enero': [],
            'Febrero': [],
            'Marzo': [],
            'Abril': [],
            'Mayo': [],
            'Junio': [],
            'Julio': [],
            'Agosto': [],
            'Septiembre': [],
            'Octubre': [],
            'Noviembre': [],
            'Diciembre': []
        }
        months = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo',
                  'Junio', 'Julio', 'Agosto', 'Septiembre',
                  'Octubre', 'Noviembre', 'Diciembre']
        i_to_month = {i: month for (i, month) in enumerate(months)}

        for company in nested_data.keys():
            for year in nested_data[company].keys():
                for code in nested_data[company][year].keys():
                    for field, vals in (
                        nested_data[company][year][code].items()
                    ):
                        flatten_data['Empresa'].append(company)
                        flatten_data['Año'].append(year)
                        flatten_data['Tarifa ID'].append(code)
                        flatten_data['Nombre bloque'].append(
                            block_codes[code][0]
                        )
                        flatten_data['Concepto'].append(field)
                        for (i, val) in enumerate(vals):
                            month = i_to_month[i]
                            flatten_data[month].append(val)
        df = pd.DataFrame(data=flatten_data)
        return df


class TariffsManager:
    """Hardcore data analysis."""

    def __init__(self, excel_path: str = './data/03 - tariffs.xlsx'):
        """Friendly data structure."""
        self.df = pd.read_excel(excel_path)
        self.evolution: pd.DataFrame | None = None

    def build_time_series(
            self,
            all_years: dict[int, list[pd.DataFrame]]
    ) -> pd.DataFrame:
        """Process data to simple plot."""
        dfs = []

        for (year, dfs_year) in all_years.items():
            for df_year in dfs_year:
                df_long = df_year.melt(
                    id_vars='Concepto',
                    var_name='Mes',
                    value_name='Valor'
                )
                month_map = {
                    'Enero': 1,
                    'Febrero': 2,
                    'Marzo': 3,
                    'Abril': 4,
                    'Mayo': 5,
                    'Junio': 6,
                    'Julio': 7,
                    'Agosto': 8,
                    'Septiembre': 9,
                    'Octubre': 10,
                    'Noviembre': 11,
                    'Diciembre': 12
                }
                df_long['Mes'] = df_long['Mes'].map(month_map)
                df_long['Fecha'] = pd.to_datetime({
                    'year': year,
                    'month': df_long['Mes'],
                    'day': 1
                })
                dfs.append(df_long[['Fecha', 'Concepto', 'Valor']])

        data_all = pd.concat(dfs, ignore_index=True)
        ts_df = data_all.pivot(
            index='Fecha', columns='Concepto', values='Valor'
        )
        ts_df = ts_df.sort_index()
        return ts_df

    def plot_evolution(
            self, ts_df: pd.DataFrame,
            variables_labels: list[str]
    ) -> plt.Axes:
        """Visualize variables over time."""
        for concept in variables_labels:
            ax = ts_df[concept].plot(legend=True, figsize=(12.0, 6.0))
            ax.legend(loc='best')
        plt.show()

    def draw_evolution(
            self,
            companies: list[str] = ['ICE', 'CNFL'],
            tariff_id: str = 'T-MT',
            start_year: int = 2013,
            end_year: int = 2025,
            variable_labels=['ICE - a.Periodo Punta (máxima)',
                             'ICE - c.Periodo Valle (máxima)',
                             'CNFL - a. Energía Punta (Máxima)',
                             'CNFL - c. Energía Valle (Máxima)']
    ) -> pd.DataFrame:
        """Evolution of tariffs history over time.

        Check for new or removed *Concept* during certain year
        as all variables must have same size.

        .. Note::
            Parameter ``end_year`` it is inclusive.

        """
        df = self.df
        years_df = {y: [] for y in range(start_year, end_year + 1)}
        for year in years_df.keys():
            for company in companies:
                condition = ((df['Empresa'] == company)
                             & (df['Año'] == year)
                             & (df['Tarifa ID'] == tariff_id))
                data = df[condition].loc[:, 'Concepto':]
                data['Concepto'] = data['Concepto'].apply(
                    lambda x: f'{company} - {x}'
                )
                years_df[year].append(data)

        ts_df = self.build_time_series(years_df)
        self.plot_evolution(ts_df, variable_labels)
        return ts_df

    def describe_data(
            self
    ) -> pd.DataFrame:
        """Report data statistical metrics."""
        pass


if __name__ == '__main__':
    aresep = TariffsManager()
    peak_labels = ['ICE - Energía_Punta',
                   'CNFL - Energía_Punta',
                   'ICE - Energía Punta',
                   'CNFL - Energía Punta',
                   'ICE - a. Energía Punta',
                   'CNFL - a. Energía Punta',
                   'ICE - a.Periodo Punta (máxima)',
                   'CNFL - a. Energía Punta (Máxima)']
    valley_labels = ['ICE - Energía_Valle',
                     'CNFL - Energía_Valle',
                     'ICE - Energía Valle',
                     'CNFL - Energía Valle',
                     'ICE - b. Energía Valle',
                     'CNFL - b. Energía Valle',
                     'ICE - c.Periodo Valle (máxima)',
                     'CNFL - c. Energía Valle (Máxima)']

    data_ts = aresep.draw_evolution(
        companies=['ICE', 'CNFL'],
        tariff_id='T-MT',
        start_year=2013,
        end_year=2025,
        variable_labels=peak_labels
    )
    aresep.plot_evolution(
        ts_df=data_ts,
        variables_labels=valley_labels
    )
