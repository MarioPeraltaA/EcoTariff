"""Read and process electrical tariffs of national utilities.

Repository for cleaning, structuring, and analyzing historical electricity
tariffs with monthly resolution and multi-utility coverage of each block.

Author::

    Mario R. Peralta A.

For feedback::

    mario.peralta@ieee.org

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import yaml


class Chewer:
    """Cope with format and parsing."""

    def __init__(
            self,
            excel_path='./data/raw/01 - tarifas.xlsx'
    ):
        """From creepy data structure to fancy one."""
        self.excel_path = excel_path

    def split_services(
        self,
        df: pd.DataFrame
    ) -> dict[str, list[float]]:
        """Break service features down."""
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

    def split_tariffs(
        self,
        df: pd.DataFrame
    ) -> dict[str, dict]:
        """Classify tariffs services.

        services = {
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
            service_df = df.iloc[start + 1:end - 1]
            service_dict = self.split_services(service_df)
            tariffs[tariff_type] = service_dict
        return tariffs

    def load_excel(
            self
    ) -> tuple[dict, dict]:
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
                services = self.split_tariffs(df)
                years_dict[year] = services
            utilities_dict[company] = years_dict
        labels_dict = self.check_tariff_labels(tariff_labels)
        return (utilities_dict, labels_dict)

    def check_tariff_labels(
            self,
            labels: list[tuple[str, str]]
    ) -> dict[str, list]:
        """Verify one code per service description."""
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
            output_path: str = './data/processed/02 - tariffs.json'
    ):
        """Write out a json file."""
        (data, _) = self.load_excel()
        with open(output_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=4, ensure_ascii=False)

    def chew_data(
            self
    ) -> pd.DataFrame:
        """Turn nested into flat structure."""
        (nested_data, service_codes) = self.load_excel()
        flatten_data = {
            'Empresa': [],
            'Año': [],
            'Tarifa ID': [],
            'Servicio': [],
            'Bloque': [],
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
                        flatten_data['Servicio'].append(
                            service_codes[code][0]
                        )
                        flatten_data['Bloque'].append(field)
                        for (i, val) in enumerate(vals):
                            month = i_to_month[i]
                            flatten_data[month].append(val)
        # Homogenize
        df = pd.DataFrame(data=flatten_data)
        df['Tarifa ID'] = df['Tarifa ID'].apply(lambda x: x.strip().upper())
        return df


class TariffsManager:
    """Hardcore data analysis."""

    def __init__(
            self,
            excel_path: str = "./data/processed/03 - tariffs.xlsx",
            service_groups_path: str = "data/processed/01 - groups.yaml"
    ):
        """Friendly data structure."""
        self.df = pd.read_excel(excel_path)
        self.evolution: pd.DataFrame | None = None
        self.utilities: list[str] | None = None
        self.years: list[int] | None = None
        self.services: list[str] | None = None
        self.blocks: dict[str, list[str]] | None = None
        self.tariff_groups: dict[
            str, dict[str, list[dict[str, str]]]
        ] | None = None
        self.set_fields()
        self.classify_services(service_groups_path)
        self.set_blocks()

    def set_fields(
            self
    ):
        """Reach out columns unique values."""
        self.utilities = self.df['Empresa'].unique().tolist()
        self.years = self.df['Año'].unique().tolist()
        self.services = self.df['Tarifa ID'].unique().tolist()
        self.set_evolution()

    def classify_services(
            self,
            service_groups_path: str = "data/processed/01 - groups.yaml"
    ):
        """Group services code based on ``groups.yaml`` file.

        1. Customer Type / End User Categories
            - Residential:
                - T-RE
                - T-REH
                - T-RH
                - T-RP
            - Commercial / Service:
                - T-CO
                - T-CS
            - Industrial:
                - T-IN
            - General / Other Users:
                - T-GE
                - T-UD
            - Promotional:
                - T-6
                - T-PR

        2. Voltage Level / Connection Type
            - Medium Voltage:
                - T-MT
                - T-MTB
                - T-MTb
                - T-MT69

        3. Special Purpose or Service Type
            - Sales / Distribution sales:
                - T-CB
                - T-CBA
                - T-CBB
                - T-SD
            - Public Lighting:
                - T-AP
            - Electric Vehicle Charging:
                - T-VE
                - T-BE

        4. Tariffs linked to Network/Cost Components
            - Access and Network Charges:
                - T-TA
                - T-TPDx
                - T-TCI
            - Distributed Energy / Generation:
                - T-TDER
                - T-TCVE

        """
        try:
            with open(service_groups_path, mode="r") as file:
                groups = yaml.safe_load(file)
        except FileNotFoundError:
            logg = ("FileNotFoundError: The file "
                    f"{service_groups_path} was not found.")
            print(logg)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
        else:
            self.tariff_groups = groups

    def set_blocks(
            self
    ) -> dict[str, list[str]]:
        """Store historical blocks unique labels.

        Retain record of block's labels given the service
        all across the years, all across the companies.

        Filter blocks (features) associated to certain
        service or type of tariff.

        Returns
        -------
        blocks : dict[str, list[str]]
            Format ``(<utility>)(<service_id>): [<features>]``

        """
        df = self.df
        blocks: dict[str, list[str]] = {}

        for company in self.utilities:
            for service_id in self.services:
                condition = ((df['Empresa'] == company)
                             & (df['Tarifa ID'] == service_id))
                block_labels = df[condition]['Bloque'].unique().tolist()
                if block_labels:
                    blocks[f"({company})({service_id})"] = block_labels
        self.blocks = blocks

    def group_services(
            self
    ) -> dict[str, list[str]]:
        """Group services by blocks.

        All possible blocks throughout the years
        and across companies but under same service
        or type of tariff.

        .. note::

            This can be helpful when it comes to set
            the ``config`` file.

        """
        df = self.df
        type_service: dict[str, list[str]] = {}

        for service_id in self.services:
            service_fts = (
                df[df['Tarifa ID'] == service_id]['Bloque'].apply(
                    lambda x: x.strip().lower()
                )
            )
            service_fts = service_fts.unique().tolist()
            if service_fts:
                type_service[service_id] = service_fts
        return type_service

    def build_time_series(
            self,
            all_years: dict[int, list[pd.DataFrame]]
    ) -> pd.DataFrame:
        """Process data to simple plot."""
        dfs = []

        for (year, dfs_year) in all_years.items():
            for df_year in dfs_year:
                df_long = df_year.melt(
                    id_vars='Bloque',
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
                dfs.append(df_long[['Fecha', 'Bloque', 'Valor']])

        data_all = pd.concat(dfs, ignore_index=True)
        ts_df = data_all.pivot(
            index='Fecha', columns='Bloque', values='Valor'
        )
        ts_df = ts_df.sort_index()
        return ts_df

    def set_evolution(
            self,
    ) -> pd.DataFrame:
        """Evolution of tariffs history over time between utilities.

        Check for new or removed *Block* during certain year
        as all variables must be same size.

        .. Note::
            Parameter ``end_year`` it is inclusive.

        """
        df = self.df
        years_df = {y: [] for y in self.years}
        for year in years_df.keys():
            for company in self.utilities:
                for tariff_id in self.services:
                    condition = ((df['Empresa'] == company)
                                 & (df['Año'] == year)
                                 & (df['Tarifa ID'] == tariff_id))
                    data = df[condition].loc[:, 'Bloque':]
                    data['Bloque'] = data['Bloque'].apply(
                        lambda x: f'({company})({tariff_id}): {x}'
                    )
                    years_df[year].append(data)

        self.evolution = self.build_time_series(years_df)

    def draw_evolution(
            self,
            variable_labels: list[str] = [
                '(ICE)(T-MT): a.Periodo Punta (máxima)',
                '(ICE)(T-MT): c.Periodo Valle (máxima)',
                '(CNFL)(T-MT): a. Energía Punta (Máxima)',
                '(CNFL)(T-MT): c. Energía Valle (Máxima)'
            ]
    ) -> plt.Axes:
        """Visualize variables over time."""
        ts_df = self.evolution
        for block in variable_labels:
            ax = ts_df[block].plot(legend=True, figsize=(12.0, 6.0))
            ax.legend(loc='best')
        plt.show()

    def describe_data(
            self,
            companies: list[str] = ["ICE", "CNFL"],
            **kwargs
    ) -> pd.DataFrame:
        """Fetch tariff code and report data statistical metrics.

        Parameters
        ----------
        companies : list[str]
            Utilities labels.

        kwargs : dict[str, str]
            The *key* refers to the general category of
            group of tariffs in ``groups.yaml`` file while the
            *value* to the subgroup or sector within::

                CustomerType_EndUserCategories[Industrial]

        """
        filter_data: list[str] = []

        for utility in companies:
            for group, sector in kwargs.items():
                services = self.tariff_groups[group][sector]
                codes = [list(entry.keys())[0] for entry in services]
                filter_data.append(f"({utility})({codes})")
                pass
            pass


if __name__ == '__main__':
    aresep = TariffsManager()
    peak_labels = [
        '(ICE)(T-MT): Energía_Punta',
        '(ICE)(T-MT): Energía Punta',
        '(ICE)(T-MT): a. Energía Punta',
        '(ICE)(T-MT): a.Periodo Punta (máxima)',
        '(CNFL)(T-MT): Energía_Punta',
        '(CNFL)(T-MT): Energía Punta',
        '(CNFL)(T-MT): a. Energía Punta',
        '(CNFL)(T-MT): a. Energía Punta (Máxima)'
    ]
    valley_labels = [
        '(ICE)(T-MT): Energía_Valle',
        '(ICE)(T-MT): Energía Valle',
        '(ICE)(T-MT): b. Energía Valle',
        '(ICE)(T-MT): c.Periodo Valle (máxima)',
        '(CNFL)(T-MT): Energía_Valle',
        '(CNFL)(T-MT): Energía Valle',
        '(CNFL)(T-MT): b. Energía Valle',
        '(CNFL)(T-MT): c. Energía Valle (Máxima)'
    ]
    night_labels = [
        '(ICE)(T-MT): Energía_Noche',
        '(ICE)(T-MT): Energía Noche',
        '(ICE)(T-MT): c. Energía Noche',
        '(ICE)(T-MT): e.Periodo Noche (máxima)',
        '(CNFL)(T-MT): Energía_Noche',
        '(CNFL)(T-MT): Energía Noche',
        '(CNFL)(T-MT): c. Energía Noche',
        '(CNFL)(T-MT): e. Energía Noche (Máxima)'
    ]

    aresep.draw_evolution(
        peak_labels
    )
    aresep.draw_evolution(
        valley_labels
    )
    aresep.draw_evolution(
        night_labels
    )
