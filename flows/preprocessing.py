import luigi
import pandas as pd
import re

from download import RemoteRequestTask

class PreProcessRawData(luigi.Task):
    local_path = luigi.Parameter()

    download_specs = {
        'remote_path': 'http://dados.recife.pe.gov.br/dataset/2bd87e17-d9f6-40d9-8171-52e170adf4c4/resource/3c715480-67e4-489a-99d4-c01913c92996/download/fluxo-de-veiculo.csv',
        'method': 'get',
        'local_path': './transito_recife_02122016.csv', 
    }

    def requires(self):
        return [RemoteRequestTask(**self.download_specs)]

    def output(self):
        return luigi.LocalTarget(self.local_path)

    def run(self):
        csv_path = self.download_specs['local_path']

        df = pd.read_csv(csv_path, sep=';')
        df.dropna(axis=0, how='any', inplace=True)
        names_mapper = {
            'Logradouro': 'place', 
            'Intervalo de Datas': 'day',
            'Hora': 'time_of_day',
            'Caminhão grande': 'huge_truck',
            'Caminhão md/grd': 'big_truck',
            'Caminhão peq/md': 'medium_truck',
            'Caminhão pequeno': 'small_truck',
            'Moto': 'motocycle',
            'Não Reconhecido': 'other',
            'Ônibus grande': 'big_bus',
            'Ônibus pequeno': 'small_bus',
            'Passeio grande': 'big_car',
            'Passeio pequeno': 'small_car',
        }
        vehicules_columns = [
            'huge_truck',
            'big_truck',
            'medium_truck',
            'small_truck',
            'motocycle',
            'other',
            'big_bus',
            'small_bus',
            'big_car',
            'small_car'
        ]
        df.rename(mapper=names_mapper, inplace=True, axis='columns')
        days = df['day']
        times_of_day = df['time_of_day'].apply(lambda x: x.split('-')[0])
        df['date'] = pd.to_datetime(days + ' ' + times_of_day, dayfirst=True)
        df['day_date'] = pd.to_datetime(days, dayfirst=True)

        def float_hour(str_hour):
            hh, mm, ss = re.findall('\d\d', str_hour)
            return int(hh) + int(mm)/60.0 + int(ss)/3600.0

        df['hour_date'] = times_of_day.apply(float_hour)
        df['total'] = df[vehicules_columns].sum(axis=1)

        df.to_pickle(self.local_path)


class PreProcessDataToProblemFormat(luigi.Task):
    local_path = luigi.Parameter()

    preprocess_specs = {
        'local_path': './preprocessing.pkl', 
    }

    def requires(self):
        return [PreProcessRawData(**self.preprocess_specs)]

    def output(self):
        return luigi.LocalTarget(self.local_path)

    def run(self):
        pkl_path = self.preprocess_specs['local_path']
        df = pd.read_pickle(pkl_path)

        vehicules_columns = [
            'huge_truck',
            'big_truck',
            'medium_truck',
            'small_truck',
            'motocycle',
            'other',
            'big_bus',
            'small_bus',
            'big_car',
            'small_car',
            'total'
        ]

        def shiftted(group):
            dsorted = group.sort_values('date')
            d0 = dsorted[2:].reset_index(drop=True)

            d1 = dsorted[1:-1][vehicules_columns].reset_index(drop=True)   
            d1.columns = [c + '_1' for c in vehicules_columns]            
            
            d2 = dsorted[:-2][vehicules_columns].reset_index(drop=True)
            d2.columns = [c + '_2' for c in vehicules_columns]
            
            return pd.concat([d0, d1, d2], axis=1)

        df = df.groupby(['place', 'day']).apply(shiftted).reset_index(drop=True)
        df.to_pickle(self.local_path)


if __name__ == '__main__':
    luigi.run()
