import luigi
import pandas as pd

import pickle

from preprocessing import PreProcessDataToProblemFormat
from tools import IdentityEncoder, CategoricalEncoder


class DataTransform(luigi.Task):
    local_path = luigi.Parameter()
    preprocessing_specs = {
        'local_path': './shiftted.pkl', 
    }

    def requires(self):
        return [PreProcessDataToProblemFormat(**self.preprocessing_specs)]

    def output(self):
        return luigi.LocalTarget(self.local_path)

    def run(self):
        pkl_path = self.preprocessing_specs['local_path']
        df = pd.read_pickle(pkl_path)
        column_encoder = {}
        column_encodded = {}
        for column, encoder in self.feature_encoders.items():
            column_encodded[column] = encoder.fit_transform(df[column])
            column_encoder[column] = encoder

        pd.DataFrame(column_encodded).to_pickle(self.local_path)


class TargetEncoder(DataTransform):
    feature_encoders = {
        'total': IdentityEncoder(),
    }


class FeaturesEncoder(DataTransform):
    feature_encoders = {
        'time_of_day': CategoricalEncoder(),
        'huge_truck_1': IdentityEncoder(),
        'big_truck_1': IdentityEncoder(),
        'medium_truck_1': IdentityEncoder(),
        'small_truck_1': IdentityEncoder(),
        'motocycle_1': IdentityEncoder(),
        'other_1': IdentityEncoder(),
        'big_bus_1': IdentityEncoder(),
        'small_bus_1': IdentityEncoder(),
        'big_car_1': IdentityEncoder(),
        'small_car_1': IdentityEncoder(),
        'total_1': IdentityEncoder(),
        'huge_truck_2': IdentityEncoder(),
        'big_truck_2': IdentityEncoder(),
        'medium_truck_2': IdentityEncoder(),
        'small_truck_2': IdentityEncoder(),
        'motocycle_2': IdentityEncoder(),
        'other_2': IdentityEncoder(),
        'big_bus_2': IdentityEncoder(),
        'small_bus_2': IdentityEncoder(),
        'big_car_2': IdentityEncoder(),
        'small_car_2': IdentityEncoder(),
        'total_2': IdentityEncoder(),
    }


class FeaturesAndTargetEncoder(luigi.WrapperTask):
    local_ypath = luigi.Parameter()
    local_xpath = luigi.Parameter()

    def run(self):
        print('running encoders....')

    def requires(self):
        yield FeaturesEncoder(local_path=self.local_xpath)
        yield TargetEncoder(local_path=self.local_ypath)


if __name__ == '__main__':
    luigi.run()
