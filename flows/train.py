import luigi
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from encoding import FeaturesAndTargetEncoder

class Train(luigi.Task):
    local_path = luigi.Parameter()
    encoding_specs = {
        'local_ypath': './Yencoded.pkl',
        'local_xpath': './Xencoded.pkl'
    }

    classifier_specs = {}

    def requires(self):
        return [FeaturesAndTargetEncoder(**self.encoding_specs)]

    def output(self):
        return luigi.LocalTarget(self.local_path)

    def run(self):
        Y = pd.read_pickle(self.encoding_specs['local_ypath'])
        X = pd.read_pickle(self.encoding_specs['local_xpath'])

        model = RandomForestRegressor()
        model.fit(X, Y)
 
        with open(self.output().path, 'wb') as f:
            pickle.dump(model, f)

from sklearn.metrics import (explained_variance_score,
                             median_absolute_error,
                             r2_score,
                             mean_squared_error)


class Report(luigi.Task):
    local_path = luigi.Parameter()
    training_specs = {
        'local_path': './model.pkl'
    }

    def requires(self):
        return [Train(**self.training_specs)]

    def output(self):
        return luigi.LocalTarget(self.local_path)

    def run(self):
        Y = pd.read_pickle('./Yencoded.pkl')
        X = pd.read_pickle('./Xencoded.pkl')

        with self.output().open('w') as fout:
            with open(self.training_specs['local_path'], 'rb') as modelf:
                model = pickle.load(modelf)

                print('-'*80, file=fout)
                print(model, file=fout)
                ypred = model.predict(X)
                print('Errors report:')
                print('MSR={:>5.3f}\t\tMAR={:>5.3f}\t\tEVS={:>5.3f}\t\tR2={:>5.3f}'.format(
                    mean_squared_error(Y, ypred),
                    median_absolute_error(Y, ypred),
                    explained_variance_score(Y, ypred),
                    r2_score(Y, ypred)
                ), file=fout)
                print('-'*80, file=fout)

    
if __name__ == '__main__':
    luigi.run()
