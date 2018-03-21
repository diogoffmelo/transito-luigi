import luigi
import requests

class RemoteRequestTask(luigi.Task):
    remote_path = luigi.Parameter()
    local_path = luigi.Parameter()
    method = luigi.Parameter()

    def requires(self):
        return []

    def output(self):
        return luigi.LocalTarget(self.local_path)

    def run(self):
        response = getattr(requests, self.method.lower())(self.remote_path)
        with self.output().open('w') as f:
            f.writelines(response.text)


if __name__ == '__main__':
    luigi.run()
    