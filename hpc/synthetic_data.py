import numpy as np

from data_generator import DataGenerator


def generator(fy, fq):
    # fy: f(q_hat) -> y
    # fq: f(p, q_hat) -> q

    def _fy(data, par):
        assert all(x in data.keys() for x in par), "Missing parents!"
        return fy(data['q_hat'])

    def _fq(data, par):
        assert all(x in data.keys() for x in par), "Missing parents!"
        return fq(data['p'], data['q_hat'])

    parents = {
        'p': (),
        'q_hat': (),
        'q': ('p', 'q_hat'),
        'y': ('q_hat',)
    }

    priors = {
        'p': {'name': 'uniform', 'args': [2, 3]},
        'q_hat': {'name': 'gauss', 'args': [0, 1]}
    }

    gen_fn = {'q': _fq, 'y': _fy}

    return DataGenerator(parents=parents, generative_fn=gen_fn, prior_distr=priors)


if __name__ == '__main__':
    g = generator(fy=lambda qh: 3 + np.exp(qh), fq=lambda p, qh: qh + 2 * p)
    df = g.generate()
    print(df)
    # g.view_graph()
