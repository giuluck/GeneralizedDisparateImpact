import numpy as np

from data_generator import DataGenerator


def f_y(data, parents):
    """Generative function for variable y"""
    assert all(x in data.keys() for x in parents), "Missing parents!"
    q_hat = data['q_hat']
    y = 3 + np.exp(q_hat)
    return y


def f_q(data, parents):
    """Generative function for variable q"""
    assert all(x in data.keys() for x in parents), "Missing parents!"
    p, q_hat = data['p'], data['q_hat']
    q = q_hat + 2 * p
    return q


if __name__ == '__main__':
    parents = {
        'p': (),
        'q_hat': (),
        'q': ('p', 'q_hat'),
        'y': ('q_hat',)}

    priors = {
        'p': {
            'name': 'uniform',
            'args': [2, 3]},
        'q_hat': {
            'name': 'gauss',
            'args': [0, 1]
        }
    }

    gen_fn = {'q': f_q,
              'y': f_y}

    generator = DataGenerator(parents=parents, generative_fn=gen_fn, prior_distr=priors)
    df = generator.generate()
    print(df)
    generator.view_graph()
