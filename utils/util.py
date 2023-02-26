import numpy as np

def default_input_proc(datapath, verbose):
    '''
    input data (csv) preprocessing for the default format:
    row1
    row2
    ...
    rowN

    where each row has K digits for K dimensions, followed by a numeric label
    -> last col contains all the labels
    e.g. 

    Label ----------- V

    1, 0, 0, 9, 5, 7, 3
    0, 0, 1, 8, 6, 7, 3
    5, 4, 6, 2, 7, 7, 2
    9, 9, 8, 1, 1, 8, 4
    '''

    labels = []
    data = []
    with open(datapath, 'r') as f:
        for line in f:
            temp = line.strip().split(',')
            labels.append(int(temp[-1]))
            data.append(temp[:-1])

    data = np.array(data).astype('int16')

    samples = data.shape[0]
    dim = data.shape[1]
    if verbose:
        print(datapath)
        print('{} rows'.format(samples))
        print('{} dimensions'.format(dim))

    return data, labels
    