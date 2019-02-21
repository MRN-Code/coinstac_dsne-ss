"""
Created on Mon Jan 152017

@author: Deb
@notes: Modified by Harsh on 02/15/2018
"""
import numpy as np
import json
from tsneFunctions import normalize_columns, tsne
from itertools import chain
import sys


def local_noop(args):
    input_list = args["input"]

    computation_output = {
        "output": {
            "computation_phase": 'local_noop',
            "no_dims": input_list["no_dims"],
            "initial_dims": input_list["initial_dims"],
            "perplexity": input_list["perplexity"]
        },
        "cache": {
            "no_dims": input_list["no_dims"],
            "initial_dims": input_list["initial_dims"],
            "perplexity": input_list["perplexity"]
        }
    }

    return json.dumps(computation_output)


def local_1(args):
    ''' It will load local data and download remote data and place it on top.
    Then it will run tsne on combined data(shared + local) and return
    low dimensional local site data

    args (dictionary): {
        "shared_X" (str): remote site data,
        "shared_Label" (str):  remote site labels
        "no_dims" (int): Final plotting dimensions,
        "initial_dims" (int): number of dimensions that PCA should produce
        "perplexity" (int): initial guess for nearest neighbor
        "shared_Y" (str):  the low-dimensional remote site data
        }
        computation_phase : local

    Returns:
        localY: It is the two dimensional value of only local site data
    '''

    shared_X = args["input"]["shared_x"]
    shared_Y = args["input"]["shared_y"]

    shared_X = np.array(shared_X)
    shared_Y = np.array(shared_Y)

    no_dims = args["cache"]["no_dims"]
    initial_dims = args["cache"]["initial_dims"]
    perplexity = args["cache"]["perplexity"]
    sharedRows, sharedColumns = shared_X.shape

    local_X = np.loadtxt('test/input/simulatorRun/site1_x.txt')
#   local_Y = np.loadtxt('test/input/simulatorRun/site1_y.txt')
    (site1Rows, site1Columns) = local_X.shape

    # create combinded list by local and remote data.
    # In combined_X remote data will be placed on the
    # top of local site data
    combined_X = np.concatenate((shared_X, local_X), axis=0)
    combined_X = normalize_columns(combined_X)

    # create low dimensional position
    combined_Y = np.random.randn(combined_X.shape[0], no_dims)
    combined_Y[:shared_Y.shape[0], :] = shared_Y

    # local data computation in tsne. Basically here local indicates Combined data(remote data is placed on the top of local site data). Computation specifications are described in 'tsneFunctions'
    Y_plot = tsne(
        combined_X,
        combined_Y,
        sharedRows,
        no_dims=no_dims,
        initial_dims=initial_dims,
        perplexity=perplexity,
        computation_phase='local')

    local_embedding = Y_plot[shared_Y.shape[0]:, :]

    computation_output = {
        "output": {
            "computation_phase": 'local_site',
            "local_embedding": local_embedding.tolist()
        },
        "cache": {}
    }

    return json.dumps(computation_output)


def get_all_keys(current_dict):
    children = []
    for k in current_dict:
        yield k
        if isinstance(current_dict[k], dict):
            children.append(get_all_keys(current_dict[k]))
    for k in chain.from_iterable(children):
        yield k


if __name__ == '__main__':

    np.random.seed(0)

    parsed_args = json.loads(sys.stdin.read())
    if "computation_phase" not in list(get_all_keys(parsed_args)):
        computation_output = local_noop(parsed_args)
        sys.stdout.write(computation_output)
    elif "computation_phase" in list(get_all_keys(parsed_args)):
        computation_output = local_1(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Local")
