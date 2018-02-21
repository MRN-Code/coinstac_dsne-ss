"""
Created on Mon Jan 152017

@author: Deb
@notes: Modified by Harsh on 02/20/2018
"""
import json
import numpy as np
import sys
from tsneFunctions import normalize_columns, tsne


def remote_1(args):
    ''' It will receive parameters from dsne_single_shot. After receiving parameters it will compute tsne on high dimensional remote data and pass low dimensional values of remote site data


    args (dictionary): {
        "shared_X" (str):  remote site data,
        "shared_Label" (str):  remote site labels
        "no_dims" (int): Final plotting dimensions,
        "initial_dims" (int): number of dimensions that PCA should produce
        "perplexity" (int): initial guess for nearest neighbor
        }
    computation_phase (string): remote

        normalize_columns:
        Shared data is normalized through this function

    Returns:
        Return args will contain previous args value in addition of Y[low dimensional Y values] values of shared_Y.
    args(dictionary):  {
        "shared_X" (str):  remote site data,
        "shared_Label" (str):  remote site labels
        "no_dims" (int): Final plotting dimensions,
        "initial_dims" (int): number of dimensions that PCA should produce
        "perplexity" (int): initial guess for nearest neighbor
        "shared_Y" : the low-dimensional remote site data
        }
    '''

    shared_X = np.loadtxt('test/input/simulatorRun/shared_x.txt')
    no_dims = args["input"]["local0"]["no_dims"]
    initial_dims = args["input"]["local0"]["initial_dims"]
    perplexity = args["input"]["local0"]["perplexity"]

    shared_X = normalize_columns(shared_X)
    (sharedRows, sharedColumns) = shared_X.shape

    init_Y = np.random.randn(sharedRows, no_dims)

    # shared data computation in tsne
    shared_Y = tsne(
        shared_X,
        init_Y,
        sharedRows,
        no_dims,
        initial_dims,
        perplexity,
        computation_phase='remote')

    computation_output = {
        "output": {
            "shared_y": shared_Y.tolist(),
            "shared_x": shared_X.tolist(),
            "computation_phase": 'remote_1'
        },
        "cache": {}
    }

# To be discussed with Ross
# =============================================================================
#     np.savetxt(
#         os.path.join(args["state"]["cacheDirectory"], 'shared_y.txt'),
#         shared_Y)
#
#     computation_output = {
#         "output": {
#             "shared_y": shared_Y.tolist(),
#             "shared_x": shared_X.tolist(),
#             "computation_phase": 'remote_1'
#         },
#         "cache": shared_Y.tolist()
#     }
# =============================================================================

    return json.dumps(computation_output)


def remote_2(args):

    final_embedding = np.vstack(
        [args["input"][site]["local_embedding"] for site in args["input"]])

    computation_output = {
        "output": {
            "final_embedding": final_embedding.tolist()
        },
        "success": True
    }

    return json.dumps(computation_output)


if __name__ == '__main__':

    np.random.seed(0)

    parsed_args = json.loads(sys.argv[1])

    if parsed_args["input"]["local0"]["computation_phase"] == 'local_noop':
        computation_output = remote_1(parsed_args)
        sys.stdout.write(computation_output)
    elif parsed_args["input"]["local0"]["computation_phase"] == 'local_site':
        computation_output = remote_2(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Remote")
