from os.path import dirname, abspath, join
import sys

# Find code directory relative to our directory
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, '..'))
sys.path.append(CODE_DIR)

from opinions.interval import CoupledNetworkCastorAndPollux, GraphType


def main():
    o = CoupledNetworkCastorAndPollux(6, 2, save_results=True, castor_graph_type='cycle', pollux_graph_type='star')
    o.run_simulation()


if __name__ == "__main__":
    main()