from os.path import dirname, abspath, join
import sys

# Find code directory relative to our directory
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, '..'))
sys.path.append(CODE_DIR)

from opinions.interval import CoupledNetworkCastorAndPollux, IndependentNetworkCastorAndPollux


def main():
    o = CoupledNetworkCastorAndPollux(6, 3, save_results=True)
    o.run_simulation(1)


if __name__ == "__main__":
    main()