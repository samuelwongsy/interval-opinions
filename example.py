from opinions.interval import CoupledNetworkCastorAndPollux, IndependentNetworkCastorAndPollux


def main():
    o = CoupledNetworkCastorAndPollux(6, 3, save_results=True)
    o.run_simulation(1)


if __name__ == "__main__":
    main()