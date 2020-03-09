import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="download-sr15-scenarios", description="Download SR15 scenarios"
    )

    # Command line args
    parser.add_argument("output", help="File in which to save the scenarios")
    args = parser.parse_args()

    get_sr15_scenarios(args.output)


if __name__ == "__main__":
    main()
