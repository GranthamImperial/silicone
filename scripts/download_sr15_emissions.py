import argparse
import os

import pyam


def download_or_load_sr15(filename):
    if not os.path.isfile(filename):
        get_sr15_scenarios(filename)
    return pyam.IamDataFrame(filename)


def get_sr15_scenarios(output_file):
    conn = pyam.iiasa.Connection("iamc15")
    variables_to_fetch = ["Emissions*"]
    for model in conn.models():
        print("Fetching data for {}".format(model))
        for variable in variables_to_fetch:
            print("Fetching {}".format(variable))
            try:
                var_df = conn.query(model=model, variable=variable)
            except Exception as e:
                print("Failed for {}".format(model))
                print(str(e))
                continue

            try:
                df.append(var_df, inplace=True)
            except NameError:
                df = pyam.IamDataFrame(var_df)

    print("Writing to {}".format(output_file))
    df.to_csv(output_file)


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
