# main.py
import argparse
from data_loader import load_data
from classifier import classify_process
from function import levy_fit, intermittent_fit

def main():
    parser = argparse.ArgumentParser(description="Classify data as Levy or Intermittent.")
    parser.add_argument('filepath', type=str, help='Path to the data file')
    args = parser.parse_args()

    data = load_data(args.filepath)
    levy_params = levy_fit(data)
    intermittent_params = intermittent_fit(data)
    result = classify_process(levy_params, intermittent_params)
    print(f"Classification Result: {result}")

if __name__ == "__main__":
    main()

