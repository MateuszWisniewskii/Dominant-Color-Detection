import argparse
from generate_csv_data import create_data_CSV
from pathlib import Path



if __name__ == "__main__":
    #command line arguments
    parser = argparse.ArgumentParser( description="Script for creating CSV file from given raw data")
    parser.add_argument("--output_csv", type=str, required=True, help="Name for output csv file.")
    parser.add_argument("--txt_file", type=str, required=True, help="Path for raw data txt file.")
    parser.add_argument("--images_folder", type=str, required=True, help="Path to directory with images.")

    args = parser.parse_args()

    create_data_CSV(args.output_csv, args.txt_file, args.images_folder)

    csv_path = Path(args.output_csv)

    if csv_path.exists():
        print(f"{args.output_csv} was created.")
    else:
        print(f"An error occured while creating {args.output_csv}")

# generate_csv_file.py --output_csv my_csv.csv --txt_file ./Data/GKew_2025_03_20-11_03_35.txt --images_folder ./Data/PhotosColorPicker