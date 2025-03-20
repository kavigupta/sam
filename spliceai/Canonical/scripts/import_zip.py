import os
import argparse


def filename_without_extension(filename):
    filename = os.path.basename(filename)
    filename, zip_ = os.path.splitext(filename)
    assert zip_ == ".zip", "Expected a zip file"
    return filename


def export_to(path, destination):
    os.makedirs(destination, exist_ok=True)
    os.system(f"unzip {path} -d {destination}")


def dispatch(path):
    filename = filename_without_extension(path)
    if filename in ["models_trained_on_eclip"]:
        return export_to(path, "intermediates/eclip/trained_on_eclip")
    assert filename.startswith("msp-"), f"Unrecognized filename {filename}"
    return export_to(path, "model")


def main():
    parser = argparse.ArgumentParser(description="Import a zip file into this folder")
    parser.add_argument("zipfiles", help="The zip file to import", nargs="+")
    args = parser.parse_args()
    for zipfile in args.zipfiles:
        print("Importing", zipfile)
        dispatch(zipfile)


if __name__ == "__main__":
    main()
