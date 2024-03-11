import argparse
import os
import json
import jsonlines
from tqdm import tqdm


def write_jsonl(save_path, item):
    with jsonlines.open(save_path, mode = 'a') as json_writer:
        json_writer.write(item)


def main():
    ### argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--files", type=str, nargs="+", default=None, 
                        help="The jsonl file list.")
    parser.add_argument("-s", "--save_path", type=str, default="merged_file.jsonl", 
                        help="The path to sav merged files.")
    
    # get args
    args = parser.parse_args()
    file_list = args.files
    if file_list is None:
        raise Exception("The files to merge is empty!")

    save_path = args.save_path
    if os.path.exists(save_path):
        os.remove(save_path)
        print("Remove the already existed file: {}".format(save_path))

    # merge files
    for afile in file_list:
        with open(afile, 'r') as f:
            lines = f.readlines()
        
        pbar = tqdm(lines)
        for aline in pbar:
            pbar.set_description("[{}]".format(os.path.basename(afile)))
            aline = json.loads(aline)
            write_jsonl(save_path, aline)


if __name__ == "__main__":
    main()

