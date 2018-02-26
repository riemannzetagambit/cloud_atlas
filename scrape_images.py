#!/usr/bin/env python
"""
A script to acquire images and store them according to a set of labels provided by the user
"""

import argparse
import os
import subprocess

from constants import CLOUD_TYPES
from google-images-download import IMPORTANT_FUNCTIONS


def arguments():
    parser = argparse.ArgumentParser(description='Process arguments to pass to google image scraper',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-cloud_types',
                        type=str,
                        help='Comma delimited list of cloud types to download',
                        default=','.join(CLOUD_TYPES))
    parser.add_argument('-l', '--limit', 
                         help='Number of images to download per cloud type. Current maximum is 100.', 
                         type=str, 
                         required=False)
    parser.add_argument('-c', '--color', help='Color to filter on', type=str, required=False,
                        choices=['red', 'orange', 'yellow', 'green', 'teal', 'blue', 'purple', 'pink', 'white', 'gray', 'black', 'brown'])
    args = parser.parse_args()
    return args, parser


# I'll modify what I've found in google-images-download to be more portable


def main(args):
    for ct in args.cloud_types:
        google-images-download-function
    return

if __name__ == '__main__':
    args, parser = arguments()
    main(args)
    return
