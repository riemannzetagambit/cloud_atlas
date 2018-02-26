#!/usr/bin/env python
"""
A script to acquire images and store them according to a set of labels provided by the user
"""

import argparse
import logging
import os
import subprocess

from constants import CLOUD_TYPES
from google-images-download import bulk_download


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
    parser.add_argument('-o', '--output_directory', 
                        help='download images in a specific directory', 
                        type=str, 
                        default='downloads',
                        required=False)
    parser.add_argument('-c', '--color', help='Color to filter on', type=str, required=False,
                        choices=['red', 'orange', 'yellow', 'green', 'teal', 'blue', 'purple', 'pink', 'white', 'gray', 'black', 'brown'])
    args = parser.parse_args()
    return args, parser


# I'll modify what I've found in google-images-download to be more portable


def main(args):
    logger = logging.get_logger(__name__)
    for ct in args.cloud_types:
        logger.info('Getting images for {}'.format(ct))
        output_directory = '{}/{}'.format(output_directory, ct)
        errorCount = bulk_download(search_keyword=ct,
                                   suffix_keywords=[],
                                   limit=args.limit,
                                   output_directory,
                                   delay_time=None,
                                   color=args.color,
                                   url=None,
                                   similar_images=None,
                                   specific_site=None,
                                   format=args.format)

if __name__ == '__main__':
    args, parser = arguments()
    main(args)
