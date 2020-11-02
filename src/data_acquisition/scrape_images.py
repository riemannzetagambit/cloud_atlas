#!/usr/bin/env python
"""
A script to acquire images and store them according to a set of labels provided by the user
"""

import argparse
import logging
import os
import subprocess
from urllib.parse import quote
import time

from google_images_download import google_images_download

if __name__ == '__main__' and __package__ is None:
    # run this if executing as a script
    from os import sys, path
    # for constants
    sys.path.append(path.abspath(path.join(path.dirname(__file__), '..')))
from constants import CLOUD_TYPES


def arguments():
    parser = argparse.ArgumentParser(description='Process arguments to pass to google image scraper',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-cloud_types',
                        type=str,
                        help='Comma delimited list of cloud types to download',
                        default=','.join(CLOUD_TYPES))
    parser.add_argument('-l', '--limit',
                         help='Number of images to download per cloud type. Current maximum is 100.',
                         type=int,
                         default=20,
                         required=False)
    parser.add_argument('-o', '--output_directory',
                        help='download images in a specific directory',
                        type=str,
                        default='downloads',
                        required=False)
    parser.add_argument('-c', '--color',
                        help='Color to filter on',
                        type=str,
                        required=False,
                        default='blue',
                        choices=['red', 'orange', 'yellow', 'green', 'teal', 'blue', 'purple', 'pink', 'white', 'gray', 'black', 'brown'])
    parser.add_argument('-f', '--image_format',
                        help='download images with specific format',
                        type=str,
                        default='png',
                        required=False)
    parser.add_argument('-s', '--size',
                        help='Image size to use. Should be "icon", "medium", "large", or a custom size '
                             'of the form "NxM", N and M integers',
                        type=str,
                        default='medium',
                        required=False)
    args = parser.parse_args()
    return args, parser


def main(args):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    cloud_types = args.cloud_types.split(',')
    for ct in cloud_types:
        logger.info('Getting images for {}'.format(ct))
        output_directory = '{}/{}'.format(args.output_directory, ct)
        print(output_directory)
        # for fixing the search keyword
        ct = ct + ' cloud'
        response = google_images_download.googleimagesdownload()   #class instantiation

        arguments = {"keywords":"Polar bears,baloons,Beaches","limit":20,"print_urls":True}   #creating list of
        search_args = {'keywords': ct,
                       'limit': args.limit,
                       'size': args.size,
                       'format': args.image_format,
                       'color': args.color,
                       'output_directory': output_directory}
        paths = response.download(search_args)

if __name__ == '__main__':
    args, parser = arguments()
    main(args)
