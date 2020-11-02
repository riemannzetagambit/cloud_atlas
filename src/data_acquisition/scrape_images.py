#!/usr/bin/env python
"""
A script to acquire images and store them according to a set of labels provided by the user
"""

import argparse
import logging
import os
import subprocess

if __name__ == '__main__' and __package__ is None:
    # run this if executing as a script
    from os import sys, path
    # for constants
    sys.path.append(path.abspath(path.join(path.dirname(__file__), '..')))
    # for google_image_download
    sys.path.append(path.dirname(path.abspath(__file__)))
from constants import CLOUD_TYPES
from google_images_download import bulk_download


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


# I'll modify what I've found in google-images-download to be more portable
def bulk_download(search_keyword,
                  suffix_keywords,
                  limit,
                  output_directory,
                  delay_time,
                  color=None,
                  url=None,
                  similar_images=None,
                  specific_site=None,
                  image_format=None,
                  size=None):
    errorCount = 0
    if url:
        search_keyword = [str(datetime.datetime.now()).split('.')[0]]
    if similar_images:
        search_keyword = [str(datetime.datetime.now()).split('.')[0]]

    # appending a dummy value to Suffix Keywords array if it is blank
    if len(suffix_keywords) == 0:
        suffix_keywords.append('')

    for sky in suffix_keywords:
        i = 0
        while i < len(search_keyword):
            items = []
            iteration = "\n" + "Item no.: " + str(i + 1) + " -->" + " Item name = " + str(search_keyword[i] + str(sky))
            print(iteration)
            print("Evaluating...")
            search_term = search_keyword[i] + sky
            dir_name = search_term + ('-' + color if color else '')

            # make a search keyword  directory
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

            params = build_url_parameters(color=color, image_format=image_format, size=size)
            # color_param = ('&tbs=ic:specific,isc:' + args.color) if args.color else ''
            # check the args and choose the URL
            if url is not None:
                pass
            elif similar_images is not None:
                keywordem = similar_images()
                url = 'https://www.google.com/search?q=' + keywordem + '&espv=2&biw=1366&bih=667&site=webhp&source=lnms&tbm=isch&sa=X&ei=XosDVaCXD8TasATItgE&ved=0CAcQ_AUoAg'
            elif specific_site is not None:
                url = 'https://www.google.com/search?q=' + quote(
                    search_term) + 'site:' + specific_site + '&espv=2&biw=1366&bih=667&site=webhp&source=lnms&tbm=isch' + params + '&sa=X&ei=XosDVaCXD8TasATItgE&ved=0CAcQ_AUoAg'
            else:
                url = 'https://www.google.com/search?q=' + quote(
                    search_term) + '&espv=2&biw=1366&bih=667&site=webhp&source=lnms&tbm=isch' + params + '&sa=X&ei=XosDVaCXD8TasATItgE&ved=0CAcQ_AUoAg'
            raw_html = (download_page(url))
            time.sleep(0.1)
            items = items + (_images_get_all_items(raw_html))
            print("Total Image Links = " + str(len(items)))

            #If search does not return anything, do not try to force download
            if len(items) <= 1:
                print('***** This search result did not return any results...please try a different search filter *****')
                break

            print("Starting Download...")

            k = 0
            success_count = 0
            while (k < len(items)):
                try:
                    image_url = items[k]
                    #print("\n" + str(image_url))
                    req = Request(image_url, headers={
                        "User-Agent": "Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17"})
                    try:
                        response = urlopen(req, None, 15)
                        image_name = str(items[k][(items[k].rfind('/')) + 1:])
                        if '?' in image_name:
                            image_name = image_name[:image_name.find('?')]
                        if ".jpg" in image_name or ".JPG" in image_name or ".gif" in image_name or ".png" in image_name or ".bmp" in image_name or ".svg" in image_name or ".webp" in image_name or ".ico" in image_name:
                            output_file_name = '{od}/{image_name}_{num}.{fmt}'.format(od=output_directory,
                                                                                      image_name=output_directory.split('/')[-1],
                                                                                      num=success_count,
                                                                                      fmt=image_format)
                            output_file = open(output_file_name, 'wb')
                        else:
                            if image_format is not None:
                                output_file_name = '{od}/{image_name}_{num}.{fmt}'.format(od=output_directory,
                                                                                          image_name=output_directory.split('/')[-1],
                                                                                          num=success_count,
                                                                                          fmt=image_format)
                                output_file = open(output_file_name, 'wb')
                                image_name = image_name + "." + image_format
                            else:
                                output_file_name = '{od}/{image_name}_{num}.{fmt}'.format(od=output_directory,
                                                                                          image_name=output_directory.split('/')[-1],
                                                                                          num=success_count,
                                                                                          fmt=image_format)
                                output_file = open(output_file_name, 'wb')

                        data = response.read()
                        output_file.write(data)
                        response.close()

                        print("Completed ====> " + str(success_count + 1) + ". " + image_name)
                        k = k + 1
                        success_count += 1
                        if success_count == limit:
                            break
                    except UnicodeEncodeError as e:
                        errorCount +=1
                        print ("UnicodeEncodeError on an image...trying next one..." + " Error: " + str(e))
                        k = k + 1

                except HTTPError as e:  # If there is any HTTPError
                    errorCount += 1
                    print("HTTPError on an image...trying next one..." + " Error: " + str(e))
                    k = k + 1

                except URLError as e:
                    errorCount += 1
                    print("URLError on an image...trying next one..." + " Error: " + str(e))
                    k = k + 1

                except ssl.CertificateError as e:
                    errorCount += 1
                    print("CertificateError on an image...trying next one..." + " Error: " + str(e))
                    k = k + 1

                except IOError as e:  # If there is any IOError
                    errorCount += 1
                    print("IOError on an image...trying next one..." + " Error: " + str(e))
                    k = k + 1

                if delay_time is not None:
                    time.sleep(int(delay_time))

            if success_count < limit:
                print("\n\nUnfortunately all " + str(limit) + " could not be downloaded because some images were not downloadable. " + str(success_count) + " is all we got for this search filter!")
            i = i + 1
    return errorCount


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
        errorCount = bulk_download(search_keyword=[ct],
                                   suffix_keywords=[],
                                   limit=args.limit,
                                   output_directory=output_directory,
                                   delay_time=None,
                                   color=args.color,
                                   url=None,
                                   similar_images=None,
                                   specific_site=None,
                                   image_format=args.image_format,
                                   size=args.size)

if __name__ == '__main__':
    args, parser = arguments()
    main(args)
