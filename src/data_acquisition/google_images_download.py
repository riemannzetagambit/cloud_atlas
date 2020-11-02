# In[ ]:
#  coding: utf-8

# there is so much refactoring to be done to this script but I will only do the bare minimum to download as I want
###### Searching and Downloading Google Images to the local disk ######

# Import Libraries
import sys
version = (3, 0)
cur_version = sys.version_info
if cur_version >= version:  # If the Current Version of Python is 3.0 or above
    import urllib.request
    from urllib.request import Request, urlopen
    from urllib.request import URLError, HTTPError
    from urllib.parse import quote
else:  # If the Current Version of Python is 2.x
    import urllib2
    from urllib2 import Request, urlopen
    from urllib2 import URLError, HTTPError
    from urllib import quote
import time  # Importing the time library to check the time of code execution
import os
import re
import argparse
import ssl
import datetime

def arguments():
    # Taking command line arguments from users
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--keywords', help='delimited list input', type=str, required=False)
    parser.add_argument('-sk', '--suffix_keywords', help='comma separated additional words added to main keyword', type=str, required=False)
    parser.add_argument('-l', '--limit', help='delimited list input', type=str, required=False)
    parser.add_argument('-f', '--image_format', help='download images with specific format', type=str, required=False,
                        choices=['jpg', 'gif', 'png', 'bmp', 'svg', 'webp', 'ico'])
    parser.add_argument('-u', '--url', help='search with google image URL', type=str, required=False)
    parser.add_argument('-x', '--single_image', help='downloading a single image from URL', type=str, required=False)
    parser.add_argument('-o', '--output_directory', help='download images in a specific directory', type=str, required=False)
    parser.add_argument('-d', '--delay', help='delay in seconds to wait between downloading two images', type=str, required=False)
    parser.add_argument('-c', '--color', help='filter on color', type=str, required=False,
                        choices=['red', 'orange', 'yellow', 'green', 'teal', 'blue', 'purple', 'pink', 'white', 'gray', 'black', 'brown'])
    parser.add_argument('-ct', '--color_type', help='filter on color', type=str, required=False,
                        choices=['full-color', 'black-and-white', 'transparent'])
    parser.add_argument('-r', '--usage_rights', help='usage rights', type=str, required=False,
                        choices=['labled-for-reuse-with-modifications','labled-for-reuse','labled-for-noncommercial-reuse-with-modification','labled-for-nocommercial-reuse'])
    parser.add_argument('-s', '--size', help='image size', type=str, required=False,
                        choices=['large','medium','icon'])
    parser.add_argument('-t', '--type', help='image type', type=str, required=False,
                        choices=['face','photo','clip-art','line-drawing','animated'])
    parser.add_argument('-w', '--time', help='image age', type=str, required=False,
                        choices=['past-24-hours','past-7-days'])
    parser.add_argument('-a', '--aspect_ratio', help='comma separated additional words added to keywords', type=str, required=False,
                        choices=['tall', 'square', 'wide', 'panoramic'])
    parser.add_argument('-si', '--similar_images', help='downloads images very similar to the image URL you provide', type=str, required=False)
    parser.add_argument('-ss', '--specific_site', help='downloads images that are indexed from a specific website', type=str, required=False)

    args = parser.parse_args()

#------ Initialization Complete ------#

# Downloading entire Web Document (Raw Page Content)
def download_page(url):
    version = (3, 0)
    cur_version = sys.version_info
    if cur_version >= version:  # If the Current Version of Python is 3.0 or above
        try:
            headers = {}
            headers['User-Agent'] = "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36"
            req = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(req)
            respData = str(resp.read())
            return respData
        except Exception as e:
            print(str(e))
    else:  # If the Current Version of Python is 2.x
        try:
            headers = {}
            headers['User-Agent'] = "Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17"
            req = urllib2.Request(url, headers=headers)
            try:
                response = urllib2.urlopen(req)
            except URLError:  # Handling SSL certificate failed
                context = ssl._create_unverified_context()
                response = urlopen(req, context=context)
            page = response.read()
            return page
        except:
            return "Page Not found"


# Finding 'Next Image' from the given raw page
def _images_get_next_item(s):
    start_line = s.find('rg_di')
    if start_line == -1:  # If no links are found then give an error!
        end_quote = 0
        link = "no_links"
        return link, end_quote
    else:
        start_line = s.find('"class="rg_meta"')
        start_content = s.find('"ou"', start_line + 1)
        end_content = s.find(',"ow"', start_content + 1)
        content_raw = str(s[start_content + 6:end_content - 1])
        return content_raw, end_content


# Getting all links with the help of '_images_get_next_image'
def _images_get_all_items(page):
    items = []
    while True:
        item, end_content = _images_get_next_item(page)
        if item == "no_links":
            break
        else:
            items.append(item)  # Append all the links in the list named 'Links'
            time.sleep(0.1)  # Timer could be used to slow down the request for image downloads
            page = page[end_content:]
    return items


def similar_images():
    version = (3, 0)
    cur_version = sys.version_info
    if cur_version >= version:  # If the Current Version of Python is 3.0 or above
        try:
            searchUrl = 'https://www.google.com/searchbyimage?site=search&sa=X&image_url=' + args.similar_images
            headers = {}
            headers['User-Agent'] = "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36"

            req1 = urllib.request.Request(searchUrl, headers=headers)
            resp1 = urllib.request.urlopen(req1)
            content = str(resp1.read())
            l1 = content.find('AMhZZ')
            l2 = content.find('&', l1)
            urll = content[l1:l2]

            newurl = "https://www.google.com/search?tbs=sbi:" + urll + "&site=search&sa=X"
            req2 = urllib.request.Request(newurl, headers=headers)
            resp2 = urllib.request.urlopen(req2)
            # print(resp2.read())
            l3 = content.find('/search?sa=X&amp;q=')
            l4 = content.find(';', l3 + 19)
            urll2 = content[l3 + 19:l4]
            return urll2
        except:
            return "Cloud not connect to Google Imagees endpoint"
    else:  # If the Current Version of Python is 2.x
        try:
            searchUrl = 'https://www.google.com/searchbyimage?site=search&sa=X&image_url=' + args.similar_images
            headers = {}
            headers['User-Agent'] = "Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17"

            req1 = urllib2.Request(searchUrl, headers=headers)
            resp1 = urllib2.urlopen(req1)
            content = str(resp1.read())
            l1 = content.find('AMhZZ')
            l2 = content.find('&', l1)
            urll = content[l1:l2]

            newurl = "https://www.google.com/search?tbs=sbi:" + urll + "&site=search&sa=X"
            #print newurl
            req2 = urllib2.Request(newurl, headers=headers)
            resp2 = urllib2.urlopen(req2)
            # print(resp2.read())
            l3 = content.find('/search?sa=X&amp;q=')
            l4 = content.find(';', l3 + 19)
            urll2 = content[l3 + 19:l4]
            return(urll2)
        except:
            return "Cloud not connect to Google Imagees endpoint"

#Building URL parameters
def build_url_parameters(color=None,
                         color_type=None,
                         usage_rights=None,
                         size=None,
                         type=None,
                         time=None,
                         aspect_ratio=None,
                         image_format=None):

    # specifically for size
    m = re.match('(?P<w>\d+)x(?P<h>\d+)', size)
    if m:
        custom_size_str = 'iszw:{},iszh:{}'.format(m.group('w'), m.group('h'))
        size = 'custom'
    built_url = "&tbs="
    counter = 0
    params = {'color':[color,{'red':'ic:specific,isc:red', 'orange':'ic:specific,isc:orange', 'yellow':'ic:specific,isc:yellow', 'green':'ic:specific,isc:green', 'teal':'ic:specific,isc:teel', 'blue':'ic:specific,isc:blue', 'purple':'ic:specific,isc:purple', 'pink':'ic:specific,isc:pink', 'white':'ic:specific,isc:white', 'gray':'ic:specific,isc:gray', 'black':'ic:specific,isc:black', 'brown':'ic:specific,isc:brown'}],
              'color_type':[color_type,{'full-color':'ic:color', 'black-and-white':'ic:gray','transparent':'ic:trans'}],
              'usage_rights':[usage_rights,{'labled-for-reuse-with-modifications':'sur:fmc','labled-for-reuse':'sur:fc','labled-for-noncommercial-reuse-with-modification':'sur:fm','labled-for-nocommercial-reuse':'sur:f'}],
              'size':[size,{'large':'isz:l','medium':'isz:m','icon':'isz:i', 'custom': custom_size_str}],
              'type':[type,{'face':'itp:face','photo':'itp:photo','clip-art':'itp:clip-art','line-drawing':'itp:lineart','animated':'itp:animated'}],
              'time':[time,{'past-24-hours':'qdr:d','past-7-days':'qdr:w'}],
              'aspect_ratio':[aspect_ratio,{'tall':'iar:t','square':'iar:s','wide':'iar:w','panoramic':'iar:xw'}],
              'format':[image_format,{'jpg':'ift:jpg','gif':'ift:gif','png':'ift:png','bmp':'ift:bmp','svg':'ift:svg','webp':'webp','ico':'ift:ico'}]}
    for key, value in params.items():
        if value[0] is not None:
            ext_param = value[1][value[0]]
            # counter will tell if it is first param added or not
            if counter == 0:
                # add it to the built url
                built_url = built_url + ext_param
                counter += 1
            else:
                built_url = built_url + ',' + ext_param
                counter += 1
    return built_url

#function to download single image
def single_image(output_directory, url):
    try:
        os.makedirs(output_directory)
    except OSError as e:
        if e.errno != 17:
            raise
            # time.sleep might help here
        pass
    req = Request(url, headers={
        "User-Agent": "Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17"})
    response = urlopen(req, None, 10)
    image_name = str(url[(url.rfind('/')) + 1:])
    if '?' in image_name:
        image_name = image_name[:image_name.find('?')]
    if ".jpg" in image_name or ".gif" in image_name or ".png" in image_name or ".bmp" in image_name or ".svg" in image_name or ".webp" in image_name or ".ico" in image_name:
        output_file = open(output_directory + "/" + image_name, 'wb')
    else:
        output_file = open(output_directory + "/" + image_name + ".jpg", 'wb')
        image_name = image_name + ".jpg"

    data = response.read()
    output_file.write(data)
    response.close()
    print("completed ====> " + image_name)
    return


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

if __name__ == '__main__':
    args, parser = arguments()
    cwd = os.path.dirname(os.path.realpath(__file__))

    #Initialization and Validation of user arguments
    if args.keywords:
        search_keyword = [str(item) for item in args.keywords.split(',')]

    #Additional words added to keywords
    if args.suffix_keywords:
        suffix_keywords = [" " + str(sk) for sk in args.suffix_keywords.split(',')]
    else:
        suffix_keywords = []

    # Setting limit on number of images to be downloaded
    if args.limit:
        limit = int(args.limit)
        if int(args.limit) >= 100:
            limit = 100
    else:
        limit = 100

    # If single_image or url argument not present then keywords is mandatory argument
    if args.single_image is None and args.url is None and args.similar_images is None and args.keywords is None:
                parser.error('Keywords is a required argument!')

    # If this argument is present, set the custom output directory
    if args.output_directory:
        output_directory = os.path.join(cwd, args.output_directory)
    else:
        output_directory = os.path.join(cwd, 'downloads')

    # Set the delay parameter if this argument is present
    if args.delay:
        try:
            delay_time = int(args.delay)
        except ValueError:
            parser.error('Delay parameter should be an integer!')
    else:
        delay_time = 0
    if args.single_image:       #Download Single Image using a URL
        single_image_url = args.single_image
        single_image(output_directory=output_directory, url = single_image_url)
    else:                       # or download multiple images based on keywords/keyphrase search
        t0 = time.time()  # start the timer
        # TODO(dstone): just make this take args. But the code I inherited is totally whack anyway, so it's a big hassle
        # to refactor bulk_download to take args
        errorCount = bulk_download(search_keyword,
                                   suffix_keywords,
                                   limit,
                                   output_directory,
                                   delay_time=delay_time,
                                   color=args.color,
                                   url=args.url,
                                   similar_images=args.similar_images,
                                   specific_site=args.specific_site,
                                   image_format=args.image_format)

        print("\nEverything downloaded!")
        print("Total Errors: " + str(errorCount) + "\n")
        t1 = time.time()  # stop the timer
        total_time = t1 - t0  # Calculating the total time required to crawl, find and download all the links of 60,000 images
        print("Total time taken: " + str(total_time) + " Seconds")
