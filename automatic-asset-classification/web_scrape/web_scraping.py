loc = "data/web_scrap_db/"

import csv
import requests
from requests.exceptions import InvalidSchema, ConnectionError, Timeout
import logging

logging.basicConfig(filename = 'automatic-asset-classification/web_scrape/web_scrape.log', level = logging.INFO)

types = ["embankment", "flood_gate", "flood_wall", "outfall", "reservoir", "weir"]

for type in types:

    #if type == "embankment" or type == "flood_wall" or type == "outfall" or type == "flood_gate" or type == "reservior":
    #    continue

    output = "data/raw/" + str(type) + "/"

    with open(loc + str(type) + "_image_db.csv") as csvfile:
        csvrows = csv.reader(csvfile, delimiter=',', quotechar='"')
        i = 0
        for row in csvrows:
            #filename = row[0]
            url = row[0]
            i += 1
            filename = str(type) + "_" + str(i) + ".jpg"
            print(url)
            try:
                result = requests.get(url, timeout = 20)
                if result.status_code == 200:
                    image = result.raw.read()
                    open(output + filename,"wb").write(image)
            except InvalidSchema:
                print(i, "of ", type, "not worked")
                logging.exception("InvalidSchema " + str(type) + "row number" + str(i) + "not worked, url: " + url)
            except ConnectionError:
                print(i, "of ", type, "connection error")
                logging.exception("ConnectionError " + str(type) + "row number" + str(i) + "not worked, url: " + url)
            except Timeout:
                print(i, "of ", type, "timeout error")
                logging.exception("Timeout " + str(type) + "row number" + str(i) + "not worked, url: " + url)
                try:
                    result = requests.get(url, timeout = 20)
                    if result.status_code == 200:
                        image = result.raw.read()
                        open(output + filename,"wb").write(image)
                except InvalidSchema:
                    print(i, "of ", type, "second attempt not worked")
                    logging.exception("InvalidSchema " + str(type) + "row number" + str(i) + "not worked, url: " + url)
                except ConnectionError:
                    print(i, "of ", type, "connection error")
                    logging.exception("ConnectionError " + str(type) + "row number" + str(i) + "second attempt not worked, url: " + url)
                except Timeout:
                    print(i, "of ", type, "timeout error")
                    logging.exception("Timeout " + str(type) + "row number" + str(i) + "timed out twice, url: " + url)
