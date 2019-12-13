from fastai.vision import download_images

loc = "data/web_scrap_db/"
#download ebmankment images
download_images(loc + "flood_embankment_image_db.csv", "data/raw/"
                "embankment")
#download flood gate images
download_images(loc + "flood_gate_images_db.csv", "data/raw/flood_gate")
#download flood wall images
download_images(loc + "flood_wall_image_db.csv", "data/raw/flood_wall")
#download outfall images
download_images(loc + "outfall_image_db.csv", "data/raw/outfall")
#download resevoir images
download_images(loc + "reservoir_image_db.csv", "data/raw/resevoir")
#download weir images
download_images(loc + "weirs_image_db.csv", "data/raw/weir")
