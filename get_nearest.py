import cv2
import numpy as np
import skimage
import scipy.stats as stats
from operator import itemgetter
from tqdm import tqdm
import pandas as pd
import ast, sys, os, time
from pymongo import MongoClient, UpdateOne
import configparser, pprint
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import multiprocessing as mp 


#Converts JPEG to YUV and return y,u,v channels
def convert_to_yuv(image):
    img = cv2.imread(image)
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    Y, U, V = cv2.split(yuv)
    return Y, U, V

#generate feature descripture for all 3 channels of the image
def get_feature_descriptor(image):
    Y, U, V = convert_to_yuv(image)
    channels = {'Y': Y, 'U': U, 'V': V}
    channel_moments = {}
    
    #iterate for each of the channel components
    for channel in channels.keys():
        blocks = skimage.util.view_as_blocks(channels[channel], (100,100))
        
        #calculates the mean of all the elements in each block and flattens it to form a 1D array
        means = np.mean(blocks, axis = (2,3))
        means = means.flatten()

        #same as mean but SD
        sd = np.std(blocks, axis = (2,3))
        sd = sd.flatten()

        #calculate skewness across each 12*16 (100*100 blocks) //original image is 1200*1600//
        skew = np.array([stats.skew(blocks[i, j].flatten()) for i in range(12) for j in range(16)])
        skew = skew.flatten()
        
        #forms a 1D array of [m1, sd1, sk1, m2, sd2, sk2.......mn, sdn, skn] where n is the number of 100*100 windows
        channel_moments[channel] = list(np.stack((means,sd,skew), axis = -1).flatten())
    
    return (image, channel_moments, )

"""applies color index formula across 2 image moments. 
https://en.wikipedia.org/wiki/Color_moments#Color_indexing"""
def get_color_index(fd1, fd2):
    channelSums = np.zeros(shape=(1,0))
    for channel in ['Y', 'U', 'V']:
        channelSums = np.append(channelSums, np.absolute(np.subtract(fd1[channel], fd2[channel])))
    return (image, np.sum(channelSums))

#get top k images that match with image_name
def get_k_similar(image_name, collection, k):

    # get color moments for image to compare
    target = collection.find_one({'_id': image_name})
    target_moments = target['moments']

    #list of tuples of similarity measures [(img2, 220), (img3, 400), (img4,120)]
    similarity_measures = []
    rows = list(collection.find({}, projection={'moments': 1}))    
    similarity_measures = [(row['_id'], get_color_index(row['moments'], target_moments)) for row in rows]

    #sort distances and return till k+1 bc first match will be the target image with distance 0
    similarity_measures.sort(key = itemgetter(1))
    return similarity_measures[:k+1]

#function to save target image and k matches in order vertically into a file.
def plot_image(image, similar_images):
    img = cv2.imread('Hands/' + image)
    other_imgs = [cv2.imread('Hands/' + image[0]) for image in similar_images]
    image_tuple = (img, ) + tuple(other_imgs)
    vis = np.concatenate(image_tuple, axis=0)
    cv2.imwrite('out.jpg', vis)

#multiproc function to generate moments and insert into mongo.
#called only if COLOR_MOMENTS.build_vectors is True
def generate_and_insert_moments(collection):
    pool = mp.Pool(mp.cpu_count())
    args = ['Hands/' + img for img in os.listdir('Hands')]
    upserts = []
    for channel_moments in tqdm(pool.imap_unordered(get_feature_descriptor, args), total=len(args),mininterval=1):
        image_name = channel_moments[0].replace('Hands/', '')
        upserts.append(
            UpdateOne(
                filter={'_id': image_name},
                update={'$set': {'_id': image_name, 'moments': channel_moments[1]}},
                upsert=True
            )
        )
    collection.bulk_write(upserts)

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    mp.set_start_method('spawn')

    #initialize local mongo client
    client = MongoClient('mongodb://localhost:27017/')
    db = client.local
    collection = db.image_feature_descriptor

    #generate color moments and insert fresh into mongo
    if config['COLOR_MOMENTS'].getboolean('build_vectors'):
        generate_and_insert_moments(collection)
    
    image = config['COLOR_MOMENTS']['file_name']
    k = config['COLOR_MOMENTS'].getint('k')
    
    similar_images = get_k_similar(image,collection, k)
    pprint.pprint(similar_images, indent=4)
    plot_image(image, similar_images)
    client.close()