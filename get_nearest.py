import cv2
import numpy as np
import skimage
import scipy.stats as stats
from operator import itemgetter
from tqdm import tqdm
import pandas as pd
import ast, sys, os, time, traceback
from pymongo import MongoClient, UpdateOne
import configparser, pprint
from functools import partial
import multiprocessing as mp
from jinja2 import Environment, select_autoescape, FileSystemLoader


def get_collection_obj(collection_name):
    """create new mongo client
    
    Returns:
        client -- mongo client object
    """
    client = MongoClient('mongodb://localhost:27017/')
    db = client.local
    collection = db[collection_name]
    return collection

def get_dist(desc2, desc1):
    """returns distance between 2 vectors (sq of euclidean)
    
    Arguments:
        desc2 {np.ndarray} -- first vector
        desc1 {np.ndarray} -- second vector
    
    Returns:
        np.ndarray -- sum of square of difference of each vector value
    """
    return (np.subtract(desc1,desc2)**2).sum()


def convert_to_yuv(image):
    """returns yuv channels for jpeg image
    
    Arguments:
        image {numpy arrayu} -- array of pixels for image
    
    Returns:
        channels -- yuv channels
    """
    img = cv2.imread(image)
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    Y, U, V = cv2.split(yuv)
    return Y, U, V


def get_feature_descriptor(image):
    """generate all feature descriptors for an image
    
    Arguments:
        image {numpy array} -- image pixel array
    
    Returns:
        tuple -- tuple of image name and channel moments for Y, U and V
    """
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

def chunk_records(record_ids, chunk_size):
    for i in range(0, len(record_ids), chunk_size):
        yield record_ids[i:i + chunk_size]

def get_chunk_matches(target_descriptors, ids):
    matches = []
    collection = get_collection_obj('sift_descriptors')
    rows = list(collection.find({"_id": {"$in":ids}}, projection={"descriptors":1}))
    for row in rows:
    # for row in collection.find({"_id": {"$in":ids}}, projection={"descriptors":1}):
        count = get_closest_matches(target_descriptors, np.array(row['descriptors']))
        matches.append((row['_id'], count))
    return matches

def get_sift_descriptors(image_name):
    sift = cv2.xfeatures2d.SIFT_create()
    gray = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    kp, desc = sift.detectAndCompute(gray, None)
    kp = [{ 'angle': point.angle,
            'class_id': point.class_id,
            'octave': point.octave,
            'pt': point.pt,
            'response': point.response,
            'size': point.size} for point in kp] 
    return (image_name, kp, desc)


def get_closest_matches(target_descriptors, descriptor_list):
    count = 0
    st = time.time()
    for desc1 in target_descriptors:
        min_distances = np.sort(np.apply_along_axis(get_dist, 1, descriptor_list,desc1=desc1).flatten())[:2]
        if 10 * 10 * min_distances[0] < 6 * 6 * min_distances[1]:
            count += 1
    print(time.time() - st)
    return count

def get_color_index(fd1, fd2):
    """applies color indexing formula across 2 image moments. 
       https://en.wikipedia.org/wiki/Color_moments#Color_indexing
    
    Arguments:
        fd1 {vector} -- feature desc 1
        fd2 {vector} -- feature desc 2
    
    Returns:
        float -- total distance
    """
    channelSums = np.zeros(shape=(1,0))
    for channel in ['Y', 'U', 'V']:
        channelSums = np.append(channelSums, np.absolute(np.subtract(fd1[channel], fd2[channel])))
    return np.sum(channelSums)


def get_k_similar_color_moment(image_name, k, images_directory):
    """get top k similar imagess
    
    Arguments:
        image_name {str} -- image name
        collection {collection} -- mongo collection object
        k {int} -- number of images to return
    
    Returns:
        list -- top k image list 
    """
    collection = get_collection_obj('color_moment_descriptors')
    # get color moments for image to compare
    target = collection.find_one({'_id': image_name})
    target_moments = target['moments']

    #list of tuples of similarity measures [(img2, 220), (img3, 400), (img4,120)]
    similarity_measures = []

    #get all image names in directory to filter at mongo to fetch data for those names only
    query = {"_id": {
        "$in": os.listdir(images_directory)
    }}
    rows = list(collection.find(query, projection={'moments': 1}))    
    similarity_measures = [(row['_id'], get_color_index(row['moments'], target_moments)) for row in rows]

    #sort distances and return till k+1 bc first match will be the target image with distance 0
    similarity_measures.sort(key = itemgetter(1))
    return similarity_measures[:k+1]


def get_k_similar_sift(image_name, k, images_directory, pool, chunk_size):
    """function that returns the k closest images to image
    
    Arguments:
        image {str} -- name of the image file *include extension*
        k {int} -- number of top similar images to return
        pool {pool} -- pool object to do multiproc 
        chunk_size {int} -- chunks of data to split and process for sift
    
    Returns:
        [list] -- [list of tuples of image name and distance]
    """
    collection = get_collection_obj('sift_descriptors')
    target = collection.find_one({'_id': image_name}, projection={"descriptors":1})
    target_descriptors = np.array(target['descriptors'])    
    matches = []
    ids = os.listdir(images_directory)

    partial_func = partial(get_chunk_matches,target_descriptors)
    for op in tqdm(pool.imap_unordered(partial_func,list(chunk_records(ids,chunk_size))), total=int(len(ids)/chunk_size), mininterval=1):
        matches = matches + op
    matches.sort(key = itemgetter(1), reverse=True)
    return matches[:k+1]

def plot_image(similar_images, images_directory):
    """saves image into a file
    
    Arguments:
        image {str} -- name of the image
        similar_images {list} -- top similar images
    """

    templateLoader = FileSystemLoader(searchpath="./")
    env = Environment(
        loader=templateLoader,
        autoescape=select_autoescape(['html', 'xml'])
    )
    template = env.get_template('template.html')
    optext = template.render({'images':similar_images, 'images_directory': images_directory})
    text_file = open("k_similar_images.html", "w")
    text_file.write(optext)
    text_file.close()
    print("Ouput saved in k_similar_images.html")


def generate_and_insert_moments(type, images_directory):
    """function to compute and populate mongodb with sift descriptors or color moments
    Arguments:
        collection {collection obj} -- mongo collection object 
        type {str} -- model name (sift, color_moments)
    """
    pool = mp.Pool(mp.cpu_count())
    collection = get_collection_obj('color_moment_descriptors')

    args = [images_directory + img for img in os.listdir(images_directory)]
    upserts = []
    if type == 'color_moment':
        for op in tqdm(pool.imap_unordered(get_feature_descriptor, args), total=len(args),mininterval=1):
            image_name = op[0].replace(images_directory, '')
            upserts.append(
                UpdateOne(
                    filter={'_id': image_name},
                    update={'$set': {'_id': image_name, 'moments': op[1]}},
                    upsert=True
                )
            )
    
    if type == 'sift':
        for op in tqdm(pool.imap_unordered(get_sift_descriptors, args), total=len(args),mininterval=1): 
            image_name = op[0].replace(images_directory, '')
            upserts.append(
                UpdateOne(
                    filter={'_id': image_name},
                    update={'$set': {'_id': image_name, 'key_points': op[1], 'descriptors': op[2].tolist()}},
                    upsert=True
                )
            )

    if upserts: 
        # collection.delete_many({})
        collection.bulk_write(upserts)

if __name__ == '__main__':
    try:
        config = configparser.ConfigParser()
        config.read('config.ini')
        mp.set_start_method('spawn')
        pool = mp.Pool(mp.cpu_count())

        model = config['MAIN']['model']
        image_name = config['MAIN']['file_name']
        images_directory = config['MAIN']['images_directory']
        k = config['MAIN'].getint('k')

        #generate color moments or sift descriptors and keypoints and insert fresh into mongo
        if config['MAIN'].getboolean('build_vectors'):
            print("Building models for " + model + "..............")
            generate_and_insert_moments(model, images_directory)
        
        similar_images = []
        print("Models built. getting {0} closest matches for {1}".format(k, image_name))
        if model == 'sift':
            chunk_size = config['MAIN'].getint('chunk_size')
            similar_images = get_k_similar_sift(image_name, k, images_directory, pool, chunk_size)
        elif model == 'color_moment':
            similar_images = get_k_similar_color_moment(image_name, k, images_directory)
        plot_image(similar_images, images_directory)
    
    except Exception as e:
        traceback.print_exc()
    finally:
        print("closing pool")
        pool.close()

