import cv2 as cv
import numpy as np
from scipy import spatial
import os, time
from tqdm import tqdm
import pandas as pd
from pymongo import MongoClient, UpdateOne
from operator import itemgetter
import multiprocessing as mp 
import os
from functools import partial
import configparser
from itertools import product

def get_collection_obj():
    client = MongoClient('mongodb://localhost:27017/')
    db = client.local
    collection = db.sift_descriptors
    return collection

def get_dist(desc2, desc1):
    return np.sum(np.square(np.subtract(desc1,desc2)))

def get_closest_matches(target_descriptors, descriptor_list):
    count = 0
    st = time.time()
    for desc1 in target_descriptors:
        min_distances = np.array([np.apply_along_axis(get_dist, 1, descriptor_list,desc1=desc1)]).flatten()
        min_distances = np.sort(min_distances)[:2]
        if 10 * 10 * min_distances[0] < 6 * 6 * min_distances[1]:
            count += 1
            # print('yay')
    en = time.time()
    # print(en - st)
    return count

def chunk_records(record_ids, chunk_size):
    for i in range(0, len(record_ids), chunk_size):
        yield record_ids[i:i + chunk_size]


def get_chunk_matches(target_descriptors, ids):
    matches = []
    collection = get_collection_obj()
    rows = list(collection.find({"_id": {"$in":ids}}, projection={"descriptors":1}))
    for row in rows:
    # for row in collection.find({"_id": {"$in":ids}}, projection={"descriptors":1}):
        count = get_closest_matches(target_descriptors, np.array(row['descriptors']))
        matches.append((row['_id'], count))
    return matches

def get_sift_descriptors(image):
    sift = cv.xfeatures2d.SIFT_create()
    gray = cv.imread(image, cv.IMREAD_GRAYSCALE)
    kp, desc = sift.detectAndCompute(gray, None)
    kp = [{ 'angle': point.angle,
            'class_id': point.class_id,
            'octave': point.octave,
            'pt': point.pt,
            'response': point.response,
            'size': point.size} for point in kp] 
    return (image, kp, desc)



def get_k_similar(image, k, pool, chunk_size):
    collection = get_collection_obj()
    target = collection.find_one({'_id': image}, projection={"descriptors":1})
    target_descriptors = np.array(target['descriptors'])    
    matches = []
    ids = collection.find().distinct('_id')

    partial_func = partial(get_chunk_matches,target_descriptors)
    for op in tqdm(pool.imap_unordered(partial_func,list(chunk_records(ids,chunk_size))), total=int(len(ids)/chunk_size), mininterval=1):
        matches = matches + op
    # for chunk_ids in tqdm(list(chunk_records(ids, chunk_size))):
    #     matches = matches + get_chunk_matches(target_descriptors, chunk_ids)
    matches.sort(key = itemgetter(1), reverse=True)
    return matches[:k+1]

if __name__ == '__main__':
    try:
        args = ['Hands/'+img for img in os.listdir('Hands')]

        # target = 'Hand_0000002.jpg'
        # desc_1 = get_sift_descriptors('Hands/' + target)
        # desc_list_2 = []

        mp.set_start_method("spawn")

        # upserts = []
        # for op in tqdm(pool.imap_unordered(get_sift_descriptors, args), total=len(args),mininterval=1): 
        #     image_name = op[0].replace('Hands/', '')
        #     upserts.append(
        #         UpdateOne(
        #             filter={'_id': image_name},
        #             update={'$set': {'_id': image_name, 'key_points': op[1], 'descriptors': op[2].tolist()}},
        #             upsert=True
        #         )
        #     )
        # pool.close()
        # collection.bulk_write(upserts)
        pool = mp.Pool(mp.cpu_count())
        config = configparser.ConfigParser()
        config.read('config.ini')
        chunk_size = config['SIFT'].getint('chunk_size')
        print(get_k_similar('Hand_0000002.jpg', 10, pool, chunk_size))

    except Exception as e:
        print(e)

    finally:
        print('closing pool')
        pool.close()
    