import os
import pickle
import numpy as np
import random
import config as cfg


from PIL import Image
import torchvision.transforms as transforms
from models.utils import TransformerCV
from models.groupnet import group_config
def get_queries_dict(filename):
    # key:{'query':file,'positives':[files],'negatives:[files], 'neighbors':[keys]}
    with open(filename, 'rb') as handle:
        queries = pickle.load(handle)
        print("Queries Loaded.")
        return queries


def get_sets_dict(filename):
    #[key_dataset:{key_pointcloud:{'query':file,'northing':value,'easting':value}},key_dataset:{key_pointcloud:{'query':file,'northing':value,'easting':value}}, ...}
    with open(filename, 'rb') as handle:
        trajectories = pickle.load(handle)
        print("Trajectories Loaded.")
        return trajectories



def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])

def transformImg(img):

    pts_step = 5 
    transformer = TransformerCV(group_config)


    xs, ys = np.meshgrid(np.arange(pts_step, img.shape[1] - pts_step, pts_step),
                         np.arange(pts_step, img.shape[2] - pts_step, pts_step))
    xs = xs.reshape(-1, 1)
    ys = ys.reshape(-1, 1)
    pts = np.hstack((xs, ys))


    img = img.permute(1, 2, 0).detach().numpy()


    transformed_imgs = transformer.transform(img, pts)
    data = transformer.postprocess_transformed_imgs(transformed_imgs)

    return data


def tensors_to_numpy(tensor_lists):
    numpy_lists = []
    for tensor_list in tensor_lists:
        numpy_list = [tensor.cpu().numpy() for tensor in tensor_list]  
        numpy_lists.append(numpy_list)
    return numpy_lists

def load_image_file(filename):
    
    dataset_folder='/data/soomoklee/data/bev_benchmark_datasets'
    file_path=os.path.join(dataset_folder, filename)
    img = Image.open(file_path).convert('RGB')
    transform = input_transform()
    img = transform(img)  

    img = img * 255


    img = transformImg(img)

    return img



def load_image_files(filenames):
    images = []

    if isinstance(filenames, str):
        filenames = [filenames]

    for filename in filenames:

        img = load_image_file(filename)
        images.append(img)

    return images


        
    return  img, index   

def get_query_tuple(dict_value, num_pos, num_neg, QUERY_DICT, hard_neg=[], other_neg=False):
        # get query tuple for dictionary entry
        # return list [query,positives,negatives]

    #print(dict_value["query"])
    query = load_image_files(dict_value["query"])  # Nx3

    random.shuffle(dict_value["positives"])
    pos_files = []

    for i in range(num_pos):
        pos_files.append(QUERY_DICT[dict_value["positives"][i]]["query"])
    #positives= load_pc_files(dict_value["positives"][0:num_pos])
    positives = load_image_files(pos_files)

    neg_files = []
    neg_indices = []
    if(len(hard_neg) == 0):
        random.shuffle(dict_value["negatives"])
        for i in range(num_neg):
            neg_files.append(QUERY_DICT[dict_value["negatives"][i]]["query"])
            neg_indices.append(dict_value["negatives"][i])

    else:
        random.shuffle(dict_value["negatives"])
        for i in hard_neg:
            neg_files.append(QUERY_DICT[i]["query"])
            neg_indices.append(i)
        j = 0
        while(len(neg_files) < num_neg):

            if not dict_value["negatives"][j] in hard_neg:
                neg_files.append(
                    QUERY_DICT[dict_value["negatives"][j]]["query"])
                neg_indices.append(dict_value["negatives"][j])
            j += 1

    negatives = load_image_files(neg_files)

    if other_neg is False:

        return [query, positives, negatives]
    # For Quadruplet Loss
    else:
        print("aaa")
        # get neighbors of negatives and query
        neighbors = []
        for pos in dict_value["positives"]:
            neighbors.append(pos)
        for neg in neg_indices:
            for pos in QUERY_DICT[neg]["positives"]:
                neighbors.append(pos)
        possible_negs = list(set(QUERY_DICT.keys())-set(neighbors))
        random.shuffle(possible_negs)

        if(len(possible_negs) == 0):
            return [query, positives, negatives, np.array([])]

        neg2 = load_image_file(QUERY_DICT[possible_negs[0]]["query"])

        
        return [query, positives, negatives, neg2]


