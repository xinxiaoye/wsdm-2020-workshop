import random
import pickle

random.seed(1234)

with open('../data/remap.pkl', 'rb') as f:
    trips_df = pickle.load(f)
    country_list = pickle.load(f)

with open('../data/IDmap.pkl', 'rb') as f:
    (city_map,country_map,user_map,booker_country_map,device_class_map,affiliate_id_map,year_map,month_map,weekday_map) = \
        pickle.load(f)

with open('../data/context_map.pkl', 'rb') as f:
    context_map = pickle.load(f)

user_count, city_count, country_count =\
    len(user_map), len(city_map), len(country_map)

import numpy as np
num_neg_samples = 100

def get_dataset(file,is_train=True):
    data_set = []

    with open(file, 'r') as f:
        lines = f.readlines()
        for ind, line in enumerate(lines):
            if ind % 1000 == 0:
                print ind, '-----'
            line = line.split()
            trip_id = line[0]
            # print trip_id
            user_id = user_map[int(trip_id.split('_')[0])]
            city_list = [city_map[city] for city in np.array(line[1:], dtype='int32').tolist()]
            
            history = city_list[:-1]
            pos_city = city_list[-1]
            
            def gen_neg():
                neg_sub_list = []
                neg = city_list[0]
                while len(neg_sub_list)!=num_neg_samples:
                    while neg in city_list or neg in neg_sub_list:
                        neg = random.randint(0, city_count-1)
                    neg_sub_list.append(neg)
                    neg = city_list[0]
                return neg_sub_list

            neg_list = gen_neg()
            if is_train:
                data_set.append((user_id, history, pos_city, 1, context_map[trip_id]))
                for j in range(num_neg_samples):
                    data_set.append((user_id, history, neg_list[j], 0, context_map[trip_id]))
            else:
                for j in range(num_neg_samples):
                    label = (pos_city, neg_list[j])
                    data_set.append((user_id, history,label, context_map[trip_id]))
    return data_set


train_set= get_dataset('../data/trips_train', is_train=True)
test_set = get_dataset('../data/trips_dev', is_train=False)

random.shuffle(train_set)
random.shuffle(test_set)

# assert len(test_set) == user_count
# assert(len(test_set) + len(train_set) // 2 == reviews_df.shape[0])

with open('../data/dataset_with_cntxt_neg_{}.pkl'.format(num_neg_samples), 'wb') as f:
    pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(country_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_count, city_count, country_count), f, pickle.HIGHEST_PROTOCOL)
