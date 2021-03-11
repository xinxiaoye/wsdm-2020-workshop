import pickle
import numpy as np
import heapq

with open('../data/IDmap.pkl', 'rb') as f:
    (city_map,country_map,user_map,booker_country_map,device_class_map,affiliate_id_map,year_map,month_map,weekday_map) = \
        pickle.load(f)


test_set = []

with open('../data/trips_dev', 'r') as f, open('../data/RECALL_dev_500', 'r') as f_recall:
        lines_dev = f.readlines()
        lines_recall = f_recall.readlines()
        for line_dev, line_recall in zip(lines_dev, lines_recall):
            line_dev = line_dev.split()
            trip_id = line_dev[0]
            # print trip_id
            user_id = user_map[int(trip_id.split('_')[0])]
            city_list = np.array(line_dev[1:], dtype='int32').tolist()
            # print city_list
                
            city_list = [city_map[city] for city in city_list]
            
            history = city_list[:-1]
            pos_city = city_list[-1]
            
            # recall
            recall_cities = np.array(line_recall.split()[1:], dtype='int32').tolist()
            recall_cities = [city_map[city] for city in recall_cities]
            test_set.append((user_id, history, recall_cities, pos_city))

scores_li = []
with open('../data/dev_score_recall_40_0114.pkl', 'rb') as f:
    batch_scores = pickle.load(f)

    for scores in batch_scores:
        scores = scores.reshape([-1,40])
        batch_len = scores.shape[0]
        for i in range(batch_len):
            scores_li.append(scores[i])
    
    print len(scores_li), len(test_set)

hit_num = 0
for scores_500, test_sample in zip(scores_li, test_set):
 
    index_4 = np.array(heapq.nlargest(4, range(len(scores_500)), scores_500.take))
    recall_4_cities = np.array(test_sample[2])[index_4]
    pos_city = test_sample[3]
    print pos_city, recall_4_cities
    if pos_city in recall_4_cities:
        hit_num += 1

print hit_num
print 'hitrate@4:', float(hit_num)/len(scores_li)


    