import gc
import numpy as np
import pandas as pd
from utils import *


class StoreFeatGenerator(object):
    
    def __init__(self, label, data, key, result):
        self.label = label
        self.data = data
        self.key = key
#         self.n_day = n_day        
        self.result = result
        
        
    def load(self):
#         data_temp = self.truncate_dataset(self.key, self.n_day)
        prev_day_list = [7, 14, 28, 56, 600]
        for prev_day in prev_day_list:
            data_temp = self.truncate_dataset(self.key, prev_day)
            self.result.append(self.get_store_visitor_feat(data_temp, self.label, self.key, prev_day))        # store features 
#             del data_temp
        
        for num in [58]:
            data_temp = self.truncate_dataset(self.key, num)
            self.result.append(self.get_store_day_diff_feat(data_temp, self.label, self.key, num))
#             del data_temp
                               
        for num in [600]:
            data_temp = self.truncate_dataset(self.key, num)
            self.result.append(self.get_store_weighted_visitor_feat(data_temp, self.label, self.key, num))          
            self.result.append(self.get_store_day_diff_feat(data_temp, self.label, self.key, num))      # store dow diff features
            self.result.append(self.get_store_dow_feat(data_temp, self.label, self.key, num))       # store all week feat
            self.result.append(self.get_store_dow_weighted_feat(data_temp, self.label, self.key, num))       # store dow exp feat
            self.result.append(self.get_store_holiday_feat(data_temp, self.label, self.key, num))        # store holiday feat        
            self.result.append(self.get_first_last_time(data_temp, self.label, self.key, num))             # first time and last time
#             del data_temp
#         gc.collect()
            
        return self.result
            
    def truncate_dataset(self, key, n_day):
        # print ('key[0]', key[0])    
        start_date = date_add_days(key[0],-n_day)
        # print ('start_date', start_date)  
        data_temp = self.data[(self.data.visit_date < key[0]) & (self.data.visit_date > start_date)]
        return data_temp                  
            

    def get_store_visitor_feat(self, data_temp, label, key, n_day):
        """
        get_store_visitor_feat + get_store_week_feat
        """
        # print ('key', key)
        # print ('n_day', n_day)
#         data_temp = truncate_dataset(key, n_day)

        result = data_temp.groupby(['store_id'], as_index=False)['visitors'].agg({'store_min{}'.format(n_day): 'min',
                                                                                 'store_mean{}'.format(n_day): 'mean',
                                                                                 'store_median{}'.format(n_day): 'median',
                                                                                 'store_max{}'.format(n_day): 'max',
                                                                                 'store_count{}'.format(n_day): 'count',
                                                                                 'store_std{}'.format(n_day): 'std',
                                                                                 'store_skew{}'.format(n_day): 'skew'})

        result2 = data_temp.groupby(['store_id', 'dow'], as_index=False)['visitors'].agg({'store_dow_min{}'.format(n_day): 'min',
                                                                                         'store_dow_mean{}'.format(n_day): 'mean',
                                                                                         'store_dow_median{}'.format(n_day): 'median',
                                                                                         'store_dow_max{}'.format(n_day): 'max',
                                                                                         'store_dow_count{}'.format(n_day): 'count',
                                                                                         'store_dow_std{}'.format(n_day): 'std',
                                                                                         'store_dow_skew{}'.format(n_day): 'skew'})    

        result = left_merge(label, result, on=['store_id']).fillna(0)
        result2 = left_merge(label, result2, on=['store_id', 'dow']).fillna(0)
        final_result = pd.concat([result, result2], axis=1)
        return final_result                
        
        
    def get_store_weighted_visitor_feat(self, data_temp, label, key, n_day):
    #     data_temp = truncate_dataset(key, n_day)
        data_temp['diff_of_day'] = data_temp['visit_date'].apply(lambda x: diff_of_days(key[0],x))
        result_list = pd.DataFrame()
    #     for weight in [0.85, 0.9, 0.95, 0.985]:
        for weight in [0.85, 0.9, 0.95, 0.985 ,0.97, 0.98, 0.99, 0.999, 0.9999]:
            # print (weight)
            data_temp['weight_{}'.format(weight)] = data_temp['diff_of_day'].apply(lambda x: weight**x)
            data_temp['visitors_{}'.format(weight)] = data_temp['visitors'] * data_temp['weight_{}'.format(weight)]
            result1 = data_temp.groupby(['store_id'], as_index=False)['visitors_{}'.format(weight)].agg({'store_exp{}_sum{}'.format(weight, n_day): 'sum'})
            result2 = data_temp.groupby(['store_id'], as_index=False)['weight_{}'.format(weight)].agg({'store_exp{}_weight_sum{}'.format(weight, n_day): 'sum'})
            result = result1.merge(result2, on=['store_id'], how='left')
            result['store_exp{}_sum{}'.format(weight, n_day)] = result['store_exp{}_sum{}'.format(weight, n_day)]/ \
                result['store_exp{}_weight_sum{}'.format(weight, n_day)]
            result = left_merge(label, result, on=['store_id']).fillna(0)   
            result_list = pd.concat([result_list, result], axis=1)

        return result_list


    def get_store_day_diff_feat(self, data_temp, label, key, n_day):
    #     data_temp = truncate_dataset(key, n_day)
        result = data_temp.set_index(['store_id','visit_date'])['visitors'].unstack()
        result = result.diff(axis=1).iloc[:,1:]
        c = result.columns
        result['store_diff_mean'] = np.abs(result[c]).mean(axis=1)
        result['store_diff_std'] = result[c].std(axis=1)
        result['store_diff_max'] = result[c].max(axis=1)
        result['store_diff_min'] = result[c].min(axis=1)
        result['store_diff_skew'] = result[c].skew(axis=1)
        result['store_diff_kurtosis'] = result[c].kurtosis(axis=1)
        result = left_merge(label, result[['store_diff_mean', 'store_diff_std', 'store_diff_max', 'store_diff_min', 'store_diff_skew', \
                                           'store_diff_kurtosis']],on=['store_id']).fillna(0)
        return result


    def get_store_dow_feat(self, data_temp, label, key, n_day):
#         data_temp = truncate_dataset(key, n_day)
        result_temp = data_temp.groupby(['store_id', 'dow'],as_index=False)['visitors'].agg({'store_dow_mean{}'.format(n_day): 'mean',
                                                                         'store_dow_median{}'.format(n_day): 'median',
                                                                         'store_dow_sum{}'.format(n_day): 'max',
                                                                         'store_dow_count{}'.format(n_day): 'count',
                                                                         'store_dow_min{}'.format(n_day): 'min',
                                                                         'store_dow_std{}'.format(n_day): 'std',
                                                                         'store_dow_skew{}'.format(n_day): 'skew'})
        result = pd.DataFrame()
        for i in range(7):
            result_sub = result_temp[result_temp['dow']==i].copy()
            result_sub = result_sub.set_index('store_id')
            result_sub = result_sub.add_prefix(str(i))
            result_sub = left_merge(label, result_sub, on=['store_id']).fillna(0)
            result = pd.concat([result,result_sub],axis=1)
        return result



    def get_store_dow_weighted_feat(self, data_temp, label, key, n_day):
#         data_temp = truncate_dataset(key, n_day)
        data_temp['diff_of_day'] = data_temp['visit_date'].apply(lambda x: diff_of_days(key[0],x))

        result = None
        for weight in [0.9,0.95,0.97,0.98,0.985,0.99,0.999,0.9999]:
            data_temp['weight'] = data_temp['diff_of_day'].apply(lambda x: weight**x)
            data_temp['visitors1'] = data_temp['visitors'] * data_temp['weight']
            result1 = data_temp.groupby(['store_id', 'dow'], as_index=False)['visitors1'].agg({'store_dow_exp_sum{}_{}'.format(n_day,weight): 'sum'})
            result2 = data_temp.groupby(['store_id', 'dow'], as_index=False)['weight'].agg({'store_dow_exp_weight_sum{}_{}'.format(n_day,weight): 'sum'})
            result_temp = result1.merge(result2, on=['store_id', 'dow'], how='left')
            result_temp['store_dow_exp_sum{}_{}'.format(n_day,weight)] = result_temp['store_dow_exp_sum{}_{}'.format(n_day,weight)]/result_temp['store_dow_exp_weight_sum{}_{}'.format(n_day,weight)]
            if result is None:
                result = result_temp
            else:
                result = result.merge(result_temp,on=['store_id','dow'],how='left')
        result = left_merge(label, result, on=['store_id', 'dow']).fillna(0)
        return result


    def get_store_holiday_feat(self, data_temp, label, key, n_day):
#         data_temp = truncate_dataset(key, n_day)
        result1 = data_temp.groupby(['store_id', 'holiday_flg'], as_index=False)['visitors'].agg(
            {'store_holiday_min{}'.format(n_day): 'min',
             'store_holiday_mean{}'.format(n_day): 'mean',
             'store_holiday_median{}'.format(n_day): 'median',
             'store_holiday_max{}'.format(n_day): 'max',
             'store_holiday_count{}'.format(n_day): 'count',
             'store_holiday_std{}'.format(n_day): 'std',
             'store_holiday_skew{}'.format(n_day): 'skew'})
        result1 = left_merge(label, result1, on=['store_id', 'holiday_flg']).fillna(0)
        result2 = data_temp.groupby(['store_id', 'holiday_flg2'], as_index=False)['visitors'].agg(
            {'store_holiday2_min{}'.format(n_day): 'min',
             'store_holiday2_mean{}'.format(n_day): 'mean',
             'store_holiday2_median{}'.format(n_day): 'median',
             'store_holiday2_max{}'.format(n_day): 'max',
             'store_holiday2_count{}'.format(n_day): 'count',
             'store_holiday2_std{}'.format(n_day): 'std',
             'store_holiday2_skew{}'.format(n_day): 'skew'})
        result2 = left_merge(label, result2, on=['store_id', 'holiday_flg2']).fillna(0)
        result = pd.concat([result1, result2], axis=1)
        return result        

    def get_first_last_time(self, data_temp, label, key, n_day):
#         data_temp = truncate_dataset(key, n_day)
        data_temp = data_temp.sort_values('visit_date')

        result = data_temp.groupby('store_id')['visit_date'].agg([("first_time", lambda x: diff_of_days(key[0],np.min(x))),
                                                                  ('last_time',lambda x: diff_of_days(key[0],np.max(x)))])
        result = left_merge(label, result, on=['store_id']).fillna(0)
        return result


    
    
    
class GenreFeatGenerator(StoreFeatGenerator):

    def __init__(self, label, data, key, result):
        super().__init__(label, data, key, result)
        
        
    def load(self):
#         data_temp = self.truncate_dataset(self.key, self.n_day)
        prev_day_list = [7, 14, 28, 56, 600]
        for prev_day in prev_day_list:
            data_temp = self.truncate_dataset(self.key, prev_day)
            self.result.append(self.get_genre_visitor_feat(data_temp, self.label, self.key, prev_day))   # store features        
#             del data_temp
        
        for num in [600]:
            data_temp = self.truncate_dataset(self.key, num)
            self.result.append(self.get_genre_weighted_visitor_feat(data_temp, self.label, self.key, num))          
            self.result.append(self.get_genre_dow_weighted_feat(data_temp, self.label, self.key, num))       # store dow exp feat
#             del data_temp
            
#         gc.collect()
            
        return self.result        
    
    
    def get_genre_visitor_feat(self, data_temp, label, key, n_day):
        """
        get_genre_visitor_feat + get_genre_week_feat 
        """
    #     data_temp = truncate_dataset(key, n_day)
        result = data_temp.groupby(['air_genre_name'], as_index=False)['visitors'].agg({'genre_min{}'.format(n_day): 'min',
                                                                                 'genre_mean{}'.format(n_day): 'mean',
                                                                                 'genre_median{}'.format(n_day): 'median',
                                                                                 'genre_max{}'.format(n_day): 'max',
                                                                                 'genre_count{}'.format(n_day): 'count',
                                                                                 'genre_std{}'.format(n_day): 'std',
                                                                                 'genre_skew{}'.format(n_day): 'skew'})

        result2 = data_temp.groupby(['air_genre_name', 'dow'], as_index=False)['visitors'].agg({'genre_dow_min{}'.format(n_day): 'min',
                                                                                             'genre_dow_mean{}'.format(n_day): 'mean',
                                                                                             'genre_dow_median{}'.format(n_day): 'median',
                                                                                             'genre_dow_max{}'.format(n_day): 'max',
                                                                                             'genre_dow_count{}'.format(n_day): 'count',
                                                                                             'genre_dow_std{}'.format(n_day): 'std',
                                                                                             'genre_dow_skew{}'.format(n_day): 'skew'})

        result = left_merge(label, result, on=['air_genre_name']).fillna(0)
        result2 = left_merge(label, result2, on=['air_genre_name', 'dow']).fillna(0)
        final_result = pd.concat([result, result2], axis=1)

        return final_result


    def get_genre_weighted_visitor_feat(self, data_temp, label, key, n_day):
    #     data_temp = truncate_dataset(key, n_day)
        data_temp['diff_of_day'] = data_temp['visit_date'].apply(lambda x: diff_of_days(key[0],x))
        result_list = pd.DataFrame()
    #     for weight in [0.85, 0.9, 0.95, 0.985]:
        for weight in [0.85, 0.9, 0.95, 0.985 ,0.97, 0.98, 0.99, 0.999, 0.9999]:
            # print (weight)
            data_temp['weight_{}'.format(weight)] = data_temp['diff_of_day'].apply(lambda x: weight**x)
            data_temp['visitors_{}'.format(weight)] = data_temp['visitors'] * data_temp['weight_{}'.format(weight)]
            result1 = data_temp.groupby(['air_genre_name'], as_index=False)['visitors_{}'.format(weight)].agg({'genre_exp{}_sum{}'.format(weight, n_day): 'sum'})
            result2 = data_temp.groupby(['air_genre_name'], as_index=False)['weight_{}'.format(weight)].agg({'genre_exp{}_weight_sum{}'.format(weight, n_day): 'sum'})
            result = result1.merge(result2, on=['air_genre_name'], how='left')
            result['genre_exp{}_sum{}'.format(weight, n_day)] = result['genre_exp{}_sum{}'.format(weight, n_day)]/ \
                result['genre_exp{}_weight_sum{}'.format(weight, n_day)]
            result = left_merge(label, result, on=['air_genre_name']).fillna(0)   
            result_list = pd.concat([result_list, result], axis=1)    

        return result_list


    def get_genre_dow_weighted_feat(self, data_temp, label, key, n_day):
#         data_temp = truncate_dataset(key, n_day)
        data_temp['diff_of_day'] = data_temp['visit_date'].apply(lambda x: diff_of_days(key[0],x))
        result_list = pd.DataFrame()
    #     for weight in [0.85, 0.9, 0.95, 0.985]:
        for weight in [0.9,0.95,0.97,0.98,0.985,0.99,0.999,0.9999]:
            # print (weight)
            data_temp['weight_{}'.format(weight)] = data_temp['diff_of_day'].apply(lambda x: weight**x)
            data_temp['visitors_{}'.format(weight)] = data_temp['visitors'] * data_temp['weight_{}'.format(weight)]
            result1 = data_temp.groupby(['air_genre_name', 'dow'], as_index=False)['visitors_{}'.format(weight)].agg({'genre_dow_exp{}_sum{}'.format(weight, n_day): 'sum'})
            result2 = data_temp.groupby(['air_genre_name', 'dow'], as_index=False)['weight_{}'.format(weight)].agg({'genre_dow_exp{}_weight_sum{}'.format(weight, n_day): 'sum'})
            result = result1.merge(result2, on=['air_genre_name', 'dow'], how='left')
            result['genre_dow_exp{}_sum{}'.format(weight, n_day)] = result['genre_dow_exp{}_sum{}'.format(weight, n_day)]/ \
                result['genre_dow_exp{}_weight_sum{}'.format(weight, n_day)]
            result = left_merge(label, result, on=['air_genre_name', 'dow']).fillna(0)   
            result_list = pd.concat([result_list, result], axis=1)    

        return result_list    