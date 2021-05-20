# Recruit_Restaurant_Visitor_Forecasting
"""
Predict how many future visitors a restaurant will receive
"""

**DataSource**
- [Recruit Restaurant Visitor Forecasting](https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting/overview)
- [Weather Data for Recruit Restaurant Competition](https://www.kaggle.com/huntermcgushion/rrv-weather-data?select=air_store_info_with_nearest_active_station.csv)

**Our Process:**

### Data Exploration

1. run rrvf_eda_part_1.ipynb
2. run rrvf_eda_part_2.ipynb

### Feature Processing + Modeling

1. run rrvf_feature_engineering_additional.ipynb
2. run rrvf_modeling_1st_modularization.ipynb *(must import feature_engineering.py & utils.py)*
- Train DataSet Size: 4.3GB
- Test DataSet Size: 106MB
- Total Time: 2hr
3. *(additional)* run rrvf_modeling_1st_version_test.ipynb to test rrvf_modeling_1st_modularization.ipynb

**Reference**
1. [1st Place LGB Model(public:0.470, private:0.502)](https://www.kaggle.com/pureheart/1st-place-lgb-model-public-0-470-private-0-502/code)
2. [Winning solution (link to kernel inside)](https://www.kaggle.com/c/ga-customer-revenue-prediction/discussion/82614)
