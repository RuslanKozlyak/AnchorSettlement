import pandas as pd
import numpy as np

def map_names_to_numbers(gdf):
    level_mapping = {
    "Сверхкрупный город": 9,
    "Крупнейший город": 8,
    "Крупный город": 7,
    "Большой город": 6,
    "Средний город": 5,
    "Малый город": 4,
    "Крупное сельское поселение": 3,
    "Большое сельское поселение": 2,
    "Среднее сельское поселение": 1,
    "Малое сельское поселение": 0
    }

    aglomeration_mapping = {
        "Центр агломерации": 2,
        "В агломерации": 1,
        "Вне агломерации": 0
    }

    gdf['level'] = gdf['level'].map(level_mapping) + 1
    gdf['agglomeration_status'] = gdf['agglomeration_status'].map(aglomeration_mapping) + 1
    return gdf

def merge_with_service(gdf, provision_gdf):
    provision_gdf = provision_gdf
    provision_gdf['x'] = provision_gdf.geometry.x
    provision_gdf['y'] = provision_gdf.geometry.y
    provision_gdf = provision_gdf[['x','y','sum_provision']]
    provision_gdf['x'] = provision_gdf['x'].round(6)
    provision_gdf['y'] = provision_gdf['y'].round(6)

    gdf['x'] = gdf.geometry.x
    gdf['y'] = gdf.geometry.y
    gdf['x'] = gdf['x'].round(6)
    gdf['y'] = gdf['y'].round(6)

    gdf = pd.merge(gdf, provision_gdf, left_on=['x', 'y'], right_on=['x', 'y'], how='left')
    return gdf

def prepare_data(gdf, acc_matrix, provision_gdf, service_types):
    gdf = map_names_to_numbers(gdf)

    columns_to_sum = ["provision_" + col for col in service_types]
    provision_gdf['sum_provision'] = provision_gdf[columns_to_sum].sum(axis=1) / len(service_types)

    gdf = merge_with_service(gdf, provision_gdf)

    acc_matrix = acc_matrix.to_numpy()

    acc_matrix[acc_matrix >= 1.0+308] = np.inf
    average_distances = np.nanmean(np.where(acc_matrix == np.inf, np.nan, acc_matrix), axis=1)
    gdf['average_distances'] = average_distances
    return gdf


  