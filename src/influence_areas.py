import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union

IN_AGGLOMERATION = {}
default_crs = 'EPSG:32633'

def _get_agglomeration_around_node(start_node, max_time, towns, accessibility_matrix, radius):

    distances_from_start = accessibility_matrix.loc[start_node]
    within_time_nodes = distances_from_start[distances_from_start <= max_time].index
    if within_time_nodes.empty:
        return None

    nodes_data = towns.set_index('id').loc[within_time_nodes]
    nodes_data['geometry'] = nodes_data.apply(lambda row: Point(row['geometry'].x, row['geometry'].y), axis=1)
    nodes_gdf = gpd.GeoDataFrame(nodes_data, geometry='geometry', crs=default_crs)

    distance = {node: (max_time - distances_from_start[node]) * radius for node in within_time_nodes}
    nodes_gdf["left_distance"] = nodes_gdf.index.map(distance)
    agglomeration_geom = nodes_gdf.buffer(nodes_gdf["left_distance"]).union_all()

    return {
        "geometry": agglomeration_geom,
        "nodes_in_agglomeration": list(within_time_nodes)
    }


def _build_influence_areas(towns, accessibility_matrix, min_influence_score, radius):
    node_influence = towns.set_index('id')['influence_score']
    node_names = towns.set_index('id')['name']
    agglomerations = []

    levels = sorted(towns['level'].unique())

    for level_index, level in enumerate(reversed(levels)):
        max_time = 80 - 10 * level_index
        level_nodes = towns[towns['level'] == level].sort_values(by='influence_score', ascending=False)

        for node, influence in level_nodes[['id', 'influence_score']].itertuples(index=False):
            if node in IN_AGGLOMERATION or influence < min_influence_score:
                continue

            agglomeration = _get_agglomeration_around_node(node, max_time, towns, accessibility_matrix, radius)

            if agglomeration:
                agglomeration["name"] = node_names.get(node)
                agglomerations.append(agglomeration)

                for member_node in agglomeration["nodes_in_agglomeration"]:
                    if node_influence.get(member_node) < min_influence_score:
                        IN_AGGLOMERATION[member_node] = True

    if agglomerations:
        influence_gdf = gpd.GeoDataFrame(
            agglomerations, columns=["name",  "geometry"]
        ).set_geometry('geometry')
        influence_gdf.set_crs(default_crs, inplace=True)
    else:
        influence_gdf = gpd.GeoDataFrame(columns=["name", "geometry"])

    return influence_gdf

def _merge_intersecting_areas(gdf, towns):
    merged_geometries = []
    processed_indices = set()

    # Основной цикл по агломерациям
    for i, row_i in gdf.iterrows():
        if i in processed_indices:
            continue

        overlapping_agglomerations = [row_i]
        geometry = row_i['geometry']
        merged_names = {row_i['name']}

        # Первый цикл: проверка пересечений и объединение агломераций
        for j, row_j in gdf.iterrows():
            if i != j and j not in processed_indices:
                if geometry.intersects(row_j['geometry']):
                    overlapping_agglomerations.append(row_j)
                    geometry = unary_union([geometry, row_j['geometry']])  # Используем unary_union
                    merged_names.add(row_j['name'])
                    processed_indices.add(j)

        # Второй цикл: дополнительная проверка для объединения с новыми полигонами
        still_intersecting = True
        while still_intersecting:
            still_intersecting = False
            for j, row_j in gdf.iterrows():
                if j not in processed_indices:
                    if geometry.intersects(row_j['geometry']):
                        overlapping_agglomerations.append(row_j)
                        geometry = unary_union([geometry, row_j['geometry']])  # Используем unary_union
                        merged_names.add(row_j['name'])
                        processed_indices.add(j)
                        still_intersecting = True

        # Проверяем валидность геометрии и исправляем ее, если необходимо
        if not geometry.is_valid:
            geometry = geometry.buffer(0)

        # Вычисляем влияние, ядро зоны и его координаты
        towns_in_agglomeration = towns[towns.intersects(geometry)]
        influence_from_towns = towns_in_agglomeration['influence_score'].sum()

        if not towns_in_agglomeration.empty:
            core_town = towns_in_agglomeration.loc[towns_in_agglomeration['influence_score'].idxmax()]
            core_name = core_town['name']
            core_influence = core_town['influence_score']
            core_location = core_town['geometry']
        else:
            core_name = None
            core_influence = 0
            core_location = None

        merged_agglomeration = {
            'geometry': geometry,
            'type': 'Polycentric' if len(merged_names) > 1 else 'Monocentric',
            'core_cities': ', '.join(merged_names),
            'influence_score': influence_from_towns,
            'core_name': core_name,
            'core_influence': core_influence,
            'core_location': core_location,  # Сохраняем координаты ядра
        }
        merged_geometries.append(merged_agglomeration)

        processed_indices.add(i)

    return gpd.GeoDataFrame(merged_geometries, crs=gdf.crs)

    merged_geometries = []
    processed_indices = set()

    # Основной цикл по агломерациям
    for i, row_i in gdf.iterrows():
        if i in processed_indices:
            continue

        overlapping_agglomerations = [row_i]
        geometry = row_i['geometry']
        merged_names = {row_i['name']}

        # Первый цикл: проверка пересечений и объединение агломераций
        for j, row_j in gdf.iterrows():
            if i != j and j not in processed_indices:
                if geometry.intersects(row_j['geometry']):
                    overlapping_agglomerations.append(row_j)
                    geometry = unary_union([geometry, row_j['geometry']])  # Используем unary_union
                    merged_names.add(row_j['name'])
                    processed_indices.add(j)

        # Второй цикл: дополнительная проверка для объединения с новыми полигонами
        still_intersecting = True
        while still_intersecting:
            still_intersecting = False
            for j, row_j in gdf.iterrows():
                if j not in processed_indices:
                    if geometry.intersects(row_j['geometry']):
                        overlapping_agglomerations.append(row_j)
                        geometry = unary_union([geometry, row_j['geometry']])  # Используем unary_union
                        merged_names.add(row_j['name'])
                        processed_indices.add(j)
                        still_intersecting = True

        # Проверяем валидность геометрии и исправляем ее, если необходимо
        if not geometry.is_valid:
            geometry = geometry.buffer(0)

        # Вычисляем влияние и ядро зоны
        towns_in_agglomeration = towns[towns.intersects(geometry)]
        influence_from_towns = towns_in_agglomeration['influence_score'].sum()

        if not towns_in_agglomeration.empty:
            core_town = towns_in_agglomeration.loc[towns_in_agglomeration['influence_score'].idxmax()]
            core_name = core_town['name']
            core_influence = core_town['influence_score']
        else:
            core_name = None
            core_influence = 0

        merged_agglomeration = {
            'geometry': geometry,
            'type': 'Polycentric' if len(merged_names) > 1 else 'Monocentric',
            'core_cities': ', '.join(merged_names),
            'influence_score': influence_from_towns,
            'core_name': core_name,
            'core_influence': core_influence,
        }
        merged_geometries.append(merged_agglomeration)

        processed_indices.add(i)

    return gpd.GeoDataFrame(merged_geometries, crs=gdf.crs)


def _simplify_multipolygons( gdf):
    """
    Simplifies multipolygons to keep only the largest polygon for each agglomeration.

    Parameters:
    - gdf: GeoDataFrame containing the agglomeration geometries.

    Returns:
    - A GeoDataFrame with simplified geometries.
    """
    gdf['geometry'] = gdf['geometry'].apply(
        lambda geom: max(geom.geoms, key=lambda g: g.area) if isinstance(geom, MultiPolygon) else geom
    )
    return gdf


def get_influence_areas(towns, accessibility_matrix, radius=60, min_influence_score=0.3):
    towns = towns.to_crs(default_crs)

    influence_gdf = _build_influence_areas(towns, accessibility_matrix, min_influence_score, radius)

    influence_gdf = _simplify_multipolygons(influence_gdf)

    influence_gdf = _merge_intersecting_areas(influence_gdf, towns)

    influence_gdf = _simplify_multipolygons(influence_gdf)

    influence_gdf['geometry'] = influence_gdf['geometry'].apply(
        lambda geom: Polygon(geom.exterior) if geom.is_valid else geom
    )

    return influence_gdf

def calculate_influence_score(gdf, weights_dict):
    scaler = MinMaxScaler()

    # Список колонок, которые будут использоваться для вычислений
    features_columns = ['population', 'level', 'agglomeration_status', 
                        'agglomeration_level', 'average_distances', 'sum_provision']

    # Проверяем, что все ключи из features_columns присутствуют в weights_dict
    missing_columns = [col for col in features_columns if col not in weights_dict]
    if missing_columns:
        raise ValueError(f"Weights are missing for columns: {missing_columns}")

    # Нормализуем данные
    data = gdf.copy()
    data = pd.DataFrame(
        scaler.fit_transform(data[features_columns]),
        columns=features_columns
    )

    # Преобразуем словарь весов в список весов в том же порядке, что и features_columns
    weights = [weights_dict[col] for col in features_columns]

    # Вычисляем итоговые значения
    gdf['sum'] = data.dot(weights)
    gdf['influence_score'] = (gdf['sum'] - gdf['sum'].min()) / (gdf['sum'].max() - gdf['sum'].min())

    return gdf