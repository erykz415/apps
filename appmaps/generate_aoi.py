import json


def bbox_to_geojson(min_lon, min_lat, max_lon, max_lat, name="AOI", filename="aoi.geojson"):
    # Definicja poligonu
    polygon = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": name
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [min_lon, min_lat],
                        [max_lon, min_lat],
                        [max_lon, max_lat],
                        [min_lon, max_lat],
                        [min_lon, min_lat]  # zamknięcie poligonu
                    ]]
                }
            }
        ]
    }

    # Zapis do pliku
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(polygon, f, indent=2)

    print(f"Plik AOI zapisany jako: {filename}")



# bbox_to_geojson(-74.03, 40.70, -73.90, 40.85, name="New York", filename="newyork_aoi.geojson")
# bbox_to_geojson(100.0, 0.5, 120.0, 7.5, name="Malezja", filename="malezja_aoi.geojson")
# bbox_to_geojson(
#     min_lon=116.88801742392343,
#     min_lat=5.23720622276667,
#     max_lon=117.32715531433384,
#     max_lat=5.676344113177066,
#     name="Polska",
#     filename="malaysia_aoi.geojson"
# )
#
# bbox_to_geojson(
#     min_lon=79.65,
#     min_lat=5.9,
#     max_lon=81.9,
#     max_lat=9.85,
#     name="Sri Lanka",
#     filename="srilanka_aoi.geojson"
# )

bbox_to_geojson(
    min_lon=4.86,
    min_lat=52.34,
    max_lon=5.00,
    max_lat=52.42,
    name="Amsterdam",
    filename="amsterdam_aoi.geojson"
)
