import requests

url = "https://earthwcommunity.maps.arcgis.com/arcgis/rest/services/Global_Data_Set_XvsX_0/FeatureServer/0/query"
params = {
    "where": "1=1",
    "outFields": "*",
    "f": "json",
    "resultRecordCount": 10  # just test with 10 first
}
r = requests.get(url, params=params)
print(r.status_code)
print(r.json())