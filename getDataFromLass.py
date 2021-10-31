import requests
from requests.structures import CaseInsensitiveDict
import numpy as np
import json
def getLassData():
        
    url = "https://pm25.lass-net.org/API-1.0.0/device/B827EB8390F7/history/"

    headers = CaseInsensitiveDict()
    headers["accept"] = "application/json"


    resp = requests.get(url , headers = headers)
    text = resp.text
    text = json.loads(text)


    feeds = text["feeds"]
    MAPS = feeds[0]["MAPS"]

    T10value = []
    for i in reversed(range(11)):
        values = list(MAPS[-(i+1)].values())[0]
        s_d0 = values["s_d0"]
        s_d1 = values["s_d1"]
        s_d2 = values["s_d2"]
        s_g8 = values["s_g8"]
        s_h0 = values["s_h0"]
        s_gg = values["s_gg"]
        fan = 0
        ac = 0
        T10value.append([s_d0,s_d1,s_d2,s_g8,s_gg,s_h0,fan,ac])
    return T10value
        
        




