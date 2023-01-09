import urllib.request 


def download(target_path:str, end_date: str, start_date="2004-09-06"):
    url = "https://sdw-wsrest.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.?startPeriod="+start_date+"&endPeriod="+end_date+"&format=csvdata"
    urllib.request.urlretrieve(url, target_path)