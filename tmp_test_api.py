import requests

def test_twse():
    url = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
    resp = requests.get(url, verify=False)
    data = resp.json()
    print("TWSE samples:")
    for d in data[:3]: print(d)

def test_tpex():
    url = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes"
    resp = requests.get(url)
    data = resp.json()
    print("TPEX samples:")
    for d in data[:3]: print(d)

if __name__ == "__main__":
    test_twse()
    test_tpex()
