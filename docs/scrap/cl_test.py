import requests

# url = "https://www.courtlistener.com/api/rest/v4/clusters/"
url = "https://www.courtlistener.com/api/rest/v4/opinions"

headers = {"Authorization": "Token 4cd64f7c2c009657cfa3133893d1cd846cfa7050"}

resp = requests.get(url, headers=headers)
data = resp.json()
print(data.keys())
print(data['count'])
print(len(data["results"]))
# print(data["results"][0])

print(data["results"][0]["plain_text"][:500])
print(data["results"][0]["id"])
# print(data["results"][0]["plain_text"])