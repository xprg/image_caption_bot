import json

descriptions=None

with open('/content/drive/MyDrive/descriptions.txt','r') as f:
  descriptions=f.read()

json_acceptable_string=descriptions.replace("'","\"")
descriptions=json.loads(json_acceptable_string)




