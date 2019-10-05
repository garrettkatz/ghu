import json

with open("datadsst.json", "r") as file:
	result1 = json.load(file)

for v in result1.values():
	print(len(v[0]),len(v[1]))