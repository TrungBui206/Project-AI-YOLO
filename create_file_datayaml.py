content = """path: dataset
train: images/train
val: images/val

nc: 2
names:
  0: car
  1: truck
"""

with open("data.yaml", "w", encoding="utf-8") as f:
    f.write(content)

print("Created data.yaml successfully!")