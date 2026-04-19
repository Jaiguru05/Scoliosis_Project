from model import SimpleModel
from dataset import load_dummy_data

model = SimpleModel()
data = load_dummy_data()

output = model(data)

print("Prediction shape:", output.shape)
