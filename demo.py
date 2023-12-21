import os
import torch

from PIL import Image

from variables import net, saved_model_path, data_transform, classes

from train_collision_avoidance import train
from test_collision_avoidance import test

inference_path = ""


def predict_class(image_path):
    image = Image.open(image_path)
    input_tensor = data_transform(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)

    softmax_output = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class = torch.argmax(softmax_output).item()

    return predicted_class


use_pretrained_model = input("Use pretrained model? [y/n] ")

if use_pretrained_model == 'n':
    train_set_path = input("What is your training path with labeled classes? ")
    testing_option = input("Would you like to test after training? [y/n] ")

    if testing_option == 'y':
        test_set_path = input("What is your testing path with labeled classes? ")
    else:
        test_set_path = None

    inference_path = input("What is your inference path? ")

    batch_size = int(input("Specify batch size: "))
    epoch_count = int(input("Specify epoch count: "))

    print("\n")

    train(train_set_path, batch_size, epoch_count)

    if testing_option == 'y':
        print("\n")

        test(test_set_path, batch_size)
elif use_pretrained_model == 'y':
    inference_path = input("What is your inference path? ")

print("\n")

model = net
model.load_state_dict(torch.load(saved_model_path))
model.eval()

for filename in os.listdir(inference_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(inference_path, filename)
        predicted_class = predict_class(image_path)
        print(f"Image: {filename}, Predicted Class: {classes[predicted_class]}")
