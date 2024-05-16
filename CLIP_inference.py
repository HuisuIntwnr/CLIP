import os
import clip
import torch
from torchvision.datasets import CIFAR100

def inference(model, preprocess, image, text):
    # Preprocess the inputs
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_input = clip.tokenize(f"a photo of a {text}").to(device)

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)    # input: PIL image
        text_features = model.encode_text(text_input)

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T)

    return similarity

def inference_image_set(model, preprocess, device, image_set, text):
    # Preprocess the inputs
    image_inputs = torch.cat([preprocess(image).unsqueeze(0) for image in image_set]).to(device)
    text_input = clip.tokenize(f"a photo of a {text}").to(device)

    # Calculate features
    with torch.no_grad():
        images_features = torch.cat([model.encode_image(image_input.unsqueeze(0)) for image_input in image_inputs]) #.to(device)    # input: PIL image
        text_features = model.encode_text(text_input)

    # Pick the top 5 most similar labels for the image
    images_features /= images_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * text_features @ images_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(5)

    return values, indices




if __name__ == '__main__':
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    # Download the dataset
    cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

    # Prepare the inputs
    image, class_id = cifar100[0]        # image: PIL image(32, 32)
    image_set = [cifar100[i][0].resize((320,320)) for i in range(10)]
    index_set = [cifar100[i][1] for i in range(10)]

    scores, indices = inference_image_set(model, preprocess, image_set, 'camel')

    # Print the result
    image_set[indices[0]].show()
    print("\nTop predictions:\n")
    for value, index in zip(scores, indices):
        #todo : class index 수정
        print(f"{cifar100.classes[index_set[index]]:>16s}: {100 * value.item():.2f}%")
 

