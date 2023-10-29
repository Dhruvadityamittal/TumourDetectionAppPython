from PIL import Image
from models import train_transfer_learning_model
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Load the test image

import torchvision.transforms as transforms

data_transforms = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def predict(model_name, test_image):
    tumour_dict = {0 : "Glioma Tumor", 1 : "Meningioma Tumor", 2 :"Normal",
                   3 :"Pituitary_Tumor"}
    
    model_to_use = train_transfer_learning_model(model_name).to(device)
    model_to_use.load_state_dict(torch.load("Pretrained_Models/"+model_name+".pth",
                                            map_location= device))
    model_to_use.eval()
    

    transformed_test_image = data_transforms(Image.open(test_image))

    transformed_test_image = torch.unsqueeze(transformed_test_image, 0).to(device)

    return tumour_dict[torch.argmax(model_to_use(transformed_test_image)).item()]

