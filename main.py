import streamlit as st
import torch
from PIL import Image
from torchvision import transforms, models
from torchvision.models import MobileNet_V2_Weights
import torch.nn as nn

# Define class names
class_names = ['accordion', 'banjo', 'drum', 'flute', 'guitar', 'harmonica', 'saxophone', 'sitar', 'tabla', 'violin']

# Load model
model_path = 'models/trained_resnet.pt'
model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
model.fc = model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features=1280, out_features=len(class_names)),
        nn.LogSoftmax(dim=1))
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

# Streamlit UI
st.title('Musical instrument classifier')

uploaded_image = st.file_uploader('Upload an image of a musical instrument', type=('jpg', 'jpeg', 'png'))

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert('RGB')
    col_1, col_2 = st.columns(2)

    with col_1:
        resized_img = image.resize((256, 256))
        st.image(resized_img)

    with col_2:
        if st.button('Predict'):
            with torch.no_grad():
                img = transform(image).unsqueeze(0)
                out = model(img)
                _, pred = torch.max(out, 1)
                prediction = class_names[pred.item()]
                st.success(f'Prediction: {prediction}')
                st.snow()
