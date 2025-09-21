import uvicorn
import torchvision.transforms as transforms
from PIL import Image
import torch
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.templating import Jinja2Templates
import os

# 랜덤 시드 고정
torch.manual_seed(777)

# torch.nn.Module:  PyTorch의 모든 Neural Network의 Base Class
class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # 첫번째층
        # ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 두번째층
        # ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 전결합층 7x7x64 inputs -> 10 outputs
        self.fc = torch.nn.Linear(7 * 7 * 64, 10, bias=True)

        # 전결합층 한정으로 가중치 초기화
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)   # 전결합층을 위해서 Flatten
        out = self.fc(out)
        return out


app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/")
def hello(request: Request):
    return templates.TemplateResponse("index.html", {'request': request,'a':2})

@app.post('/uploader')
async def uploader_file(request:Request, file:UploadFile = File(...)):
    content = await file.read()
    file_save_folder = './'
    with open(os.path.join(file_save_folder, file.filename), "wb") as fp:
        fp.write(content)
    output = infer(file_save_folder+file.filename)
    return templates.TemplateResponse("CNN_result.html", {'request':request, 'result':output})

def infer(filename): #다시 불러와서 추론 하기
    model = CNN()
    model.load_state_dict(torch.load("cnn_model.pt", map_location=torch.device('cpu')))

    model.eval()  #평가모드로 설정하영야 함. 이 과정을 거치지 않으면 일관성 없는 추론결과가 출력
    with torch.no_grad():  #학습을 진행하지 않을 것이므로 torch.no_grad(), gradient descent를 하지마라고 명령하는 것
        img = Image.open(filename)
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28,28)), # 모델 인풋에 맞게 
            transforms.ToTensor(), #torchtensor 타입으로 맞춰주기
        ])

        img_tensor = transform(img) #[1,28,28]
        img_tensor = img_tensor.unsqueeze(0) #[1,1,28,28]

        print(img_tensor.shape)

        prediction = model(img_tensor)
        # cnn은 10개의 output으로 각 10개의 클래스에 대한 feature값이 나온다. 
            # 이를 axis 1방향으로 max값을 찾는다는 것 

        result = torch.argmax(prediction, 1) #tensor([결과])
        result = result.tolist()[0] #결과라고 나오도록
        return result
    
if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8888)
