import torch
import torchvision
import torchvision.transforms as transforms
import gradio as gr
import numpy as np


if __name__ == '__main__':

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    toPIL = transforms.ToPILImage()

    sigma = 0.05
    N = 10

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    import matplotlib.pyplot as plt
    import numpy as np
    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
        

    import torch.optim as optim
    
    criterion = nn.CrossEntropyLoss()
    net = Net()

    ########### EVERYTHING ABOVE IS MODEL TRAINING ###########

    # Gradient Generating Method
    def returnGrad(img, model, criterion, device = 'cpu'):
        model.to(device)
        img = img.to(device)
        img.requires_grad_(True).retain_grad()
        pred = model(img)
        loss = criterion(pred, torch.tensor([int(torch.max(pred[0], 0)[1])]).to(device))
        loss.backward()
        
    #    S_c = torch.max(pred[0].data, 0)[0]
        Sc_dx = img.grad
        
        return Sc_dx

    def smoothgrad(img):
        img = torch.tensor(img)
        sg_total = torch.zeros_like(img, dtype = torch.float32)
        for i in range(N):
            noise = torch.tensor(np.random.normal(0, sigma, img.shape), dtype = torch.float32)
            noise_img = img + noise
            sg_total += returnGrad(noise_img, model = net, criterion = criterion)
        return sg_total
    
    def VisualizeImageGrayscale(image_3d):
        r"""Returns a 3D tensor as a grayscale normalized between 0 and 1 2D tensor.
        """
        vmin = torch.min(image_3d)
        image_2d = image_3d - vmin
        vmax = torch.max(image_2d)
        return (image_2d / vmax)

    def VisualizeNumpyImageGrayscale(image_3d):
        r"""Returns a 3D tensor as a grayscale normalized between 0 and 1 2D tensor.
        """
        vmin = np.min(image_3d)
        image_2d = image_3d - vmin
        vmax = np.max(image_2d)
        return (image_2d / vmax)

    def smoothgradchange(img, sig, Number):
        img = img.transpose(2, 1, 0)
        img = torch.tensor(img).unsqueeze(0)
        sigma = sig
        N = Number
        sg = smoothgrad(img)
        sg = sg.numpy().squeeze(0).transpose(1,2,0)
        print(np.shape(sg))
        print(sg)
        sg = toPIL((VisualizeNumpyImageGrayscale(sg)*255).astype('uint8'))
        return gr.Image.update(value = sg)
    
    with gr.Blocks() as demo:
        img = gr.Image()
        sig = gr.Slider(0, 1, None, 0.1, label = "sigma")
        Number = gr.Slider(1, 20, None, step = 1, label = "N")
        simulate_btn = gr.Button("simulate")
        simulate_btn.click(fn = smoothgradchange, 
                           inputs=[img, sig, Number], 
                           outputs= img 
                           )
    demo.launch()