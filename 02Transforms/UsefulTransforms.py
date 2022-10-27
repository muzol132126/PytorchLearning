from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

img = Image.open("dataset/train/ants/0013035.jpg")

writer = SummaryWriter("logs")

# 1. PIL to Tensor
trans_totensor = transforms.ToTensor()
tensor_img = trans_totensor(img)
# print(tensor_img)
writer.add_image("ToTensor", tensor_img)

# 2. Nomalize
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm = trans_norm(tensor_img) # type=Tensor
# print(img_norm)
writer.add_image("Normalize", img_norm)

# 3. Resize
print(img.size)
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize(img)
writer.add_image("Resize", trans_totensor(img_resize))
print(img_resize.size)

# 4. Compose
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans_totensor, trans_norm])
img_resize_2 = trans_compose(img)
writer.add_image("Resize2", img_resize_2, 1)

# 5. RandomCrop
trans_random = transforms.RandomCrop(512)
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, 1)



writer.close()