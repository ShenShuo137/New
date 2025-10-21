import torch
from PIL import Image
from torchvision import transforms

# 这是 InternVL 官方的图像变换器
def build_transform(input_size):
    """
    构建图像预处理的变换器。
    - input_size: 模型的输入图像尺寸 (例如 448)。
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

# 这是 InternVL 官方的核心函数：动态图块处理
def dynamic_preprocess(image, image_size, use_thumbnail, max_num=12):
    """
    将单张图片动态地切割成多个图块 (patches)。
    - image: PIL.Image 对象。
    - image_size: 模型的标准输入尺寸。
    - use_thumbnail: 是否使用缩略图。
    - max_num: 最多切割成多少个图块。
    """
    if use_thumbnail:
        image = image.resize((image_size, image_size))
    
    # 获取原始宽高比
    orig_width, orig_height = image.size
    
    # 根据宽高比决定如何切割
    if orig_width > orig_height: # 横向图片
        # 垂直方向不变，水平方向切割成多块
        num = int(orig_width / orig_height)
        num = min(num, max_num)
        images = []
        for i in range(num):
            left = i * orig_height
            right = (i + 1) * orig_height
            # 切割出一个正方形区域
            crop_image = image.crop((left, 0, right, orig_height))
            images.append(crop_image)
    else: # 纵向或方形图片
        # 水平方向不变，垂直方向切割成多块
        num = int(orig_height / orig_width)
        num = min(num, max_num)
        images = []
        for i in range(num):
            top = i * orig_width
            bottom = (i + 1) * orig_width
            # 切割出一个正方形区域
            crop_image = image.crop((0, top, orig_width, bottom))
            images.append(crop_image)
            
    return images

# 这是您找到的主函数，它将上面两个函数整合起来
def load_image(image_file, input_size=448, max_num=12):
    """
    加载并处理单张图片文件，返回一个包含所有图块的张量。
    - image_file: 图片文件路径。
    - input_size: 模型输入尺寸。
    - max_num: 最大图块数。
    """
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    
    # 动态地将图片切割成一个或多个图块
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    
    # 对每个图块应用变换
    pixel_values = [transform(image) for image in images]
    
    # 将图块列表堆叠成一个单一的张量 [N, C, H, W]
    pixel_values = torch.stack(pixel_values)
    
    return pixel_values
