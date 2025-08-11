import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from patch_internvl import patch_model

#ImageNet 数据集的均值和标准差，用于图像归一化
IMAGENET_MEAN = (0.485, 0.456, 0.406) #均值
IMAGENET_STD = (0.229, 0.224, 0.225) #标准差

#图像预处理函数
def build_transform(input_size):
    """
    构建图像预处理转换流程。

    Args:
        input_size (int): 目标图像的尺寸，用于调整图像大小。

    Returns:
        torchvision.transforms.Compose: 包含一系列图像转换操作的组合对象。

    Note:
        该转换流程包含以下步骤：
        1. 确保图像为RGB模式
        2. 调整图像大小为指定尺寸
        3. 将图像转换为张量
        4. 使用ImageNet的均值和标准差进行归一化

    Example:
        >>> transform = build_transform(224)
        >>> img = Image.open("example.jpg")
        >>> img_tensor = transform(img)
    """
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img), #将图像转换为RGB模式
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),#调整图像大小为
        T.ToTensor(), #将图像转换为张量
        T.Normalize(mean=MEAN, std=STD) #归一化
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """
    根据给定的目标宽高比列表，找到与输入图像宽高比最接近的目标宽高比。
    
    参数:
        aspect_ratio (float): 输入图像的宽高比（width/height）
        target_ratios (list): 目标宽高比列表，每个元素是一个元组(width, height)
        width (int): 输入图像的宽度
        height (int): 输入图像的高度
        image_size (int): 目标图像的尺寸
    
    返回:
        tuple: 最佳匹配的目标宽高比，格式为(width, height)
    
    算法说明:
        1. 初始化最佳差异为无穷大，最佳比例为(1,1)
        2. 计算输入图像的面积
        3. 遍历所有目标宽高比:
           a. 计算当前目标宽高比的值
           b. 计算与输入宽高比的差异
           c. 如果找到更好的匹配(差异更小)，则更新最佳比例
           d. 如果差异相同，则选择面积更大的比例
        4. 返回最佳匹配的比例
    """
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """
    动态预处理图像，将其分割成多个块以适应不同的宽高比。

    参数:
        image (PIL.Image): 输入的图像对象。
        min_num (int, optional): 分割块的最小数量，默认为1。
        max_num (int, optional): 分割块的最大数量，默认为12。
        image_size (int, optional): 每个分割块的大小，默认为448。
        use_thumbnail (bool, optional): 是否添加缩略图，默认为False。

    返回:
        list: 包含处理后的图像块的列表。

    功能说明:
        1. 计算原始图像的宽高比。
        2. 生成符合最小和最大块数限制的所有可能的宽高比组合。
        3. 找到与原始图像宽高比最接近的目标宽高比。
        4. 根据目标宽高比计算目标宽度和高度，并确定分割块的数量。
        5. 将图像调整到目标大小并分割成多个块。
        6. 如果需要，添加一个缩略图到结果列表中。
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    """
    加载并预处理图像文件，将其转换为模型可接受的格式。

    Args:
        image_file (str): 图像文件的路径。
        input_size (int, optional): 输入图像的目标大小。默认值为448。
        max_num (int, optional): 最大处理的图像数量。默认值为12。

    Returns:
        torch.Tensor: 预处理后的图像像素值张量，形状为 (batch_size, channels, height, width)。

    Note:
        该函数会执行以下步骤：
        1. 打开图像文件并转换为RGB格式
        2. 构建图像预处理转换
        3. 动态预处理图像，生成多个缩略图（如果需要）
        4. 对每张图像应用预处理转换
        5. 将所有图像像素值堆叠成一个张量
    """
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# If you have an 80G A100 GPU, you can put the entire model on a single GPU.
# Otherwise, you need to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.

def load_model(
    model_path: str = "OpenGVLab/InternVL2_5-8B",
    dtype: torch.dtype = torch.bfloat16,
    quant_method: str = "qjl",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> tuple:
    """
    加载预训练模型并应用量化
    
    参数:
        model_path: 模型路径或HuggingFace标识符
        dtype: 模型计算精度
        quant_method: 量化方法
        device: 目标设备
        
    返回:
        (model, tokenizer) 元组
    """
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map='auto' if device == "cuda" else None
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        use_fast=False
    )
    model = model.eval().requires_grad_(False).to(device)
    patch_model(model, quant_method)
    return model, tokenizer

def run_inference(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    image_path: str = "test.png",
    question: str = "<image>\nDescribe the image in 100 words.",
    max_num: int = 12,
    batch_size: int = 1,
    max_new_tokens: int = 300,
    do_sample: bool = True
) -> list:
    """
    执行图像描述生成推理
    
    参数:
        model: 预加载的模型
        tokenizer: 预加载的分词器
        image_path: 输入图像路径
        question: 提示问题模板
        max_num: 最大分块数
        batch_size: 批处理大小
        max_new_tokens: 生成的最大token数
        do_sample: 是否使用采样生成
        
    返回:
        模型生成的描述列表
    """
    generation_config = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
    
    # 加载并预处理图像
    pixel_values = load_image(image_path, max_num=max_num)
    pixel_values = pixel_values.to(dtype=model.dtype, device=model.device)
    
    # 构造批处理数据
    num_patches_list = [pixel_values.size(0)] * batch_size
    pixel_values_batch = torch.cat([pixel_values]*batch_size, dim=0)
    
    # 构造问题列表
    questions = [question] * len(num_patches_list)
    
    # 执行批量推理
    responses = model.batch_chat(
        tokenizer,
        pixel_values_batch,
        num_patches_list=num_patches_list,
        questions=questions,
        generation_config=generation_config
    )
    return responses

# 原有代码保持不变...

if __name__ == "__main__":
    # 加载模型
    model, tokenizer = load_model(
        model_path="OpenGVLab/InternVL2_5-8B",
        dtype=torch.bfloat16,
        quant_method="qjl"
    )
    
    # 执行推理
    responses = run_inference(
        model,
        tokenizer,
        image_path="test.png",
        question="<image>\nDescribe the image in 100 words.",
        max_num=12,
        batch_size=1
    )
    
    # 输出结果
    for response in responses:
        print(f"Assistant: {response}")