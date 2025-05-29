import numpy as np
from PIL import Image
import OpenEXR
import Imath


def png_to_exr(png_path, exr_path, convert_to_linear=True):
    """
    将PNG图像转换为EXR格式

    参数:
        png_path (str): 输入PNG文件路径
        exr_path (str): 输出EXR文件路径
        convert_to_linear (bool): 是否将sRGB转换为线性空间 (默认True)
    """
    # 读取PNG图像
    with Image.open(png_path) as img:
        # 转换为RGB(A)并获取数据
        if img.mode == 'RGBA':
            channels = 4
            data = np.array(img)
        else:
            channels = 3
            data = np.array(img.convert('RGB'))

    # 将8位整型转换为0-1范围的浮点数
    data = data.astype(np.float32) / 255.0

    # sRGB到线性空间的转换 (可选)
    if convert_to_linear:
        # 仅对RGB通道进行转换，忽略alpha通道
        rgb_channels = data[..., :3]
        linear_rgb = np.where(
            rgb_channels <= 0.04045,
            rgb_channels / 12.92,
            ((rgb_channels + 0.055) / 1.055) ** 2.4
        )
        data[..., :3] = linear_rgb

    # 准备EXR头部信息
    header = OpenEXR.Header(data.shape[1], data.shape[0])
    header['channels'] = {
        'R': Imath.Channel(Imath.PixelType(OpenEXR.FLOAT)),
        'G': Imath.Channel(Imath.PixelType(OpenEXR.FLOAT)),
        'B': Imath.Channel(Imath.PixelType(OpenEXR.FLOAT))
    }

    # 添加alpha通道（如果存在）
    if channels == 4:
        header['channels']['A'] = Imath.Channel(Imath.PixelType(OpenEXR.FLOAT))

    # 分离通道数据
    r = data[:, :, 0].tobytes()
    g = data[:, :, 1].tobytes()
    b = data[:, :, 2].tobytes()
    channels_data = {'R': r, 'G': g, 'B': b}

    if channels == 4:
        a = data[:, :, 3].tobytes()
        channels_data['A'] = a

    # 写入EXR文件
    exr = OpenEXR.OutputFile(exr_path, header)
    exr.writePixels(channels_data)
    exr.close()


# 使用示例
if __name__ == "__main__":
    png_to_exr("Classroom-BMFR-001.png", "Classroom-BMFR-001.exr")