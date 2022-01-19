from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from utilities_d_v2 import *
from io import BytesIO
import cv2 as cv

'''
file che contiene le funzioni che applicano all'immagine le trasformazioni
elementari come se fossero filtri, quindi sono scalabili rispetto ad alpha
e intensity
'''

# modifica del contrasto
def Contrast(in_image, intensity, alpha):
    image = array_to_pil(in_image)
    enhancer = ImageEnhance.Contrast(image)
    enhanced = pil_to_array(enhancer.enhance(alpha), "float")
    return interpolate(in_image, enhanced, intensity)


# modifica della luminosità
def Brightness(in_image, intensity, alpha):
    image = array_to_pil(in_image)
    enhancer = ImageEnhance.Brightness(image)
    enhanced = pil_to_array(enhancer.enhance(alpha), "float")
    return interpolate(in_image, enhanced, intensity)


# modifica della saturazione
def Saturation(in_image, intensity, alpha):
    image = array_to_pil(in_image)
    enhancer = ImageEnhance.Color(image)
    enhanced = pil_to_array(enhancer.enhance(alpha), "float")
    return interpolate(in_image, enhanced, intensity)


# modifica della nitidezza
def Sharpness(in_image, intensity, alpha):
    image = array_to_pil(in_image)
    enhancer = ImageEnhance.Sharpness(image)
    enhanced = pil_to_array(enhancer.enhance(alpha), "float")
    return interpolate(in_image, enhanced, intensity)


# correzzione gamma
def Gamma(in_image, intensity, alpha):
    image = change_array_type(in_image, "int")
    gamma = 1.0 / alpha
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    result = change_array_type(cv.LUT(image, table), "float")
    return interpolate(in_image, result, intensity)

# applicazione effetto seppia
def Sepia(in_image, intensity, alpha):
    image = array_to_pil(in_image)
    alpha = 1 - alpha
    matrix = [
        0.393 + 0.607 * alpha, 0.769 - 0.769 * alpha, 0.189 - 0.189 * alpha, 0,
        0.349 - 0.349 * alpha, 0.686 + 0.314 * alpha, 0.168 - 0.168 * alpha, 0,
        0.272 - 0.272 * alpha, 0.534 - 0.534 * alpha, 0.131 + 0.869 * alpha, 0]
    image = pil_to_array(image.convert('RGB', matrix), "float")
    return interpolate(in_image, image, intensity)


# # aggiunta rumore
# def noise(in_image, intensity, alpha):
#     image = pil_to_array(in_image, "float")
#     noise = np.random.uniform(0.0, 1.0, image.shape)
#     noisy = norm_clip(image + noise*alpha)
#     return interpolate(image, noisy, intensity)

# compressione jpeg
def Jpeg(in_image, intensity, alpha):
    image = array_to_pil(in_image)
    buffer = BytesIO()
    image.save(buffer, "JPEG", quality=int(alpha))
    buffer.seek(0)
    compressed = pil_to_array(Image.open(buffer), "float")
    return interpolate(in_image, compressed, intensity)


# applicazione effetto blur
def Blur(in_image, intensity, alpha):
    image = array_to_pil(in_image)
    result = pil_to_array(image.filter(ImageFilter.BoxBlur(alpha)), "float")
    return interpolate(in_image, result, intensity)