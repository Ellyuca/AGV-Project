
from PIL import ImageChops
import numpy as np
from utilities_d_v2 import *

''' file che contiene le funzioni che applicano i filtri '''

# produce gradiente radiale con colore esterno azzurro scuro e interno chiaro
def hudson(in_image, intensity=1, alpha=1):
    bg_image = radial_gradient(in_image.shape[:2][::-1], [166, 177, 255], [52, 33, 52])
    image = alpha_compositing(in_image.copy(), bg_image.copy(), lambda s, b: ImageChops.multiply(s, b))
    image = simple_blend(in_image, image, 0.5)
    image = adjust_brightness(image, 1.2 * alpha)
    image = adjust_contrast(image, 0.9 * alpha)
    image = adjust_saturation(image, 1.1 * alpha)
    return interpolate(in_image, image, intensity)


# crea immagine a tinta unita opaca di colore rosa e la applica usando blending soft light
def stinson(in_image, intensity=1, alpha=1):
    bg_image = create_filled_image(in_image.shape[:2][::-1], [240, 149, 128, 0.2])
    image = alpha_compositing(in_image.copy(), bg_image.copy(), soft_light_blending)
    image = adjust_contrast(image, 0.75 * alpha)
    image = adjust_saturation(image, 0.85 * alpha)
    image = adjust_brightness(image, 1.15 * alpha)
    return interpolate(in_image, image, intensity)


# prima applica bleding con immagine a tinta unita marrone scuro poi applica un altro
# blending con immagine a tinta unita marrone chiaro
def slumber(in_image, intensity=1, alpha=1):
    bg_image_1 = create_filled_image(in_image.shape[:2][::-1], [69, 41, 12, 0.4])
    image = alpha_compositing(in_image.copy(), bg_image_1.copy(), lambda s, b: ImageChops.lighter(s, b))
    bg_image_2 = create_filled_image(image.shape[:2][::-1], [125, 105, 24, 0.5])
    image = alpha_compositing(image.copy(), bg_image_2.copy(), soft_light_blending)
    image = adjust_saturation(image, 0.66 * alpha)
    image = adjust_brightness(image, 1.05 * alpha)
    return interpolate(in_image, image, intensity)


# produce gradiente lineare verticale che va dal giallo al blu e lo applica all'immagine
def perpetua(in_image, intensity=1, alpha=1):
    bg_image = linear_gradient(in_image.shape[:2][::-1], [230, 193, 61], [0, 91, 154])
    image = alpha_compositing(in_image.copy(), bg_image.copy(), soft_light_blending)
    image = simple_blend(in_image, image, 0.5 * alpha)
    return interpolate(in_image, image, intensity)


def rise(in_image, intensity=1, alpha=1):

    # applica colore rosa chiaro all'immagine iniziale
    bg_image_1 = create_filled_image(in_image.shape[:2][::-1], [236, 205, 169, 0.15])
    image_1 = alpha_compositing(in_image.copy(), bg_image_1.copy(), lambda s, b: ImageChops.multiply(s, b))
    
    # applica colore marrone scuro all'immagine iniziale
    bg_image_2 = create_filled_image(in_image.shape[:2][::-1], [50, 30, 7, 0.4])
    image_2 = alpha_compositing(in_image.copy(), bg_image_2.copy(), lambda s, b: ImageChops.multiply(s, b))

    # crea radiente gradiale internamente bianco ed esternamente nero
    gradient_image_1 = radial_gradient(in_image.shape[:2][::-1], [255, 255, 255], [0, 0, 0]).convert('L')
    
    # unisce le tre immagini precedenti 
    image_1 = composite(image_1, image_2, gradient_image_1)

    # applica colore rosa chiaro all'immagine prodotta in precedenza
    bg_image_3 = create_filled_image(in_image.shape[:2][::-1], [232, 197, 152, 0.8])
    image_2 = alpha_compositing(image_1.copy(), bg_image_3.copy(), overlay_blending)

    # crea radiente gradiale internamente bianco ed esternamente nero e lo applica all'immagine precedente
    gradient_image_2 = radial_gradient(in_image.shape[:2][::-1], [255, 255, 255], [0, 0, 0]).convert('L')
    image_3 = composite(image_2, image_1, gradient_image_2)

    image = simple_blend(image_1, image_3, 0.6)
    image = adjust_brightness(image, 1.05 * alpha)
    image = sepia_tone(image, 0.2 * alpha)
    image = adjust_contrast(image, 0.9 * alpha)
    image = adjust_saturation(image, 0.9 * alpha)
    return interpolate(in_image, image, intensity)