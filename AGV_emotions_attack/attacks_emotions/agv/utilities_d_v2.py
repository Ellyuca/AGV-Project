from PIL import Image, ImageEnhance, ImageChops
import numpy as np
import math

'''
file che contiene le trasformazioni 
'''


# converte l'immagine da array di numpy a immagine di pillow
def array_to_pil(image):
    if type(image) == np.ndarray:
        if image.dtype == np.float32:
            image = (image*255).astype(np.uint8)
        image = Image.fromarray(image)
    return image


# converte l'immagine da oggetto di pillow ad array di numpy
# con mode == "float" i valori sono compresi tra 0 e 1
# con mode == "int" i valori sono compresi tra 0 e 255
def pil_to_array(image, mode):
    if type(image) != np.ndarray:
        image = np.array(image).astype(np.uint8)
        if mode == "float":
            image = (image/255).astype(np.float32)
    return image


# permette di passare dalla rappresentazione dell'immagine con
# valori dei pixel tra 0 e 255 alla rappresentazione con valori
# tra 0 e 1 (e viceversa)
def change_array_type(image, mode):
    if mode == "int" and image.dtype != np.uint8:
        image = (image*255).astype(np.uint8)
    if mode == "float" and image.dtype != np.float32:
        image = (image/255).astype(np.float32)
    return image


# comprime i valori dei pixel tra 0 e 255
def clip(image):
    return np.clip(image, 0, 255)


# comprime i valori dei pixel tra 0 e 1
def norm_clip(image):
    return np.clip(image, 0.0, 1.0).astype(np.float32)

# applica interpolazione 
def interpolate(org_image, mod_image, intensity):
    return norm_clip((mod_image * intensity + org_image * (1 - intensity)))


# regola il contrasto dell'immagine
def adjust_contrast(image, alpha):
    image = array_to_pil(image)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(alpha)
    return pil_to_array(image, "float")


# regola la luminosità dell'immgine
def adjust_brightness(image, alpha):
    image = array_to_pil(image)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(alpha)
    return pil_to_array(image, "float")


# regola la saturazione dell'immagine
def adjust_saturation(image, alpha):
    image = array_to_pil(image)
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(alpha)
    return pil_to_array(image, "float")


'''
Applica filtro color seppia all'immagine
Forumla reperita da: https://www.w3.org/TR/filter-effects-1/#sepiaEquivalent
new_red = red*(0.393 + 0.607*(1 - alpha)) + green*(0.769 - 0.769*(1 - alpha)) + blue*(0.189 - 0.189 * (1 - alpha))
new_green = red*(0.349 + 0.349*(1 - alpha)) + green*(0.686 - 0.314*(1 - alpha)) + blue*(0.168 - 0.168 * (1 - alpha))
new_blue = red*(0.272 + 0.272*(1 - alpha)) + green*(0.534 - 0.534*(1 - alpha)) + blue*(0.131 - 0.869 * (1 - alpha))
'''
def sepia_tone(in_image, alpha):
    image = array_to_pil(in_image)
    alpha = 1 - alpha
    matrix = [
        0.393 + 0.607 * alpha, 0.769 - 0.769 * alpha, 0.189 - 0.189 * alpha, 0,
        0.349 - 0.349 * alpha, 0.686 + 0.314 * alpha, 0.168 - 0.168 * alpha, 0,
        0.272 - 0.272 * alpha, 0.534 - 0.534 * alpha, 0.131 + 0.869 * alpha, 0]
    image = image.convert('RGB', matrix)
    return pil_to_array(image, "float")


# crea un immagine dei colori e dimensioni passati come parametri
def create_filled_image(size, colors):
    if len(colors) == 4:
        colors[3] = int(colors[3]*255)
        return Image.new("RGBA", size, tuple(colors))
    if len(colors) == 3:
        return Image.new("RGB", size, tuple(colors))


# crea gradiente gradiale dei colori passati come parametri
def radial_gradient(size, inner_color, outer_color):
    # definisce il colore esterno
    outer_image = create_filled_image(size, outer_color)
    # definisce il clore interno
    inner_image = create_filled_image(size, inner_color)
    w, h = size
    x = np.linspace(-w/2, w/2, w)
    y = np.linspace(-h/2, h/2, h)[:, None]  # shape da (h, 1) a (1, h)
    # calcola la distanza massima dal centro
    d_max = math.sqrt((w/2)**2 + (h/2)**2)
    # calcola la distanza dal centro e la normalizza rispetto alla distanza massima
    mask = np.sqrt(x**2 + y**2) / d_max
    # mask = 2 * (mask - 0.5)
    # mask = 1 - mask
    mask *= 255
    mask = mask.clip(0, 255)
    mask = Image.fromarray(np.uint8(mask.round()))
    return Image.composite(outer_image, inner_image, mask)


# crea gradiente lineare dei colori passati come parametri
def linear_gradient(size, start_color, end_color):
    # crea immagine del primo colore
    start_image = create_filled_image(size, start_color)
    # crea immagine del secondo colore
    end_image = create_filled_image(size, end_color)
    w, h = size
    start, end = (0, 255)
    row = np.linspace(start, end, num=h, dtype=(np.uint8))
    mask = np.tile(row, (w, 1)).T
    mask = Image.fromarray(mask)
    return Image.composite(start_image, end_image, mask)

# converte immagine da RGBA ad RGB
def alpha_to_rgb(alpha_image):
    img = alpha_image.convert('RGB')
    return img


'''
Foruma reperita da: https://www.w3.org/TR/compositing-1/#blending
Cr =  (αs - αs x αb) x Cs + αs x αb x B(Cb, Cs) + (αb - αs x αb) x Cb
Dove:
 - Cr è l'immagine risultante
 - B è la forula per il blending
 - Cs è l'immagine originale
 - Cb è il colore da applicare a Cs
 - αs è il valore del canale alpha di Cs
 - αb è il valore del canale alpha di Cb
'''
def alpha_compositing(main_image, bg_image, blending_function):
    main_image = array_to_pil(main_image)
    # aggiunge il canale alpha alle immagini se non lo hanno
    if main_image.mode != 'RGBA':
        main_image.putalpha(Image.new('L', main_image.size, 255))
    if bg_image.mode != 'RGBA':
        bg_image.putalpha(Image.new('L', bg_image.size, 255))

    # estrae il canale alpha dalle immagini
    a_s = main_image.split()[3]
    C_s = main_image.convert('RGB')
    a_b = bg_image.split()[3]
    C_b = bg_image.convert('RGB')

    img_blend = blending_function(C_s, C_b)  # B(Cb, Cs)
    alpha_mul = ImageChops.multiply(a_s, a_b)  # αs x αb
    alpha_sub_s = ImageChops.subtract(a_s, alpha_mul)  # αs - αs x αb
    alpha_sub_b = ImageChops.subtract(a_b, alpha_mul)  # αb - αs x αb
    C_1 = ImageChops.multiply(alpha_to_rgb(
        alpha_sub_s), C_s)  # (αs - αs x αb) x Cs
    C_2 = ImageChops.multiply(alpha_to_rgb(
        alpha_mul), img_blend)  # αs x αb x B(Cb, Cs)
    C_3 = ImageChops.multiply(alpha_to_rgb(
        alpha_sub_b), C_b)  # (αb - αs x αb) x Cb
    C = ImageChops.add(C_1, C_2)  # (αs - αs x αb) x Cs + αs x αb x B(Cb, Cs)
    # (αs - αs x αb) x Cs + αs x αb x B(Cb, Cs) + (αb - αs x αb) x Cb
    C = ImageChops.add(C, C_3)
    return pil_to_array(C, "float")


'''
Formula reperita da: https://www.w3.org/TR/compositing-1/#blendingsoftlight
if(Cs <= 0.5)
    B(Cb, Cs) = Cb - (1 - 2 x Cs) x Cb x (1 - Cb)
else
    B(Cb, Cs) = Cb + (2 x Cs - 1) x (D(Cb) - Cb)

dove:
 if(Cb <= 0.25)
    D(Cb) = ((16 * Cb - 12) x Cb + 4) x Cb
 else
    D(Cb) = sqrt(Cb)
'''
def soft_light_blending(main_image, bg_image):

    C_s = pil_to_array(bg_image, "int").astype(np.float32)
    C_b = pil_to_array(main_image, "int").astype(np.float32)
    C_b_t = C_b / 255

    # calcola D(Cb)
    D = np.where(C_b_t > 0.25, C_b_t ** 0.5,
                 ((16 * C_b_t - 12) * C_b_t + 4) * C_b_t)
    D = D * 255
    D = array_to_pil(D.astype(np.uint8))

    O_1 = clip(255 * (1 - 2 * C_s / 255)).astype(np.uint8)  # 1 - 2 x Cs
    O_1 = array_to_pil(O_1)
    O_2 = clip((C_b * (1 - C_b / 255)).astype(np.uint8))  # Cb x (1 - Cb)
    O_2 = array_to_pil(O_2)
    C_1 = ImageChops.subtract(array_to_pil(C_b.astype(np.uint8)), ImageChops.multiply(
        O_1, O_2))  # Cb - (1 - 2 x Cs) x Cb x (1 - Cb)
    C_1 = pil_to_array(C_1, "int")

    O_1 = clip(2 * C_s - 255).astype(np.uint8)  # 2 x Cs - 1
    O_1 = array_to_pil(O_1)
    O_2 = ImageChops.subtract(D, array_to_pil(
        C_b.astype(np.uint8)))  # D(Cb) - Cb
    O_3 = ImageChops.multiply(O_1, O_2)
    # Cb + (2 x Cs - 1) x (D(Cb) - Cb)
    C_2 = ImageChops.add(array_to_pil(C_b.astype(np.uint8)), O_3)
    C_2 = pil_to_array(C_2, "int")

    B = np.where(C_s <= 128, C_1, C_2)
    return array_to_pil(B)


'''
Formula reperita da: https://www.w3.org/TR/compositing-1/#blendinghardlight
if(Cs <= 0.5)
    B(Cb, Cs) = Multiply(Cb, 2 x Cs)
else
    B(Cb, Cs) = Screen(Cb, 2 x Cs -1)
'''
def hard_light_blending(main_image, bg_image):
    C_s = pil_to_array(bg_image, "int").astype(np.float32)
    C_b = pil_to_array(main_image, "int").astype(np.float32)
    O_1 = clip(2 * C_s).astype(np.uint8)  # 2 x Cs
    O_1 = array_to_pil(O_1)
    B_1 = ImageChops.multiply(array_to_pil(
        C_b.astype(np.uint8)), O_1)  # Multiply(Cb, 2 x Cs)
    B_1 = pil_to_array(B_1, "int")
    O_2 = clip(2 * C_s - 255).astype(np.uint8)  # 2 x Cs -1
    O_2 = array_to_pil(O_2)
    B_2 = ImageChops.screen(array_to_pil(
        C_b.astype(np.uint8)), O_2)  # Screen(Cb, 2 x Cs -1)
    B_2 = pil_to_array(B_2, "int")
    B = np.where(C_s <= 128, B_1, B_2)
    B = array_to_pil(B)
    return B


'''
Formula reperita da: https://www.w3.org/TR/compositing-1/#blendingoverlay
B(Cb, Cs) = HardLight(Cs, Cb)
'''
def overlay_blending(main_image, bg_image):
    return hard_light_blending(bg_image, main_image)


# effettua blending tra le due immagini passate
def simple_blend(in_image, bg_image, alpha):
    in_image = array_to_pil(in_image)
    bg_image = array_to_pil(bg_image)
    image = Image.blend(in_image, bg_image, alpha)
    return pil_to_array(image, "float")


# unisce tra di loro le tre immagini passate
def composite(image_1, image_2, image_3):
    image_1 = array_to_pil(image_1)
    image_2 = array_to_pil(image_2)
    image_3 = array_to_pil(image_3)
    image = Image.composite(image_1, image_2, image_3)
    return pil_to_array(image, "float")
