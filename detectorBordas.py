import cv2
import numpy as np
import math
from pathlib import Path as _P

def detectarPriwitt(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    altura, largura = img.shape
    imgPriwitt = np.zeros((altura, largura), dtype=np.float32)

    for i in range(altura):
        for j in range(largura):
            imgPriwitt[i, j] = filtroPriwitt(img, i, j)

    print(f"Detecção Priwitt finalizada")
    
    return imgPriwitt

def detectarFreiChen(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    altura, largura = img.shape
    imgFreiChen = np.zeros((altura, largura), dtype=np.float32)

    for i in range(altura):
        for j in range(largura):
            imgFreiChen[i, j] = filtroFreiChen(img, i, j)

    print(f"Detecção Frei Chen finalizada")
    
    return imgFreiChen

def detectarCanny(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    altura, largura = img.shape
    imgBorrada = np.zeros((altura, largura), dtype=np.uint8)

    ##### 1. Suavização (Gaussiano)
    for i in range(altura):
        for j in range(largura):
            imgBorrada[i, j] = filtroGaussiano(img, i, j)
    
    print(f"Passo 1 finalizado - Suavização")

    ##### 2. Cálculo do Gradiente (Sobel)
    imgMag = np.zeros((altura, largura), dtype=np.uint8)
    imgAng = np.zeros((altura, largura), dtype=np.float32)

    for i in range(altura):
        for j in range(largura):
            mag, ang = filtroSobel(imgBorrada, i, j)
            imgMag[i, j] = mag
            imgAng[i, j] = ang

    print(f"Passo 2 finalizado - Cálculo do Gradiente")

    ##### 3. Supressão de Não-Máximos
    imgSupressao = np.zeros((altura, largura), dtype=np.uint8)
    for i in range(altura):
        for j in range(largura):
            imgSupressao[i, j] = supressaoNaoMaximos(imgMag, imgAng, i, j)

    print(f"Passo 3 finalizado - Supressão de Não-Máximos")

    ##### 4. Limiação por Histerese
    limiar_baixo = 35
    limiar_alto = 105
    
    imgLimiar = np.zeros((altura, largura), dtype=np.uint8)
    for i in range(altura):
        for j in range(largura):
            imgLimiar[i, j] = classificarBorda(imgSupressao, i, j, limiar_baixo, limiar_alto)

    imgHisterese = np.zeros((altura, largura), dtype=np.uint8)
    for i in range(altura):
        for j in range(largura):
            imgHisterese[i, j] = rastrearBorda(imgLimiar, i, j)

    print(f"Passo 4 finalizado - Limiação por Histerese")
    
    return img, imgBorrada, imgMag, imgSupressao, imgHisterese

def detectarCannyCor(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    
    b, g, r = cv2.split(img)
    altura, largura = b.shape
    canais = [b, g, r]
    imgFinal = np.zeros((altura, largura), dtype=np.uint8)
    
    for canal in canais:
        ##### 1. Suavização (Gaussiano)
        imgBorrada = np.zeros((altura, largura), dtype=np.uint8)

        for i in range(altura):
            for j in range(largura):
                imgBorrada[i, j] = filtroGaussiano(canal, i, j)
        
        print(f"Passo 1 finalizado - Suavização")

        ##### 2. Cálculo do Gradiente (Sobel)
        imgMag = np.zeros((altura, largura), dtype=np.uint8)
        imgAng = np.zeros((altura, largura), dtype=np.float32)

        for i in range(altura):
            for j in range(largura):
                mag, ang = filtroSobel(imgBorrada, i, j)
                imgMag[i, j] = mag
                imgAng[i, j] = ang

        print(f"Passo 2 finalizado - Cálculo do Gradiente")

        ##### 3. Supressão de Não-Máximos
        imgSupressao = np.zeros((altura, largura), dtype=np.uint8)
        for i in range(altura):
            for j in range(largura):
                imgSupressao[i, j] = supressaoNaoMaximos(imgMag, imgAng, i, j)

        print(f"Passo 3 finalizado - Supressão de Não-Máximos")

        ##### 4. Limiação por Histerese
        limiar_baixo = 35
        limiar_alto = 105
        
        imgLimiar = np.zeros((altura, largura), dtype=np.uint8)
        for i in range(altura):
            for j in range(largura):
                imgLimiar[i, j] = classificarBorda(imgSupressao, i, j, limiar_baixo, limiar_alto)

        for i in range(altura):
            for j in range(largura):
                valor_borda = rastrearBorda(imgLimiar, i, j)
                if valor_borda > 255:
                    valor_borda = 255
                imgFinal[i, j] += valor_borda

        print(f"Passo 4 finalizado - Limiação por Histerese")

    return imgFinal

def filtroPriwitt(img, i, j):
    kernel_x = [
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ]
    
    kernel_y = [
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ]
    
    altura, largura = img.shape
    
    if i < 1 or i >= altura - 1 or j < 1 or j >= largura - 1:
        return img[i, j]
        
    gx = 0
    gy = 0
    
    for u in range(-1, 2):
        for v in range(-1, 2):
            pixel = img[i + u, j + v]
            gx += pixel * kernel_x[u + 1][v + 1]
            gy += pixel * kernel_y[u + 1][v + 1]
            
    magnitude = int((gx**2 + gy**2)**0.5)
    
    if magnitude > 255:
        return 255
    elif magnitude < 0:
        return 0
        
    return magnitude

def filtroFreiChen(img, i, j):
    kernel_x = [
        [-1, 0, 1],
        [-1.414, 0, 1.414],
        [-1, 0, 1]
    ]
    
    kernel_y = [
        [-1, -1.414, -1],
        [0, 0, 0],
        [1, 1.414, 1]
    ]
    
    altura, largura = img.shape
    
    if i < 1 or i >= altura - 1 or j < 1 or j >= largura - 1:
        return img[i, j]
        
    gx = 0
    gy = 0
    
    for u in range(-1, 2):
        for v in range(-1, 2):
            pixel = img[i + u, j + v]
            gx += pixel * kernel_x[u + 1][v + 1]
            gy += pixel * kernel_y[u + 1][v + 1]
            
    magnitude = int((gx**2 + gy**2)**0.5)
    
    if magnitude > 255:
        return 255
    elif magnitude < 0:
        return 0
        
    return magnitude

def filtroGaussiano(img, i, j):
    kernel = [
        [1, 4, 7, 4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1, 4, 7, 4, 1]
    ]
    
    altura, largura = img.shape
    
    if i < 2 or i >= altura - 2 or j < 2 or j >= largura - 2:
        return img[i, j]
        
    soma = 0
    for u in range(-2, 3):
        for v in range(-2, 3):
            soma += img[i + u, j + v] * kernel[u + 2][v + 2]
            
    return soma // 273

def filtroSobel(img, i, j):
    kernel_x = [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]
    
    kernel_y = [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ]
    
    altura, largura = img.shape
    
    if i < 1 or i >= altura - 1 or j < 1 or j >= largura - 1:
        return img[i, j], 0
        
    gx = 0
    gy = 0
    
    for u in range(-1, 2):
        for v in range(-1, 2):
            pixel = img[i + u, j + v]
            gx += pixel * kernel_x[u + 1][v + 1]
            gy += pixel * kernel_y[u + 1][v + 1]
            
    magnitude = int((gx**2 + gy**2)**0.5)
    
    angulo = math.atan2(gy, gx) * 180 / math.pi
    if angulo < 0:
        angulo += 180
        
    if magnitude > 255:
        magnitude = 255
    elif magnitude < 0:
        magnitude = 0
        
    return magnitude, angulo

def supressaoNaoMaximos(mag, ang, i, j):
    altura, largura = mag.shape
    
    if i < 1 or i >= altura - 1 or j < 1 or j >= largura - 1:
        return 0
        
    q = 255
    r = 255
    angulo = ang[i, j]
    
    if (0 <= angulo < 22.5) or (157.5 <= angulo <= 180):
        q = mag[i, j+1]
        r = mag[i, j-1]
    elif (22.5 <= angulo < 67.5):
        q = mag[i+1, j-1]
        r = mag[i-1, j+1]
    elif (67.5 <= angulo < 112.5):
        q = mag[i+1, j]
        r = mag[i-1, j]
    elif (112.5 <= angulo < 157.5):
        q = mag[i-1, j-1]
        r = mag[i+1, j+1]
        
    if (mag[i, j] >= q) and (mag[i, j] >= r):
        return mag[i, j]
    else:
        return 0
    
def classificarBorda(img, i, j, limiar_baixo, limiar_alto):
    pixel = img[i, j]
    if pixel >= limiar_alto:
        return 255
    elif pixel >= limiar_baixo:
        return 128
    else:
        return 0

def rastrearBorda(img, i, j):
    altura, largura = img.shape
    pixel = img[i, j]
    
    if pixel == 255:
        return 255
    elif pixel == 128:
        if i < 1 or i >= altura - 1 or j < 1 or j >= largura - 1:
            return 0
            
        for u in range(-1, 2):
            for v in range(-1, 2):
                if img[i + u, j + v] == 255:
                    return 255
        return 0
    else:
        return 0

ALLOWED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp', '.avif'}

project_root = _P('.').resolve()
folder_path = str(project_root / 'Imagens')

p = _P(folder_path)
n_images = sum(1 for item in _P(folder_path).iterdir() if item.is_file())
# rglob para incluir subpastas; filtra por extensão
files = sorted([f for f in p.rglob('*') if f.suffix.lower() in ALLOWED_EXTS])
results = []
names = []

for i, f in enumerate(files):
    print(f"Processando Imagem {i+1}: {f.name}")
    
    try:
        img, imgBorrada, imgMag, imgSupressao, imgHisterese = detectarCanny(str(f))
        prewitt = detectarPriwitt(str(f))
        freiChen = detectarFreiChen(str(f))
        cannyCor = detectarCannyCor(str(f))
    except Exception as e:
        print(f"Ignorado {f}: {e}")
        continue

    if img is None or imgHisterese is None:
        # Se a função não conseguiu ler/processar o arquivo, avisa e pula
        print(f"Falha ao processar {f}")
        continue

    names.append(f.name)
    imgResults = [img, imgBorrada, imgMag, imgSupressao, imgHisterese, prewitt, freiChen, cannyCor]
    results.append(imgResults)

for i in range(n_images):
    if results[i] is not None:
        # Exporta imagens
        cv2.imwrite(f"Resultados/passo0_cinza_{names[i]}", results[i][0])
        cv2.imwrite(f"Resultados/passo1_gauss_{names[i]}", results[i][1])
        cv2.imwrite(f"Resultados/passo2_sobel_{names[i]}", results[i][2])
        cv2.imwrite(f"Resultados/passo3_supressao_{names[i]}", results[i][3])
        cv2.imwrite(f"Resultados/passo4_histerese_{names[i]}", results[i][4])
        cv2.imwrite(f"Resultados/prewitt_{names[i]}", results[i][5])
        cv2.imwrite(f"Resultados/freichen_{names[i]}", results[i][6])
        cv2.imwrite(f"Resultados/canny_cor_{names[i]}", results[i][7])
        print(f"Resultados da imagem {names[i]} exportados com sucesso!")