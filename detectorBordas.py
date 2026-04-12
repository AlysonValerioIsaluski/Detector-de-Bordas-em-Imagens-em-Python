import cv2
from pathlib import Path as _P

def detectarCanny(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    # cv2.imwrite(f"Imagens/gray_{nome_arquivo}", img) # exporta imagens em escala de cinza
    
    altura, largura = img.shape

    # 1. Suavização (Gaussiano)

    # 2. Cálculo do Gradiente (Sobel/Prewitt)

    # 3. Supressão de Não-Máximos

    # 4. Limiação por Histerese

    return img

ALLOWED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

project_root = _P('.').resolve()
folder_path = str(project_root / 'Imagens')

p = _P(folder_path)
n_images = sum(1 for item in folder_path.iterdir() if item.is_file())
# rglob para incluir subpastas; filtra por extensão
files = sorted([f for f in p.rglob('*') if f.suffix.lower() in ALLOWED_EXTS])
results1 = []
names = []

for i, f in enumerate(files):
    res1 = detectarCanny(str(f))
    '''
    try:
    except Exception as e:
        print(f"Ignorado {f}: {e}")
        continue
    '''

    if res1 is None:
        # Se a função não conseguiu ler/processar o arquivo, avisa e pula
        print(f"Falha ao processar {f}")
        continue
    
    results1.append(res1)
    names.append(f.name)

for i in range(len(results1)):
    if results1[i] is not None:
        # Exporta imagens
        cv2.imwrite(f"Resultados/equaliza_{names[i]}", results1[i])
        print(f"Imagem {names[i]} exportada com sucesso!")