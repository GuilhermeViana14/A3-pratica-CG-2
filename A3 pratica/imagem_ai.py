import cv2
from matplotlib import pyplot as plt
import numpy as np
import time

def carregar_imagem(caminho):
    return cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)

def mostrar_imagem(imagem):
    plt.imshow(imagem, cmap='gray')
    plt.title('Imagem Cinza')
    plt.show()
    

def aplicar_filtro_nitidez(imagem):
    #Este kernel tem um valor de 5 e valores negativos ao seu redor que ajuda a realçar as diferenças de intensidade entre pixels
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(imagem, -1, kernel)


def aplicar_filtro_suavizacao(imagem):
    # Aplicando filtro de suavização com um kernel de 5x5
    img_suavizada = cv2.blur(imagem, (6, 6))
    return img_suavizada

def aplicar_filtro_equalizacao_histograma(imagem):
    # Este filtro serve para melhorar o contraste e realçar os detalhes na imagem
    img_equalizada = cv2.equalizeHist(imagem)
    return img_equalizada

def propriedades_imagem(imagem):
    altura, largura = imagem.shape
    print(f"Dimensões: {largura}x{altura}")
    print(f"Valores de cinza (primeiras 5x5 pixels):\n{imagem[:5,:5]}")
    print(f"Valor mínimo de cinza: {imagem.min()}")
    print(f"Valor máximo de cinza: {imagem.max()}")

def percentuais_valores_cinza(imagem):
    hist = cv2.calcHist([imagem], [0], None, [256], [0,256])
    total_pixels = imagem.size

    print("\nPercentuais de cada valor de cinza:")
    for valor, quantidade in enumerate(hist):
        percentual = (quantidade / total_pixels) * 100
        if percentual[0] > 0:
            print(f"Valor {valor}: {percentual[0]:.2f}%")

def gerar_histograma(imagem):
    plt.hist(imagem.ravel(), 256, [0, 256])
    plt.title('Histograma')
    plt.xlabel('Valores de Cinza')
    plt.ylabel('Frequência')
    plt.show()
    

def processar_imagem(caminho):
    print(f"\nProcessando imagem: {caminho}")
    imagem = carregar_imagem(caminho)

    mostrar_imagem(imagem)
    propriedades_imagem(imagem)
    gerar_histograma(imagem)
    percentuais_valores_cinza(imagem)
    cv2.waitKey(0)
    
    img_nitidez = aplicar_filtro_nitidez(imagem)
    mostrar_imagem(img_nitidez)
    gerar_histograma(img_nitidez)
    cv2.waitKey(0)

    img_suavizada = aplicar_filtro_suavizacao(img_nitidez)
    mostrar_imagem(img_suavizada)
    gerar_histograma(img_suavizada)
    cv2.waitKey(0)
    
    img_equalizada = aplicar_filtro_equalizacao_histograma(img_suavizada)
    mostrar_imagem(img_equalizada)
    gerar_histograma(img_equalizada)
    cv2.waitKey(0)

def main():
    caminhos = ["A3-pratica-CG-2-main/A3 pratica/imagem_ai/img1.jpg","A3-pratica-CG-2-main/A3 pratica/imagem_ai/img2.jpg", "A3-pratica-CG-2-main/A3 pratica/imagem_ai/img3.jpeg", "A3-pratica-CG-2-main/A3 pratica/imagem_ai/img4.jpg", "A3-pratica-CG-2-main/A3 pratica/imagem_ai/img5.png"]

    for caminho in caminhos:
        processar_imagem(caminho)
        

if __name__ == "__main__":
    main()
