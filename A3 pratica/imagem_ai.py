import cv2
from matplotlib import pyplot as plt
import numpy as np

def carregar_imagem(caminho):
    return cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)

def mostrar_imagem(imagem):
    plt.imshow(imagem, cmap='gray')
    plt.title('Imagem Cinza')
    plt.show()

def aplicar_filtro_nitidez(imagem):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(imagem, -1, kernel)

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
    cv2.waitKey(0)
    gerar_histograma(imagem)
    
    img_nitidez = aplicar_filtro_nitidez(imagem)
    mostrar_imagem(img_nitidez)
    gerar_histograma(imagem)
    cv2.waitKey(0)

def main():
    caminhos = ["imagem_ai/img1.jpg","imagem_ai/img2.jpg", "imagem_ai/img3.jpeg", "imagem_ai/img4.jpg", "imagem_ai/img5.png"]

    for caminho in caminhos:
        processar_imagem(caminho)

if __name__ == "__main__":
    main()
