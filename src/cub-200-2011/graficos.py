import matplotlib.pyplot as plt
import numpy as np

def plot_barras(valores, nomes):
    media = np.mean(valores)

    plt.figure(figsize=(10, 6))
    cores = plt.cm.Spectral(np.linspace(0, 1, len(valores)))

    barras = plt.bar(nomes, valores, color=cores, edgecolor='black', alpha=0.8)

    plt.axhline(y=media, color='red', linestyle='--', linewidth=2, label=f'Média: {media:.2f}')

    plt.title('Quantidade de pixels removidos por método', fontsize=16)
    plt.xlabel('Métodos', fontsize=12)
    plt.ylabel('Quantidade', fontsize=12)


    for barra in barras:
        altura = barra.get_height()
        plt.text(barra.get_x() + barra.get_width()/2., altura,
                 f'{altura:.2f}',
                 ha='center', va='bottom')


    plt.legend()
    plt.tight_layout()
    plt.show()


valores = [((34.75 * 5)), ((68.5 * 5)), ((61.0 * 5)), ((81.0 * 5)), ((39.5 * 5))] 
nomes = ['Grad-CAM', 'Score-CAM', 'RISE', 'Integrated Gradients', 'Grad-CAM+Score-CAM']
plot_barras(valores, nomes)
