"""
Projeto ONDES (S6) - Simulação de Múltiplos Microplásticos (Densidade Progressiva)
Método: Propagação do Espectro Angular (ASM) com Superposição de Fases

Autor: Engenharia da Computação
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift

def simular_campo_multiplo(X, Y, N, L, num_particulas, k0_meio, n_plastico, n_meio):
    """
    Gera o campo elétrico complexo inicial contendo N partículas aleatórias.
    """
    # Inicializa o campo como uma onda plana uniforme (amplitude 1, fase 0)
    mask_amplitude = np.ones((N, N))
    phase_p = np.zeros((N, N))
    
    # Fixar o seed para que o gráfico seja reprodutível a cada execução
    np.random.seed(28)
    
    # Definição do tamanho médio dos microplásticos (ex: entre 30 e 60 micrometros)
    tamanhos = np.random.uniform(30e-6, 60e-6, num_particulas)
    
    # Distribuição espacial das partículas na lâmina (evitando as bordas extremas)
    pos_x = np.random.uniform(-L/3, L/3, num_particulas)
    pos_y = np.random.uniform(-L/3, L/3, num_particulas)
    
    # Caso especial: se for apenas 1 partícula, força a ficar no centro para fins didáticos
    if num_particulas == 1:
        pos_x[0], pos_y[0] = 0.0, 0.0
        tamanhos[0] = 50e-6

    # Superposição de cada partícula no campo elétrico
    for i in range(num_particulas):
        d_p = tamanhos[i]
        # Distância de cada pixel ao centro da partícula i
        dist_quadrado = (X - pos_x[i])**2 + (Y - pos_y[i])**2
        mask_circulo = dist_quadrado < (d_p / 2)**2
        
        # Aplica atenuação de amplitude (95% de transmissão no plástico)
        mask_amplitude[mask_circulo] = 0.95
        
        # Calcula o atraso de fase induzido por esta partícula específica
        delta_phi = k0_meio * d_p * (n_plastico - n_meio)
        phase_p[mask_circulo] = delta_phi

    # Campo complexo resultante da combinação de todas as partículas
    return mask_amplitude * np.exp(1j * phase_p)

def executar_simulacao_quadrante():
    # -------------------------------------------------------------------------
    # 1. CONFIGURAÇÕES ÓPTICAS GERAIS
    # -------------------------------------------------------------------------
    lambda_vac = 532e-9  # Laser Verde
    n_meio = 1.333       # Água
    n_plastico = 1.59    # Poliestireno
    
    lambda_efetivo = lambda_vac / n_meio
    k0_meio = 2 * np.pi / lambda_efetivo

    # Configuração do Sensor
    N = 1024
    pixel_size = 4.5e-6  # Pixels ligeiramente menores para melhorar a resolução
    L = N * pixel_size
    z_prop = 0.04        # 4 centímetros de distância até o sensor

    # Grade de coordenadas espaciais
    x = np.linspace(-L/2, L/2, N)
    y = np.linspace(-L/2, L/2, N)
    X, Y = np.meshgrid(x, y)

    # Grade de frequências espaciais para o ASM
    df = 1 / L
    fx = np.arange(-N/2, N/2) * df
    fy = np.arange(-N/2, N/2) * df
    FX, FY = np.meshgrid(fx, fy)

    # Função de Transferência do Espaço Livre (H)
    term_radical = k0_meio**2 - (2*np.pi*FX)**2 - (2*np.pi*FY)**2
    mask_evan = np.where(term_radical >= 0, 1.0, 0.0)
    kz = np.sqrt(np.abs(term_radical)) * mask_evan
    H = np.exp(1j * kz * z_prop)

    # Casos de teste para o quadrante (número progressivo de plásticos)
    casos = [1, 5, 20, 60]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    axes = axes.ravel() # Planificar a matriz de eixos para iteração linear

    print("Propagando frentes de onda numéricas...")

    for idx, num_p in enumerate(casos):
        # 2. Gerar o campo inicial para a densidade atual
        U_0 = simular_campo_multiplo(X, Y, N, L, num_p, k0_meio, n_plastico, n_meio)
        
        # 3. Propagação pelo Método do Espectro Angular (FFT2)
        A_0 = fft2(U_0)
        A_0_shifted = fftshift(A_0)
        
        A_z = A_0_shifted * H
        U_z = ifft2(fftshift(A_z))
        
        # Intensidade capturada pelo sensor
        holograma_ideal = np.abs(U_z)**2
        
        # Adicionar ruído experimental gaussiano (5%)
        holograma_real = holograma_ideal + np.random.normal(0, 0.05, holograma_ideal.shape)
        holograma_real = np.maximum(holograma_real, 0.0)
        
        # 4. Plotar no quadrante correspondente
        ax = axes[idx]
        im = ax.imshow(holograma_real, extent=[-L/2*1e3, L/2*1e3, -L/2*1e3, L/2*1e3], 
                       cmap='gray', vmax=holograma_real.max()*0.85)
        
        ax.set_title(f"Cenário {idx+1}: {num_p} Partícula(s)", fontsize=12)
        ax.set_xlabel("X (mm)", fontsize=9)
        ax.set_ylabel("Y (mm)", fontsize=9)
        
        # Adicionar uma barra de intensidade individual discreta
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label('Intensidade Razoável', fontsize=8)

    plt.suptitle(f"Simulação de Difração In-line (ASM) a z = {z_prop*100:.1f} cm\nEfeito da Densidade de Microplásticos no Padrão de Interferência", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    executar_simulacao_quadrante()