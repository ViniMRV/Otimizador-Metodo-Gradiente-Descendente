import numpy as np
import matplotlib.pyplot as plt
from gradient import OptimizationMethods 
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_trajectory(path_3d, title):
    """
    Plota a trajetória do ponto (x,y,z) no espaço 3D.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Converte lista de arrays para array numpy
    path = np.array(path_3d)
    xs = path[:, 0]
    ys = path[:, 1]
    zs = path[:, 2]
    
    # Plota a linha da trajetória
    ax.plot(xs, ys, zs, 'b.-', label='Trajetória IPS 3D', markersize=4, alpha=0.8)
    
    # Marca o início (vermelho) e o fim (verde)
    ax.scatter(xs[0], ys[0], zs[0], color='red', s=50, label='Início')
    ax.scatter(xs[-1], ys[-1], zs[-1], color='green', s=100, marker='*', label='Fim')
    
    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    # 1. Salvar Primeiro
    filename = 'trajetoria_3d.png'
    plt.savefig(filename)
    print(f"[INFO] Gráfico 3D salvo como '{filename}'.")
    
    # 2. Mostrar na Tela (Substitui o plt.close())
    plt.show() 

def plot_comparison(opt, path_const, path_ips, title):
    """
    Gera dois gráficos: Convergência e Trajetória.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Gráfico 1: Convergência ---
    vals_const = [opt.rosenbrock_2d(p) for p in path_const]
    vals_ips = [opt.rosenbrock_2d(p) for p in path_ips]
    
    ax1.plot(vals_const, label='Passo Constante', color='red')
    ax1.plot(vals_ips, label='IPS (Adaptativo)', color='blue', linestyle='--')
    ax1.set_yscale('log') 
    ax1.set_xlabel('Iterações')
    ax1.set_ylabel('Valor de f(x) (Log Scale)')
    ax1.set_title('Velocidade de Convergência')
    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.5)

    # --- Gráfico 2: Trajetória nas Curvas de Nível ---
    x_range = np.linspace(-2, 2, 400)
    y_range = np.linspace(-1, 3, 400)
    X, Y = np.meshgrid(x_range, y_range)
    Z = 100 * (Y - X**2)**2 + (X - 1)**2 
    
    ax2.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='gray', alpha=0.4)
    
    path_c = np.array(path_const)
    path_i = np.array(path_ips)
    
    ax2.plot(path_c[:,0], path_c[:,1], 'r.-', label='Passo Constante', markersize=3, alpha=0.6)
    ax2.plot(path_i[:,0], path_i[:,1], 'b.-', label='IPS', markersize=3)
    ax2.plot(1, 1, 'g*', markersize=15, label='Mínimo Global (1,1)')
    
    ax2.set_title('Trajetória da Otimização')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.legend()
    
    plt.suptitle(title)
    
    # 1. Salvar Primeiro
    filename = 'resultado_grafico.png'
    plt.savefig(filename)
    print(f"\n[INFO] Gráfico salvo como '{filename}'.")
    
    # 2. Mostrar na Tela (Substitui o plt.close())
    plt.show()

def run_tests():
    opt = OptimizationMethods()
    
    # ==========================================
    # CONFIGURAÇÃO DOS TESTES
    # ==========================================
    start_pos_2d = [-1.2, 1.0] 
    MAX_ITER = 10000  
    
    print("--- INICIANDO BATERIA DE TESTES ---\n")

    # ---------------------------------------------------------
    # TESTE 1: Passo Constante vs IPS (Rosenbrock 2D)
    # ---------------------------------------------------------
    print(f"1. Comparando Passo Constante vs IPS (Rosenbrock 2D)")
    print(f"   Ponto inicial: {start_pos_2d}")

    # Rodar com Passo Constante
    path_const, iter_const = opt.gradient_descent(
        opt.rosenbrock_2d, 
        opt.grad_rosenbrock_2d, 
        start_pos_2d, 
        use_ips=False, 
        fixed_step=0.002, 
        max_iter=MAX_ITER
    )
    val_final_const = opt.rosenbrock_2d(path_const[-1])
    print(f"   [Constante] Iterações: {iter_const} | Valor Final: {val_final_const:.6f}")

    # Rodar com IPS (Estratégia 'worst')
    path_ips, iter_ips = opt.gradient_descent(
        opt.rosenbrock_2d, 
        opt.grad_rosenbrock_2d, 
        start_pos_2d, 
        use_ips=True, 
        ips_strategy='worst',
        max_iter=MAX_ITER
    )
    val_final_ips = opt.rosenbrock_2d(path_ips[-1])
    print(f"   [IPS Worst] Iterações: {iter_ips}   | Valor Final: {val_final_ips:.6f}")

    # --- GERAR GRÁFICOS ---
    plot_comparison(opt, path_const, path_ips, "Comparação: Constante vs IPS (Rosenbrock 2D)")

    # ---------------------------------------------------------
    # TESTE 2: Comparação de Estratégias IPS (Worst vs Cyclic)
    # ---------------------------------------------------------
    print("\n2. Comparando Estratégias IPS: 'Substituir Pior' vs 'Cíclico'")
    
    path_cyc, iter_cyc = opt.gradient_descent(
        opt.rosenbrock_2d, 
        opt.grad_rosenbrock_2d, 
        start_pos_2d, 
        use_ips=True, 
        ips_strategy='cyclic',
        max_iter=MAX_ITER
    )
    val_final_cyc = opt.rosenbrock_2d(path_cyc[-1])

    print(f"   [IPS Worst ] Iterações: {iter_ips} | Valor Final: {val_final_ips:.6g}")
    print(f"   [IPS Cyclic] Iterações: {iter_cyc} | Valor Final: {val_final_cyc:.6g}")

    if iter_ips < iter_cyc:
        print("   -> 'Worst' foi mais rápida (menos iterações).")
    elif iter_cyc < iter_ips:
        print("   -> 'Cyclic' foi mais rápida (menos iterações).")
    else:
        print("   -> Empate em iterações.")

    # ---------------------------------------------------------
    # TESTE 3: Rosenbrock 3D
    # ---------------------------------------------------------
    print("\n3. Teste de Extensão para 3D (Rosenbrock 3D)")
    start_pos_3d = [-1.2, 1.0, 2.0] 
    print(f"   Ponto inicial: {start_pos_3d}")
    
    path_3d, iter_3d = opt.gradient_descent(
        opt.rosenbrock_3d, 
        opt.grad_rosenbrock_3d, 
        start_pos_3d, 
        use_ips=True, 
        ips_strategy='worst',
        max_iter=MAX_ITER
    )
    final_3d = opt.rosenbrock_3d(path_3d[-1])
    
    plot_3d_trajectory(path_3d, "Otimização Rosenbrock 3D (Caminho percorrido)")
    
    print(f"   [3D IPS] Iterações: {iter_3d} | Mínimo atingido: {path_3d[-1]}")
    print(f"   Valor da função no final: {final_3d:.6f}")


if __name__ == "__main__":
    run_tests()