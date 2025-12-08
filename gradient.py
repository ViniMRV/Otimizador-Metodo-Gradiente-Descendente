import numpy as np
import matplotlib.pyplot as plt

class OptimizationMethods:
    def __init__(self):
        pass

    # --- Definição das Funções e seus Gradientes ---
    
    def func1(self, p):
        x, y = p[0], p[1]
        return x**4 + y**4 + 2*(x**2)*(y**2) + 6*x*y - 4*x - 4*y + 1

    def grad_func1(self, p):
        x, y = p[0], p[1]
        df_dx = 4*x**3 + 4*x*(y**2) + 6*y - 4
        df_dy = 4*y**3 + 4*(x**2)*y + 6*x - 4
        return np.array([df_dx, df_dy])

    def rosenbrock_2d(self, p):
        x, y = p[0], p[1]
        return 100*(y - x**2)**2 + (x - 1)**2

    def grad_rosenbrock_2d(self, p):
        x, y = p[0], p[1]
        df_dx = -400*x*(y - x**2) + 2*(x - 1)
        df_dy = 200*(y - x**2)
        return np.array([df_dx, df_dy])
    
    def rosenbrock_3d(self, p):
        # f(x,y,z) = 100(y - x^2)^2 + (x - 1)^2 + 100(z - y^2)^2 + (y - 1)^2
        x, y, z = p[0], p[1], p[2]
        return 100*(y - x**2)**2 + (x - 1)**2 + 100*(z - y**2)**2 + (y - 1)**2

    def grad_rosenbrock_3d(self, p):
        x, y, z = p[0], p[1], p[2]
        
        # Derivada parcial em relação a x (igual ao caso 2D)
        df_dx = -400*x*(y - x**2) + 2*(x - 1)
        
        # Derivada parcial em relação a y (recebe influência dos termos com x e z)
        # Termos envolvendo y: 100(y-x^2)^2 + (y-1)^2 + 100(z-y^2)^2
        df_dy = 200*(y - x**2) + 2*(y - 1) - 400*y*(z - y**2)
        
        # Derivada parcial em relação a z
        df_dz = 200*(z - y**2)
        
        return np.array([df_dx, df_dy, df_dz])

    # --- Interpolação Parabólica Sucessiva (IPS) ---
    def ips_step(self, func, x_curr, grad, r=0.0, s=0.1, t=0.2, max_iter=20, strategy='worst'):
        """
        Busca o melhor alpha.
        strategy: 'worst' (substitui quem tem maior f) ou 'cyclic' (substitui o mais antigo)
        """
        
        def g(alpha):
            return func(x_curr - alpha * grad)

        alphas = [r, s, t]
        vals = [g(r), g(s), g(t)]

        for k in range(max_iter):
            r_val, s_val, t_val = alphas[0], alphas[1], alphas[2]
            fr, fs, ft = vals[0], vals[1], vals[2]

            # Fórmula da Interpolação Parabólica (evitando divisão por zero)
            denominator = 2 * ((s_val - r_val) * (ft - fs) - (fs - fr) * (t_val - s_val))
            if abs(denominator) < 1e-15: 
                break 
            
            numerator = (fs - fr) * (t_val - r_val) * (t_val - s_val)
            u = (r_val + s_val) / 2 - numerator / denominator
            
            fu = g(u)
            
            # --- Lógica de Substituição ---
            if strategy == 'worst':
                # Substitui o ponto com maior valor de função (o "pior")
                idx_to_replace = np.argmax(vals)
            elif strategy == 'cyclic':
                # Substitui o estimativa menos recente (cíclico: 0, 1, 2, 0...)
                # No início r é o mais antigo (idx 0), na próxima iteração o antigo s vira r, etc.
                # A lógica simples aqui é usar o contador k
                idx_to_replace = k % 3
            else:
                raise ValueError("Strategy deve ser 'worst' ou 'cyclic'")

            # Atualiza os vetores
            alphas[idx_to_replace] = u
            vals[idx_to_replace] = fu
            
            # Critério de parada (se o passo u está muito próximo do s atual)
            if abs(u - s_val) < 1e-6:
                return u

        # Retorna o alpha que gerou o menor valor de função encontrado
        best_idx = np.argmin(vals)
        return alphas[best_idx]

    # --- Método do Gradiente Descendente ---
    def gradient_descent(self, func, grad_func, x0, use_ips=False, ips_strategy='worst', fixed_step=0.001, max_iter=1000, tol=1e-6):
        path = [x0]
        x = np.array(x0, dtype=float)
        
        for k in range(max_iter):
            gradient = grad_func(x)
            
            if np.linalg.norm(gradient) < tol:
                break
                
            if use_ips:
                # Passa a estratégia escolhida para o IPS
                alpha = self.ips_step(func, x, gradient, strategy=ips_strategy)
            else:
                alpha = fixed_step
            
            x = x - alpha * gradient
            path.append(x)
            
        return np.array(path), k