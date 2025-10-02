import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

st.set_page_config(page_title="Método Gráfico", layout="centered")

st.title("📈 Método Gráfico - Programação Linear")
st.write("Digite as restrições da forma **aX1 + bX2 (≤, ≥ ou =) c**")

# Número de restrições
n = st.number_input("Número de restrições", min_value=2, max_value=5, value=3, step=1)

restricoes = []
for i in range(n):
    col1, col2, col3, col4 = st.columns([2,2,2,2])
    with col1:
        a = st.number_input(f"Coeficiente X1 (R{i+1})", value=1.0, key=f"a{i}")
    with col2:
        b = st.number_input(f"Coeficiente X2 (R{i+1})", value=1.0, key=f"b{i}")
    with col3:
        sinal = st.selectbox(f"Sinal (R{i+1})", ["<=", ">=", "="], key=f"sinal{i}")
    with col4:
        c = st.number_input(f"Valor (R{i+1})", value=10.0, key=f"c{i}")
    
    restricoes.append((a,b,sinal,c))

st.subheader("Função Objetivo")
col1, col2 = st.columns(2)
with col1:
    c1 = st.number_input("Coeficiente X1", value=1.0)
with col2:
    c2 = st.number_input("Coeficiente X2", value=1.0)

tipo = st.radio("Escolha o tipo de problema:", ["Maximizar", "Minimizar"])

if st.button("Resolver"):
    pontos = []

    # Interseções entre pares de restrições
    for r1, r2 in combinations(restricoes, 2):
        a1, b1, _, c1r = r1
        a2, b2, _, c2r = r2
        A = np.array([[a1,b1],[a2,b2]])
        B = np.array([c1r,c2r])
        try:
            sol = np.linalg.solve(A,B)
            if np.all(sol >= -1e-6):
                pontos.append(sol)
        except np.linalg.LinAlgError:
            pass

    # Interseções com os eixos
    for a,b,sinal,c in restricoes:
        if b != 0:
            y = c/b
            if y >= 0: pontos.append([0,y])
        if a != 0:
            x = c/a
            if x >= 0: pontos.append([x,0])

    # Sempre adiciona origem
    pontos.append([0,0])

    # Filtrar pontos válidos
    pontos_validos = []
    for x,y in pontos:
        valido = True
        for a,b,s,c in restricoes:
            if s == "<=" and not (a*x+b*y <= c+1e-6): valido=False
            if s == ">=" and not (a*x+b*y >= c-1e-6): valido=False
            if s == "="  and not (abs(a*x+b*y-c) <= 1e-6): valido=False
        if valido:
            pontos_validos.append([x,y])

    pontos_validos = np.array(pontos_validos)

    if len(pontos_validos) == 0:
        st.error("❌ Nenhuma região viável encontrada!")
    else:
        # Avaliar função objetivo
        valores = [c1*x + c2*y for x,y in pontos_validos]
        if tipo == "Maximizar":
            idx = np.argmax(valores)
        else:
            idx = np.argmin(valores)

        x_opt, y_opt = pontos_validos[idx]
        z_opt = valores[idx]

        st.success(f"✅ Ótimo encontrado em (X1={x_opt:.2f}, X2={y_opt:.2f}) → Z = {z_opt:.2f}")
        st.write(f"**Vetor gradiente (∇Z): [{c1:.2f}, {c2:.2f}]**")

        # Gráfico
        fig, ax = plt.subplots()
        max_x = max(p[0] for p in pontos_validos)+2
        max_y = max(p[1] for p in pontos_validos)+2
        ax.set_xlim(0, max_x)
        ax.set_ylim(0, max_y)

        # Plotar restrições
        x_vals = np.linspace(0, max_x, 300)
        for a,b,s,c in restricoes:
            if b != 0:
                y_vals = (c - a*x_vals)/b
                ax.plot(x_vals, y_vals, label=f"{a}X1 + {b}X2 {s} {c}")

        # Região viável (ordenando por ângulo polar)
        if len(pontos_validos) > 2:
            angles = np.arctan2(pontos_validos[:,1], pontos_validos[:,0])
            order = np.argsort(angles)
            poligono = pontos_validos[order]
            ax.fill(poligono[:,0], poligono[:,1], 'gray', alpha=0.2)

        # Ponto ótimo
        ax.scatter(x_opt, y_opt, color="red", s=100, label=f"Ótimo Z={z_opt:.2f}")

        # Vetor gradiente
        ax.arrow(0, 0, c1, c2, head_width=0.3, head_length=0.3, fc='green', ec='green', length_includes_head=True)
        ax.text(c1, c2, "∇Z", color="green", fontsize=12)

        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.legend()
        st.pyplot(fig)
