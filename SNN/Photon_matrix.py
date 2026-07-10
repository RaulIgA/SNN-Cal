import pandas as pd
import dataset as ds
import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import jensenshannon

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 20,
    'axes.labelsize': 17,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'figure.titlesize': 24,
    'legend.fontsize': 14,
})

num_events = 4900

# 1. Carga de datos
dfk1 = pd.DataFrame(ds.readfile("./Data/PrimaryOnly/Uniform/kaon/kaon_1.dat", primary_only=True, merged=True))
dfk2 = pd.DataFrame(ds.readfile("./Data/PrimaryOnly/Uniform/kaon/kaon_2.dat", primary_only=True, merged=True))
dfk3 = pd.DataFrame(ds.readfile("./Data/PrimaryOnly/Uniform/kaon/kaon_3.dat", primary_only=True, merged=True))
dfk4 = pd.DataFrame(ds.readfile("./Data/PrimaryOnly/Uniform/kaon/kaon_4.dat", primary_only=True, merged=True))
dfk5 = pd.DataFrame(ds.readfile("./Data/PrimaryOnly/Uniform/kaon/kaon_5.dat", primary_only=True, merged=True))

dfk = pd.concat([dfk1,dfk2,dfk3,dfk4,dfk5],axis=1)

dfpi1 = pd.DataFrame(ds.readfile("./Data/PrimaryOnly/Uniform/pion/pion_1.dat", primary_only=True, merged=True))
dfpi2 = pd.DataFrame(ds.readfile("./Data/PrimaryOnly/Uniform/pion/pion_2.dat", primary_only=True, merged=True))
dfpi3 = pd.DataFrame(ds.readfile("./Data/PrimaryOnly/Uniform/pion/pion_3.dat", primary_only=True, merged=True))
dfpi4 = pd.DataFrame(ds.readfile("./Data/PrimaryOnly/Uniform/pion/pion_4.dat", primary_only=True, merged=True))
dfpi5 = pd.DataFrame(ds.readfile("./Data/PrimaryOnly/Uniform/pion/pion_5.dat", primary_only=True, merged=True))

dfpi = pd.concat([dfpi1,dfpi2,dfpi3,dfpi4,dfpi5],axis = 1 )


dfp1 = pd.DataFrame(ds.readfile("./Data/PrimaryOnly/Uniform/proton/proton_1.dat", primary_only=True, merged=True))
dfp2 = pd.DataFrame(ds.readfile("./Data/PrimaryOnly/Uniform/proton/proton_2.dat", primary_only=True, merged=True))
dfp3 = pd.DataFrame(ds.readfile("./Data/PrimaryOnly/Uniform/proton/proton_3.dat", primary_only=True, merged=True))
dfp4 = pd.DataFrame(ds.readfile("./Data/PrimaryOnly/Uniform/proton/proton_4.dat", primary_only=True, merged=True))
dfp5 = pd.DataFrame(ds.readfile("./Data/PrimaryOnly/Uniform/proton/proton_5.dat", primary_only=True, merged=True))

dfp = pd.concat([dfp1,dfp2,dfp3,dfp4,dfp5], axis = 1)

print("Shape Kaones:", dfk.shape)
print("Shape Piones:", dfpi.shape)
print("Shape Protones:", dfp.shape)

def count_initial_zero_rows(df, n_events):
    zero_rows_counts = []
    max_events = min(n_events, df.shape[1])
    
    for j in range(max_events):
        matrix = df.iloc[0, j]
        matrix = np.array(matrix)
        
        if matrix.ndim == 0:
            continue
            
        count = 0
        for row in matrix:
            if np.all(row == 0):
                count += 1
            else:
                break
                
        zero_rows_counts.append(count)
        
    return np.mean(zero_rows_counts) if zero_rows_counts else 0

mean_zero_rows_k = count_initial_zero_rows(dfk, num_events)
mean_zero_rows_pi = count_initial_zero_rows(dfpi, num_events)
mean_zero_rows_p = count_initial_zero_rows(dfp, num_events)

print(f"Media de filas iniciales con todo 0s (Kaones): {mean_zero_rows_k:.2f}")
print(f"Media de filas iniciales con todo 0s (Piones): {mean_zero_rows_pi:.2f}")
print(f"Media de filas iniciales con todo 0s (Protones): {mean_zero_rows_p:.2f}")


def get_matrix_row_sums(df, n_events):
    x_matrix_rows = []
    y_sums = []
    
    max_events = min(n_events, df.shape[1])
    
    for j in range(max_events):
        matrix = df.iloc[0, j]
        matrix = np.array(matrix)
        
        if matrix.ndim == 2:
            col_sums = np.sum(matrix, axis=0)
            num_cols = len(col_sums)
            x_matrix_rows.extend(range(num_cols))
            y_sums.extend(col_sums)
    
    return x_matrix_rows, y_sums

print("Procesando Kaones (Fila 0)...")
x_k, y_k = get_matrix_row_sums(dfk, num_events)

print("Procesando Piones (Fila 0)...")
x_pi, y_pi = get_matrix_row_sums(dfpi, num_events)

print("Procesando Protones (Fila 0)...")
x_p, y_p = get_matrix_row_sums(dfp, num_events)

fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Kaones
axs[0].scatter(x_k, np.log1p(y_k), color='blue', alpha=0.05, marker='o', s=5)
axs[0].set_title('Kaones', fontsize=20)
axs[0].set_ylabel('Suma de la columna', fontsize=17)
axs[0].grid(True, linestyle='--', alpha=0.5)

# Piones
axs[1].scatter(x_pi, np.log1p(y_pi), color='green', alpha=0.05, marker='s', s=5)
axs[1].set_title('Piones', fontsize=20)
axs[1].set_ylabel('Suma de la columna', fontsize=17)
axs[1].grid(True, linestyle='--', alpha=0.5)

# Protones
axs[2].scatter(x_p, np.log1p(y_p), color='red', alpha=0.05, marker='^', s=5)
axs[2].set_title('Protones ', fontsize=20)
axs[2].set_ylabel('Suma de la columna', fontsize=17)
axs[2].set_xlabel('Índice de la columna en la Matriz (0 - 99)', fontsize=17)
axs[2].grid(True, linestyle='--', alpha=0.5)

fig.suptitle('Fotones recogidos para cada sensor', fontsize=24, y=0.97)

plt.tight_layout()
plt.show()
plt.savefig('Graficas_dataset/Distribucion_espacial_log.png')

# Distribucion espacial 
fig_lin, axs_lin = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

axs_lin[0].scatter(x_k, y_k, color='blue', alpha=0.05, marker='o', s=5)
axs_lin[0].set_title('Kaones', fontsize=20)
axs_lin[0].set_ylabel('Suma de la columna', fontsize=17)
axs_lin[0].grid(True, linestyle='--', alpha=0.5)

axs_lin[1].scatter(x_pi, y_pi, color='green', alpha=0.05, marker='s', s=5)
axs_lin[1].set_title('Piones', fontsize=20)
axs_lin[1].set_ylabel('Suma de la columna', fontsize=17)
axs_lin[1].grid(True, linestyle='--', alpha=0.5)

axs_lin[2].scatter(x_p, y_p, color='red', alpha=0.05, marker='^', s=5)
axs_lin[2].set_title('Protones ', fontsize=20)
axs_lin[2].set_ylabel('Suma de la columna', fontsize=17)
axs_lin[2].set_xlabel('Índice de la columna en la Matriz (0 - 99)', fontsize=17)
axs_lin[2].grid(True, linestyle='--', alpha=0.5)

fig_lin.suptitle('Fotones recogidos para cada sensor', fontsize=24, y=0.97)

plt.tight_layout()
plt.savefig('Graficas_dataset/Distribucion_espacial.png')

# Distribucion temporal 
def get_matrix_col_sums(df, n_events):
    x_matrix_cols = []
    y_sums = []

    max_events = min(n_events, df.shape[1])

    for j in range(max_events):
        matrix = df.iloc[0, j]
        matrix = np.array(matrix)

        if matrix.ndim == 2:
            row_sums = np.sum(matrix, axis=1)
            num_rows = len(row_sums)
            x_matrix_cols.extend(range(num_rows))
            y_sums.extend(row_sums)

    return x_matrix_cols, y_sums

print("Procesando Kaones (temporal)...")
t_k, yt_k = get_matrix_col_sums(dfk, num_events)

print("Procesando Piones (temporal)...")
t_pi, yt_pi = get_matrix_col_sums(dfpi, num_events)

print("Procesando Protones (temporal)...")
t_p, yt_p = get_matrix_col_sums(dfp, num_events)

fig_t, axs_t = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

axs_t[0].scatter(t_k, yt_k, color='blue', alpha=0.05, marker='o', s=5)
axs_t[0].set_title('Kaones', fontsize=20)
axs_t[0].set_ylabel(r'$N_\gamma^{tot}$', fontsize=17)
axs_t[0].grid(True, linestyle='--', alpha=0.5)

axs_t[1].scatter(t_pi, yt_pi, color='green', alpha=0.05, marker='s', s=5)
axs_t[1].set_title('Piones', fontsize=20)
axs_t[1].set_ylabel(r'$N_\gamma^{tot}$', fontsize=17)
axs_t[1].grid(True, linestyle='--', alpha=0.5)

axs_t[2].scatter(t_p, yt_p, color='red', alpha=0.05, marker='^', s=5)
axs_t[2].set_title('Protones ', fontsize=20)
axs_t[2].set_ylabel(r'$N_\gamma^{tot}$', fontsize=17)
axs_t[2].set_xlabel('Índice temporal (0 - 99)', fontsize=17)
axs_t[2].grid(True, linestyle='--', alpha=0.5)

fig_t.suptitle('Fotones recogidos por intervalo temporal', fontsize=24, y=0.97)

plt.tight_layout()
plt.savefig('Graficas_dataset/Distribucion_temporal.png')

num_events = 4000

# Log(Energía) vs Log(Suma total de fotones por evento)
def extract_log_features(df, n_events):
    log_photons = []
    log_energy = []
    
    max_events = min(n_events, df.shape[1])
    
    for j in range(max_events):
        matrix = df.iloc[0, j]
        raw_energy = df.iloc[1, j]
        
        total_ph = float(np.sum(matrix))
        energy = float(np.sum(raw_energy))
        
        if total_ph > 0 and energy > 0:
            log_photons.append(np.log10(total_ph))
            log_energy.append(np.log10(energy))
            
    return log_photons, log_energy
log_ph_k,  log_e_k  = extract_log_features(dfk, num_events)
log_ph_p,  log_e_p  = extract_log_features(dfp, num_events)
log_ph_pi, log_e_pi = extract_log_features(dfpi, num_events)

fig_scatter, axes_scatter = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

# Kaones
axes_scatter[0].scatter(log_ph_k, log_e_k, color='purple', alpha=0.3, s=5)
axes_scatter[0].set_title("Kaones")
axes_scatter[0].set_xlabel("log10(Suma de todos los fotones)")
axes_scatter[0].set_ylabel("log10(Energía del evento)")

# Protones
axes_scatter[1].scatter(log_ph_p, log_e_p, color='orange', alpha=0.3, s=5)
axes_scatter[1].set_title("Protones")
axes_scatter[1].set_xlabel("log10(Suma de todos los fotones)")

# Piones
axes_scatter[2].scatter(log_ph_pi, log_e_pi, color='green', alpha=0.3, s=5)
axes_scatter[2].set_title("Piones")
axes_scatter[2].set_xlabel("log10(Suma de todos los fotones)")

plt.tight_layout()
fig_scatter.savefig('Graficas_dataset/scatter_log_energia_vs_fotones.png', dpi=300)



phk = dfk.iloc[[0],:]
php = dfp.iloc[[0],:]
phpi = dfpi.iloc[[0],:]

fotonesk_t, fotonesp_t, fotonespi_t = [], [], []

fotonesk_s, fotonesp_s, fotonespi_s = [], [], []

for j in range(num_events):
    # Kaones
    data_matrix_k = phk.iloc[0, j]
    for i in range(100):
        if int(data_matrix_k[i,:].sum()) > 0:
            fotonesk_t.append(int(data_matrix_k[i,:].sum()))
        if int(data_matrix_k[:,i].sum()) > 0:
            fotonesk_s.append(int(data_matrix_k[:,i].sum()))

    # Protones
    data_matrix_p = php.iloc[0, j]
    for i in range(100):
        if int(data_matrix_p[i,:].sum()) > 0:
            fotonesp_t.append(int(data_matrix_p[i,:].sum()))
        if int(data_matrix_p[:,i].sum()) > 0:
            fotonesp_s.append(int(data_matrix_p[:,i].sum()))

    # Piones
    data_matrix_pi = phpi.iloc[0, j]
    for i in range(100):
        if int(data_matrix_pi[i,:].sum()) > 0:
            fotonespi_t.append(int(data_matrix_pi[i,:].sum()))
        if int(data_matrix_pi[:,i].sum()) > 0:
            fotonespi_s.append(int(data_matrix_pi[:,i].sum()))

# Histograma por timestep
fig_hist_t, axes_hist_t = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

axes_hist_t[0].hist(np.log10(fotonesk_t), bins=100, color='purple', edgecolor='black')
axes_hist_t[0].set_title("Kaones")
axes_hist_t[0].set_xlabel(r"$\log_{10}N^{tot}_\gamma\mathrm{(por intervalo temporal)}$")
axes_hist_t[0].set_ylabel("Frecuencia")

axes_hist_t[1].hist(np.log10(fotonesp_t), bins=100, color='orange', edgecolor='black')
axes_hist_t[1].set_title("Protones")
axes_hist_t[1].set_xlabel(r"$\log_{10}N^{tot}_\gamma\mathrm{(por intervalo temporal)}$")

axes_hist_t[2].hist(np.log10(fotonespi_t), bins=100, color='green', edgecolor='black')
axes_hist_t[2].set_title("Piones")
axes_hist_t[2].set_xlabel(r"$\log_{10}N^{tot}_\gamma\mathrm{(por intervalo temporal)}$")

fig_hist_t.suptitle("Distribución de fotones agregados por timestep", fontsize=20)
plt.tight_layout()
fig_hist_t.savefig('Graficas_dataset/histogramas_fotones_por_timestep.png')

# Histograma por sensor
fig_hist_s, axes_hist_s = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

axes_hist_s[0].hist(np.log10(fotonesk_s), bins=100, color='purple', edgecolor='black')
axes_hist_s[0].set_title("Kaones")
axes_hist_s[0].set_xlabel(r"$\log_{10}(N_\gamma^i$)")
axes_hist_s[0].set_ylabel("Frecuencia ")

axes_hist_s[1].hist(np.log10(fotonesp_s), bins=100, color='orange', edgecolor='black')
axes_hist_s[1].set_title("Protones")
axes_hist_s[1].set_xlabel(r"$\log_{10}(N_\gamma^i$)")

axes_hist_s[2].hist(np.log10(fotonespi_s), bins=100, color='green', edgecolor='black')
axes_hist_s[2].set_title("Piones")
axes_hist_s[2].set_xlabel(r"$\log_{10}(N_\gamma^i$)")  
fig_hist_s.suptitle("Distribución de fotones agregados por sensor", fontsize=20)
plt.tight_layout()
fig_hist_s.savefig('Graficas_dataset/histogramas_fotones_por_sensor.png')

# Jensen-Shannon entre pares de particulas

def profile_js(idx_a, w_a, idx_b, w_b, n=100):
    pa = np.bincount(idx_a, weights=w_a, minlength=n)
    pb = np.bincount(idx_b, weights=w_b, minlength=n)
    return jensenshannon(pa, pb, base=2) ** 2

print("Perfil temporal p(t) (fracción de fotones por timestep):")
js_t_KP  = profile_js(t_k, yt_k, t_pi, yt_pi)
js_t_KPr = profile_js(t_k, yt_k, t_p, yt_p)
js_t_PP  = profile_js(t_pi, yt_pi, t_p, yt_p)
print(f"  JS_KP  (kaón-pión)   = {js_t_KP:.4f}")
print(f"  JS_KPr (kaón-protón) = {js_t_KPr:.4f}")
print(f"  JS_PP  (pión-protón) = {js_t_PP:.4f}")

print("Perfil espacial p(s) (fracción de fotones por sensor):")
js_s_KP  = profile_js(x_k, y_k, x_pi, y_pi)
js_s_KPr = profile_js(x_k, y_k, x_p, y_p)
js_s_PP  = profile_js(x_pi, y_pi, x_p, y_p)
print(f"  JS_KP  (kaón-pión)   = {js_s_KP:.4f}")
print(f"  JS_KPr (kaón-protón) = {js_s_KPr:.4f}")
print(f"  JS_PP  (pión-protón) = {js_s_PP:.4f}")

