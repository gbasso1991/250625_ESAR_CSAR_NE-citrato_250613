#%% CSAR
'''
Rutina para leer .csv del sensor Rugged y calcular dT/dt 
en la Temperatura de Equilibrio 
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from glob import glob
from datetime import datetime
from uncertainties import ufloat, unumpy
from scipy.optimize import curve_fit
#%% Lector Templog
def lector_templog(path):
    '''
    Busca archivo *templog.csv en directorio especificado.
    muestras = False plotea solo T(dt). 
    muestras = True plotea T(dt) con las muestras superpuestas
    Retorna arrys timestamp,temperatura 
    '''
    data = pd.read_csv(path,sep=';',header=5,
                            names=('Timestamp','T_CH1','T_CH2'),usecols=(0,1,2),
                            decimal=',',engine='python') 
    temp_CH1  = pd.Series(data['T_CH1']).to_numpy(dtype=float)
    temp_CH2  = pd.Series(data['T_CH2']).to_numpy(dtype=float)
    timestamp = np.array([datetime.strptime(date,'%Y/%m/%d %H:%M:%S') for date in data['Timestamp']]) 

    return timestamp,temp_CH1, temp_CH2
#%% Levanto data
path_agua='data/250625_041427_agua.csv'
path_FF1='data/250625_043451_1.csv'
path_FF2='data/250625_045410_2.csv'

t_agua,T_agua,_=lector_templog(path_agua)
t_FF1,T_FF1,_=lector_templog(path_FF1)
t_FF2,T_FF2,_=lector_templog(path_FF2)

t_agua_0 = np.array([(t-t_agua[0]).total_seconds() for t in t_agua])
t_FF1_0 = np.array([(t-t_FF1[0]).total_seconds() for t in t_FF1])
t_FF2_0 = np.array([(t-t_FF2[0]).total_seconds() for t in t_FF2])

#%% Ploteo todo 
fig, ax=plt.subplots(constrained_layout=True)
ax.plot(t_agua_0,T_agua,label='Agua')
ax.plot(t_FF1_0,T_FF1,label='FF1')
ax.plot(t_FF2_0,T_FF2,label='FF2')
ax.grid()
ax.legend()
plt.show()
ax.set_xlim(0,)
#%% Agua y T equilibrio
# Obtener máscara booleana donde t_agua_0 >= 1000
mask = t_agua_0 >= 800

# Filtrar T_agua usando la máscara y calcular la media
T_agua_eq = round(np.mean(T_agua[mask]),1)

print(f"Temperatura media del agua desde t=1000 s: {T_agua_eq} °C")

fig, ax=plt.subplots(figsize=(8,4),constrained_layout=True)
ax.plot(t_agua_0,T_agua,'.-',label='Agua')
ax.axhline(T_agua_eq,0,1,c='tab:red',ls='--',label='T$_{eq}$ = '+f'{T_agua_eq:.1f} °C')
ax.grid()
ax.set_xlim(0,t_agua_0[-1])
ax.legend()
ax.set_xlabel('t (s)')
ax.set_ylabel('T (°C)')
plt.show()
#%% Recorto vectores y Busco indices donde T cruza la Teq en c/caso
t_FF1_0 = np.array([(t-t_FF1[np.nonzero(T_FF1==T_FF1.min())[0][0]]).total_seconds() for t in t_FF1])
t_FF2_0 = np.array([(t-t_FF2[np.nonzero(T_FF2==T_FF2.min())[0][0]]).total_seconds() for t in t_FF2])

def filtrar_tiempo_positivo(t, T):
    mask = t >= 0
    return t[mask], T[mask]

# Aplicar a todos los conjuntos
t_FF1_0, T_FF1 = filtrar_tiempo_positivo(t_FF1_0, T_FF1)
t_FF2_0, T_FF2 = filtrar_tiempo_positivo(t_FF2_0, T_FF2)
#%%
fig, ax=plt.subplots(constrained_layout=True)
ax.plot(t_agua_0,T_agua,label='Agua')
ax.plot(t_FF1_0,T_FF1,label='FF1')
ax.plot(t_FF2_0,T_FF2,label='FF2')
ax.axhline(T_agua_eq,0,1,c='tab:red',ls='--',label='T$_{eq}$ = '+f'{T_agua_eq:.1f} °C')

ax.grid()
ax.legend()
ax.set_xlim(0,)
plt.show()

indx_cruce_FF1 = np.nonzero(T_FF1==T_agua_eq)[0]
indx_cruce_FF2 = np.nonzero(T_FF2==T_agua_eq)[0]
# %% Ploteo cruces por Teq
fig, ax=plt.subplots(figsize=(8,4),constrained_layout=True)
ax.plot(t_FF1_0,T_FF1,label='FF1')
ax.scatter(t_FF1_0[indx_cruce_FF1],T_FF1[indx_cruce_FF1],label='FF1')

ax.plot(t_FF2_0,T_FF2,label='FF2')
ax.scatter(t_FF2_0[indx_cruce_FF2],T_FF2[indx_cruce_FF2],label='FF2')



ax.axhline(T_agua_eq,0,1,c='k',ls='--',alpha=0.5,label='T$_{eq}$ = '+f'{T_agua_eq:.1f} °C')
ax.set_xlim(0,)
ax.grid()
ax.legend(ncol=2,loc='lower right')
ax.set_xlabel('t (s)')
ax.set_ylabel('T (°C)')
plt.show()

#%% Función de ajustes alrededor de Teq
def ajustes_alrededor_Teq(Teq, t, T,label ,x=1.0):
    """
    Realiza ajustes lineal alrededor de Teq ± x.
    
    Args:
        Teq (float): Temperatura de equilibrio
        t (np.array): Array de tiempos
        T (np.array): Array de temperaturas
        x (float): Rango alrededor de Teq (default=1.0)
        
    Returns:
        tuple: (dict_lin, dict_exp) donde:
            - dict_lin: Diccionario con resultados del ajuste lineal
    """
    # Crear máscara para el intervalo de interés
    mask = (T >= Teq - x) & (T <= Teq + x)
    t_interval = t[mask]
    T_interval = T[mask]
    
    # Ajuste lineal
    coeff_lin = np.polyfit(t_interval, T_interval, 1)
    poly_lin = np.poly1d(coeff_lin)
    r2_lin = np.corrcoef(T_interval, poly_lin(t_interval))[0,1]**2
    t_fine = np.linspace(t_interval.min()-80, t_interval.max()+80, 100)
    
    # Preparar diccionario para resultados lineales
    dict_lin = {
        'pendiente': coeff_lin[0],
        'ordenada': coeff_lin[1],
        'r2': r2_lin,
        't_interval': t_interval,
        'T_interval': T_interval,
        'funcion': poly_lin,
        'ecuacion': f"{coeff_lin[0]:.3f}t + {coeff_lin[1]:.3f}",
        'rango_x': x,
        'AL_t':t_fine,
        'AL_T':poly_lin(t_fine)
        }
    
    # Crear figura 
    fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True)
    ax.plot(t, T, '.-', label=label)
    
    # Plotear ajustes con el rango extendido que definiste
    ax.plot(t_fine, poly_lin(t_fine), '-',c='tab:green',lw=2, 
            label=f'Ajuste lineal: {dict_lin["ecuacion"]} (R²={r2_lin:.3f})')

    ax.axhspan(Teq-x, Teq+x, 0, 1, color='tab:red', alpha=0.3, label='T$_{eq}\pm\Delta T$ ='+ f' {T_agua_eq} $\pm$ {x} ºC')
    
    ax.set_xlabel('t (s)')
    ax.set_ylabel('T (°C)')
    ax.grid()
    ax.legend()
    #ax.set_xlim(400, 600)
    #ax.set_ylim(T_interval[0]-3, T_interval[-1]+3)
    plt.show()

    # Imprimir resultados (manteniendo tu formato)
    print("\nResultados del ajuste lineal:")
    print(f"Pendiente: {dict_lin['pendiente']:.5f} °C/s")
    print(f"Ordenada: {dict_lin['ordenada']:.5f} °C")
    print(f"Coeficiente R²: {dict_lin['r2']:.5f}")
    
    return dict_lin
#%%
def ajustes_lineal_T_arbitraria(Tcentral, t, T, label, x=1.0):
    """
    Realiza ajustes lineal alrededor de Tcentral ± x usando curve_fit.
    
    Args:
        Tcentral (float): Temperatura de equilibrio
        t (np.array): Array de tiempos
        T (np.array): Array de temperaturas
        x (float): Rango alrededor de Tcentral (default=1.0)
        
    Returns:
        tuple: (dict_lin, dict_exp) donde:
            - dict_lin: Diccionario con resultados del ajuste lineal
    """
    # Definir la función lineal para curve_fit
    def linear_func(x, a, b):
        return a * x + b
    
    # Crear máscara para el intervalo de interés
    mask = (T >= Tcentral - x) & (T <= Tcentral + x)
    t_interval = t[mask]
    T_interval = T[mask]
    
    # Ajuste lineal con curve_fit
    popt, pcov = curve_fit(linear_func, t_interval, T_interval)
    perr = np.sqrt(np.diag(pcov))  # Desviaciones estándar de los parámetros
    
    # Crear función de ajuste
    poly_lin = lambda x: linear_func(x, *popt)
    
    # Calcular R²
    residuals = T_interval - poly_lin(t_interval)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((T_interval - np.mean(T_interval))**2)
    r2_lin = 1 - (ss_res / ss_tot)
    
    t_fine = np.linspace(t_interval.min()-80, t_interval.max()+80, 100)
    
    # Crear ufloat para la pendiente con su incertidumbre
    pendiente_ufloat = ufloat(popt[0], perr[0])
    
    # Preparar diccionario para resultados lineales
    dict_lin = {
        'pendiente': pendiente_ufloat,
        'ordenada': ufloat(popt[1], perr[1]),
        'r2': r2_lin,
        't_interval': t_interval,
        'T_interval': T_interval,
        'funcion': poly_lin,
        'ecuacion': f"({popt[0]:.3f}±{perr[0]:.3f})t + ({popt[1]:.3f}±{perr[1]:.3f})",
        'rango_x': x,
        'AL_t': t_fine,
        'AL_T': poly_lin(t_fine),
        'covarianza': pcov
    }
    
    # Crear figura 
    fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True)
    ax.plot(t, T, '.-', label=label)
    
    # Plotear ajustes con el rango extendido que definiste
    ax.plot(t_fine, poly_lin(t_fine), '-', c='tab:green', lw=2, 
            label=f'Ajuste lineal: {dict_lin["ecuacion"]} (R²={r2_lin:.3f})')

    ax.axhspan(Tcentral-x, Tcentral+x, 0, 1, color='tab:red', alpha=0.3, 
               label='T$_{eq}\pm\Delta T$ ='+ f' {Tcentral} $\pm$ {x} ºC')
    
    ax.set_xlabel('t (s)')
    ax.set_ylabel('T (°C)')
    ax.grid()
    ax.legend()
    plt.show()

    # Imprimir resultados (manteniendo tu formato)
    print("\nResultados del ajuste lineal:")
    print(f"Pendiente: {dict_lin['pendiente']} °C/s")
    print(f"Ordenada: {dict_lin['ordenada']} °C")
    print(f"Coeficiente R²: {dict_lin['r2']:.5f}")
    
    
    return dict_lin
#%%# Resultados 

resultados_FF1 = ajustes_lineal_T_arbitraria(24.0, t_FF1_0, T_FF1,'FF1', x=5.0)
resultados_FF2 = ajustes_lineal_T_arbitraria(25.0, t_FF2_0, T_FF2,'FF2', x=5.0)
#%%
concentracion=ufloat(8.2,0.4)
dTdt_lineal_promedio=np.mean([resultados_FF1['pendiente'],resultados_FF2['pendiente']])
print(f'Pendiente promedio = {dTdt_lineal_promedio:.5f} ºC/s')
CSAR_lineal = dTdt_lineal_promedio*4.186e3/concentracion
print(f'CSAR = {CSAR_lineal:.0f} W/g (ajuste lineal)')


#%%
# fig, ax=plt.subplots(figsize=(10,5),constrained_layout=True)
# ax.plot(t_FF1_0,T_FF1,label='FF1',c='C0')
# ax.scatter(t_FF1_0[indx_cruce_FF1[0]],T_FF1[indx_cruce_FF1[0]],c='C0')
# ax.plot(resultados_FF1['AL_t'],resultados_FF1['AL_T'],label='AL FF1',c='C0')

# ax.plot(t_FF2_0,T_FF2,label='FF2',c='C1')
# ax.scatter(t_FF2_0[indx_cruce_FF2[0]],T_FF2[indx_cruce_FF2[0]],c='C1')
# ax.plot(resultados_FF2['AL_t'],resultados_FF2['AL_T'],label='AL FF2',c='C1')



# ax.axhline(T_agua_eq,0,1,c='k',ls='--',alpha=0.5,label='T$_{eq}$ = '+f'{T_agua_eq:.1f} °C')

# #ax.text(0.1,0.85,f'<m> = {dTdt_lineal:.3f} ºC/s',bbox=dict(alpha=0.7),fontsize=14,ha='left',transform=ax.transAxes)

# ax.set_xlim(160,420)
# ax.set_ylim(18,30)
# ax.grid()
# ax.legend(ncol=2,loc='lower right')
# ax.set_xlabel('t (s)')
# ax.set_ylabel('T (°C)')
# plt.savefig('AL_y_pendientes.png',facecolor='w',dpi=400)
# plt.show()














# # %% Exponencial

# def ajExp_alrededor_Teq(Teq, t, T, x=3.0):
#     """
#     Realiza ajuste exponencial alrededor de Teq ± x.
    
#     Args:
#         Teq (float): Temperatura de equilibrio
#         t (np.array): Array de tiempos
#         T (np.array): Array de temperaturas
#         x (float): Rango alrededor de Teq (default=1.0)
#     """
#     # Crear máscara para el intervalo de interés
#     mask = (T >= Teq - x) & (T <= Teq + x)
#     t_interval = t[mask]
#     T_interval = T[mask]
    
#     # Ajuste exponencial (T = a + b*exp(-c*t))
#     try:
#         from scipy.optimize import curve_fit
#         def exp_func(t, a, b, c):
#             return a - b * np.exp(-c * t)
        
#         # Estimación inicial para mejor convergencia
#         p0 = [Teq, x, 1/(t_interval[-1] - t_interval[0])]
#         popt, pcov = curve_fit(exp_func, t_interval, T_interval, p0=p0)
#         a_exp, b_exp, c_exp = popt
#         A_exp=ufloat(a_exp,np.sqrt(pcov[0,0]))
#         B_exp=ufloat(b_exp,np.sqrt(pcov[1,1]))
#         C_exp=ufloat(c_exp,np.sqrt(pcov[2,2]))
        
#         r2_exp = np.corrcoef(T_interval, exp_func(t_interval, *popt))[0,1]**2
#         exp_success = True

#     except Exception as e:
#         print(f"Error en ajuste exponencial: {e}")
#         exp_success = False
    
#     # Crear figura
#     fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True)
#     ax.plot(t, T, 'o-',label='Datos originales')
#     #ax.plot(t_interval, T_interval, 'o', label=f'Datos en T_eq ± {x}°C')
    
#     # Plotear ajustes
#     t_fine = np.linspace(t_interval.min()-50, t_interval.max()+50, 100)
#     if exp_success:
#         ax.plot(t_fine, exp_func(t_fine, *popt),ls='-',lw=2,
#                 label=f'Ajuste exp: {a_exp:.3f} + {b_exp:.3f}exp(-{c_exp:.3f}t) (R²={r2_exp:.3f})')
    
#     # ax.axhline(Teq, color='r', linestyle='--', label=f'T_eq = {Teq}°C')
    
#     ax.axhspan(Teq-x, Teq+x,0,1,color='tab:green',alpha=0.3,label='$\Delta T$= $\pm$1 ºC')
#     ax.set_xlabel('t (s)')
#     ax.set_ylabel('T (°C)')
#     ax.grid()
#     ax.legend()
#     #ax.set_xlim(400,600)
#     #ax.set_ylim(T_interval[0]-3,T_interval[-1]+3)
#     plt.show()
    
#     # Imprimir resultados
#     dict_exp = {
#     'T_inf_A': A_exp, 
#     'Amplitud_B': B_exp, 
#     'Tasa_decaimiento_C': C_exp,
#     'r2': r2_exp,
#     't_interval': t_interval,
#     'T_interval': T_interval,
#     'funcion': exp_func,
#     'ecuacion': f"{a_exp:.3f} + {b_exp:.3f}*exp(-{c_exp}*t) ",
#     'rango_x': x,
#     'Aexp_t':t_fine,
#     'Aexp_T':exp_func(t_fine,a_exp, b_exp, c_exp)
#     } 

#     # print("\nResultados del ajuste exponencial:")
#     if exp_success:
#         print("\nResultados del ajuste exponencial:")
#         print(f"  T_inf: {A_exp:.5f} °C")
#         print(f"  Amplitud: {B_exp:.5f} °C")
#         print(f"  Tasa decaimiento: {C_exp:.5f} 1/s")
#         print(f"  Tau: {1/C_exp:.5f} s")
#         print(f"  Coeficiente R²: {r2_exp:.5f}")
  
#     return dict_exp
# #%%
# # Aplicar la función a tus datos
# res_exp_FF1=ajExp_alrededor_Teq(T_agua_eq, t_FF1_0, T_FF1, x=3.0)
# A,B,C = res_exp_FF1['T_inf_A'],res_exp_FF1['Amplitud_B'],res_exp_FF1['Tasa_decaimiento_C'] 
# teq=(1/C)*unumpy.log(B/(A-T_agua_eq))
# print(f't eq = {teq:.3uf} s')
# dTdt_exp_FF1 = B*C*unumpy.exp(-C*teq)
# print(f'dT/dt = {dTdt_exp_FF1:.3uf} ºC/s')

# #%%
# res_exp_FF2=ajExp_alrededor_Teq(T_agua_eq, t_FF2_0, T_FF2, x=3.0)
# A,B,C = res_exp_FF2['T_inf_A'],res_exp_FF2['Amplitud_B'],res_exp_FF2['Tasa_decaimiento_C'] 
# teq=(1/C)*unumpy.log(B/(A-T_agua_eq))
# print(teq)
# dTdt_exp_FF2 = B*C*unumpy.exp(-C*teq)
# print(dTdt_exp_FF2)
# #%%
# teq=(1/C)*unumpy.log(B/(A-T_agua_eq))
# print(teq)
# # %%#%%
# teq=(1/C)*unumpy.log(B/(A-T_agua_eq))
# print(teq)
# # %%
# print(f'dT/dt = {dTdt_mean} ºC/s')
# # %%
