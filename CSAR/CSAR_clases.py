#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from glob import glob
from scipy.optimize import curve_fit
from uncertainties import ufloat, unumpy
import os
#%%
class SensorCSV:
    def __init__(self, filepath):
        self.filepath = filepath
        self.filename = filepath.split('/')[-1]
        self.timestamp, self.T_CH1, self.T_CH2 = self._load_data()
        self.t0 = self._zero_time(self.timestamp)

    def _load_data(self):
        data = pd.read_csv(self.filepath, sep=';', header=5, names=("Timestamp", "T_CH1", "T_CH2"),
                           usecols=(0, 1, 2), decimal=',', engine='python')
        timestamp = np.array([datetime.strptime(ts, '%Y/%m/%d %H:%M:%S') for ts in data['Timestamp']])
        return timestamp, data['T_CH1'].astype(float).values, data['T_CH2'].astype(float).values

    def _zero_time(self, timestamps):
        return np.array([(t - timestamps[0]).total_seconds() for t in timestamps])

class AnalisisTermico:
    def __init__(self, path='.', delta_lineal=1.0, delta_exp=3.0, t_eq_min=1000,
                 concentracion=(10.0, 0.1), hacer_ajuste_lineal=True, hacer_ajuste_exponencial=True):
        self.delta_lineal = delta_lineal
        self.delta_exp = delta_exp
        self.t_eq_min = t_eq_min
        self.concentracion = ufloat(concentracion[0], concentracion[1])
        self.hacer_ajuste_lineal = hacer_ajuste_lineal
        self.hacer_ajuste_exponencial = hacer_ajuste_exponencial
        self.files = glob(f'{path}/*.csv')
        self.sensores = [SensorCSV(f) for f in self.files]
        self.T_eq = None
        self.resultados_lineales = []
        self.resultados_exp = []
        self.muestras_procesadas = []
        self._clasificar_archivos()

    def _clasificar_archivos(self):
        self.agua = [s for s in self.sensores if 'agua' in s.filename.lower()][0]
        self.muestras = [s for s in self.sensores if 'agua' not in s.filename.lower()]

    def calcular_T_eq(self):
        t = self.agua.t0
        T = self.agua.T_CH1
        mask = t >= self.t_eq_min
        self.T_eq = round(sum(T[mask]) / len(T[mask]), 1)

        plt.figure(figsize=(10, 5))
        plt.plot(t, T, label='Agua')
        plt.axhline(self.T_eq, ls='--', c='r', label=f"T_eq = {self.T_eq} °C")
        plt.xlabel('t (s)')
        plt.ylabel('T (°C)')
        plt.grid()
        plt.legend()
        plt.savefig('termograma_agua.png', dpi=300)
        plt.show()

    def ajuste_lineal(self, t, T):
        mask = (T >= self.T_eq - self.delta_lineal) & (T <= self.T_eq + self.delta_lineal)
        t_int, T_int = t[mask], T[mask]
        coeffs = np.polyfit(t_int, T_int, 1)
        poly = np.poly1d(coeffs)
        pendiente = ufloat(coeffs[0], np.std(T_int - poly(t_int)) / np.std(t_int) / np.sqrt(len(t_int)))
        print(f'dTdt lineal = {pendiente:.4f} ºC/s')
        return {
            'pendiente': pendiente,
            'ordenada': coeffs[1],
            't_intervalo': t_int,
            'T_intervalo': T_int,
            'funcion': poly
        }

    def ajuste_exponencial(self, t, T):
        def exp_func(t, a, b, c):
            return a - b * np.exp(-c * t)

        mask = (T >= self.T_eq - self.delta_exp) & (T <= self.T_eq + self.delta_exp)
        t_int, T_int = t[mask], T[mask]

        try:
            p0 = [self.T_eq, self.delta_exp, 0.01]
            popt, pcov = curve_fit(exp_func, t_int, T_int, p0=p0)
            perr = np.sqrt(np.diag(pcov))

            T_inf = ufloat(popt[0], perr[0])
            B = ufloat(popt[1], perr[1])
            C = ufloat(popt[2], perr[2])

            teq = (1 / C) * unumpy.log(B / (T_inf - self.T_eq))
            dTdt = B * C * unumpy.exp(-C * teq)

            print(f'dTdt exponencial = {dTdt:.4f} ºC/s')

            return {
                'T_inf_A': T_inf,
                'Amplitud_B': B,
                'Tasa_decaimiento_C': C,
                'funcion': lambda t: exp_func(t, *popt),
                't_intervalo': t_int,
                'T_intervalo': T_int,
                'dTdt_eq': dTdt
            }

        except Exception as e:
            return None

    def analizar(self):
        self.calcular_T_eq()
        pendientes = []
        dTdt_exp_eq_list = []

        for muestra in self.muestras:
            idx_min = np.argmin(muestra.T_CH1)
            t_base = muestra.timestamp[idx_min]
            t_alineado = np.array([(ti - t_base).total_seconds() for ti in muestra.timestamp])
            t_pos, T_pos = t_alineado[t_alineado >= 0], muestra.T_CH1[t_alineado >= 0]

            self.muestras_procesadas.append((os.path.split(muestra.filename), t_pos, T_pos))

            if self.hacer_ajuste_lineal:
                res_lin = self.ajuste_lineal(t_pos, T_pos)
                self.resultados_lineales.append(res_lin)
                pendientes.append(res_lin['pendiente'])

            if self.hacer_ajuste_exponencial:
                res_exp = self.ajuste_exponencial(t_pos, T_pos)
                if res_exp:
                    self.resultados_exp.append(res_exp)
                    dTdt_exp_eq_list.append(res_exp['dTdt_eq'])

        # Plot de temperaturas
        plt.figure(figsize=(10, 5))
        for nombre, t, T in self.muestras_procesadas:
            plt.plot(t, T, label=nombre)
        plt.axhline(self.T_eq, ls='--', c='k', label=f"T$_eq$ = {self.T_eq} °C")
        plt.xlabel("t (s)")
        plt.ylabel("T (°C)")
        plt.legend(ncol=2, loc='lower right')
        plt.grid()
        plt.xlim(0,)
        plt.savefig('T_vs_t_all.png', dpi=300)
        plt.show()

        # Ajustes lineales y exponenciales
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), constrained_layout=True, sharex=True)

        # Lineal
        if self.hacer_ajuste_lineal:
            for i, (nombre, t, T) in enumerate(self.muestras_procesadas):
                ax1.plot(t, T, '.-', label=nombre)
                ax1.plot(self.resultados_lineales[i]['t_intervalo'],
                         self.resultados_lineales[i]['funcion'](self.resultados_lineales[i]['t_intervalo']), '-')

            valores_lin = [p.n for p in pendientes]
            prom_lin = sum(valores_lin) / len(valores_lin)
            std_lin = (sum((x - prom_lin)**2 for x in valores_lin) / len(valores_lin))**0.5
            prom_lineal = ufloat(prom_lin, std_lin)

            ax1.text(0.8, 0.3, f"<dT/dt> = {prom_lineal:.4f} °C/s", transform=ax1.transAxes,
                     fontsize=12, ha='center', va='center', bbox=dict(boxstyle="round", facecolor='tab:green', alpha=0.5))
        else:
            for nombre, t, T in self.muestras_procesadas:
                ax1.plot(t, T, '.-', label=nombre)

        ax1.set_title("Ajuste lineal", loc='left')
        ax1.axhline(self.T_eq, ls='--', c='k', label=f"T$_eq$ = {self.T_eq} °C")
        ax1.set_ylabel("T (°C)")
        ax1.grid()
        ax1.legend(ncol=2, loc='lower right')

        # Exponencial
        if self.hacer_ajuste_exponencial:
            for i, (nombre, t, T) in enumerate(self.muestras_procesadas):
                ax2.plot(t, T, '.-', label=nombre)
                if i < len(self.resultados_exp):
                    t_exp = np.linspace(t.min(), t.max(), 100)
                    ax2.plot(t_exp, self.resultados_exp[i]['funcion'](t_exp), '-')

            if dTdt_exp_eq_list:
                valores_exp = [v.n for v in dTdt_exp_eq_list]
                prom_exp = sum(valores_exp) / len(valores_exp)
                std_exp = (sum((x - prom_exp)**2 for x in valores_exp) / len(valores_exp))**0.5
                prom_exponencial = ufloat(prom_exp, std_exp)

                ax2.text(0.8, 0.3, f"<dT/dt> = {prom_exponencial:.4f} °C/s", transform=ax2.transAxes,
                         fontsize=12, ha='center', va='center', bbox=dict(boxstyle="round", facecolor='tab:green', alpha=0.5))
        else:
            for nombre, t, T in self.muestras_procesadas:
                ax2.plot(t, T, '.-', label=nombre)

        ax2.set_title("Ajuste exponencial", loc='left')
        ax2.set_xlabel("t (s)")
        ax2.set_ylabel("T (°C)")
        ax2.grid()
        ax2.legend(ncol=2, loc='lower right')
        ax2.set_xlim(0,)
        plt.savefig('ajustes_lineales_exponenciales.png', dpi=300)
        plt.show()

        # Prints finales
        print('-'*50)
        print(f'Concentracion {self.concentracion} g/L\nCap. calorifica volumetrica = 4.186 kJ/ºC/L\n')

        if self.hacer_ajuste_lineal and pendientes:
            print(f'<dTdt> lineal = {prom_lineal:.4f} ºC/s')
            CSAR_lineal = prom_lineal * 4.186e3 / self.concentracion
            print(f'CSAR = {CSAR_lineal:.0f} W/g (ajuste lineal)')

        if self.hacer_ajuste_exponencial and dTdt_exp_eq_list:
            print(f'\n<dTdt> exponencial = {prom_exponencial:.4f} ºC/s')
            CSAR_exp = prom_exponencial * 4.186e3 / self.concentracion
            print(f'CSAR = {CSAR_exp:.0f} W/g (ajuste exponencial)')

        print('-'*50)
#%%
if __name__ == '__main__':
    at = AnalisisTermico(path='data', delta_lineal=1.0, delta_exp=3.0, t_eq_min=1000,
                          concentracion=(8.2, 0.4), hacer_ajuste_lineal=False, hacer_ajuste_exponencial=False)
    at.analizar()
# %%
