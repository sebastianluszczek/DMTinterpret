import pandas as pd
import numpy as np

class DMT(object):
    'Klasa operująca na danych pomiarowych DMT'

    def __init__(self, plik):
        df = pd.read_csv(plik, delimiter=',', decimal=',')
        self.df = df
        print('DataFrame "{}" został zaimportowany do klasy\n'.format(plik))
        print(self.df.columns, '\n')

    def pokaz(self, wiersze=10):
        print(self.df.head(wiersze), '\n')

    def norm(self, dA, dB):
        self.df['p0']=1.05*(self.df['A ']+dA)-0.05*(self.df['B']-dB)
        self.df['p1']=self.df['B']-dB

    def reduction(self):
        self.df['ED'] = 34.7*(self.df['p1']-self.df['p0'])

    def preins_pore_p(self, wlvl):
        wlvl_mask = self.df['Depth (m)'] >= wlvl
        self.df['u0'] = (self.df['Depth (m)'] - wlvl) * 0.098
        self.df['u0'] = self.df['u0'][wlvl_mask]
        self.df = self.df.fillna(0.0)

    def mat_index(self):
        self.df['ID'] = (self.df['p1'] - self.df['p0'])/(self.df['p0'] - self.df['u0'])

    def dil_modul(self):
        self.df['ED'] = 34.7 * (self.df['p1'] - self.df['p0'])

    def unit_weight(self):
        'funkcja określająca ciężar wlasciwy [...]'
        Am, An, Bm, Bn, Cm, Cn, Dm, Dn = 0.585, 1.737, 0.621, 2.013, 0.657, 2.289, 0.694, 2.564

        def uw(ID,ED):
            if ID < 0.6:
                if ED >= 10 ** (Dn + Dm * np.log10(ID)):
                    return 2.05
                elif ED >= 10 ** (Cn + Cm * np.log10(ID)):
                    return 1.9
                elif ED >= 10 ** (Bn + Bm * np.log10(ID)):
                    return 1.8
                elif ED >= 10 ** (An + Am * np.log10(ID)):
                    return 1.7
                else:
                    return 1.6
            elif ID < 1.8:
                if ED >= 10 ** (Dn + Dm * np.log10(ID)):
                    return 2.1
                elif ED >= 10 ** (Cn + Cm * np.log10(ID)):
                    return 1.95
                elif ED >= 10 ** (Bn + Bm * np.log10(ID)):
                    return 1.8
                elif ED >= 10 ** (An + Am * np.log10(ID)):
                    return 1.7
                else:
                    return 1.6
            else:
                if ED >= 10 ** (Dn + Dm * np.log10(ID)):
                    return 2.15
                elif ED >= 10 ** (Cn + Cm * np.log10(ID)):
                    return 2.0
                elif ED >= 10 ** (Bn + Bm * np.log10(ID)):
                    return 1.9
                elif ED >= 10 ** (An + Am * np.log10(ID)):
                    return 1.8
                else:
                    return 1.7

        self.df['gamma'] = self.df.apply(lambda x: uw(x['ID'], x['ED']), axis = 1)

    def sigma(self):
        self.df['sigma'] = self.df['gamma'].cumsum()*0.02
        self.df['sigma_v0'] = self.df['gamma'].cumsum() * 0.02 - self.df['u0']

    def hor_stress_ind(self):
        self.df['KD'] = (self.df['p0'] - self.df['u0']) / self.df['sigma_v0']

Niep = DMT('NiepDMT.csv')
Niep.pokaz()
Niep.norm(0.1, 0.45)
Niep.reduction()
Niep.preins_pore_p(0.5)
Niep.mat_index()
Niep.dil_modul()
Niep.unit_weight()
Niep.sigma()
Niep.hor_stress_ind()
Niep.pokaz()
