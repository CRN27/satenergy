import requests, urllib3, re, pandas as pd
from io import StringIO
urllib3.disable_warnings()
URL = 'https://dataserver-coids.inpe.br/queimadas/queimadas/focos/csv/diario/Brasil/'
r = requests.get(URL, verify=False, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
arqs = sorted(set(re.findall(r'href="(focos_diario_br_\d+\.csv)"', r.text)))
print(f"Arquivos: {len(arqs)}, Último: {arqs[-1]}")
r2 = requests.get(f'{URL}{arqs[-1]}', verify=False, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
df = pd.read_csv(StringIO(r2.text))
print('Colunas:', list(df.columns))
print(df.head(3))