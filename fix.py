with open(r'C:\Users\Christian Nielsen\Documents\satenergy\dashboard\app.py', 'r', encoding='utf-8') as f:
    c = f.read()

old = '_alert_sent_cache.clear()'
if old in c:
    print('Versao nova ja instalada')
else:
    print('Versao antiga - precisa substituir')
    
print('Total linhas:', c.count('\n'))
