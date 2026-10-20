from utils import constants
keys = list(constants.SESSION_KEYS)
s = open('services/state_service.py').read()
start = s.find('_DEFAULTS')
eq = s.find('=', start)
op = s.find('{', eq)
depth = 0
i = op
end = op
while i < len(s):
    if s[i] == '{':
        depth += 1
    elif s[i] == '}':
        depth -= 1
        if depth == 0:
            end = i+1
            break
    i += 1
defs = []
for k in keys:
    if k in ['raw_data','clean_data','customer_features','rfm_results','lrfm_results','selected_segmentation_model','segmented_customers','churn_model','churn_scores']:
        defs.append(f'    "{k}": None')
    elif k in ['segmentation_runs']:
        defs.append(f'    "{k}": []')
    else:
        defs.append(f'    "{k}": {{}}')
newd = '{\n' + ',\n'.join(defs) + ',\n}'
open('services/state_service.py','w').write(s[:start]+newd+s[end:])
print('ok')
