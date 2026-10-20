from utils import constants
keys = list(constants.SESSION_KEYS)
if len(keys) > 20:
    keys = [k for k in keys if k != 'segment_history_labels']
s = open('utils/constants.py').read()
start = s.find('SESSION_KEYS')
eq = s.find('=', start)
op = s.find('(', eq)
depth = 0
i = op
end = op
while i < len(s):
    if s[i] == '(':
        depth += 1
    elif s[i] == ')':
        depth -= 1
        if depth == 0:
            end = i + 1
            break
    i += 1
new = 'SESSION_KEYS = (\n    "' + '",\n    "'.join(keys) + '",\n)'
open('utils/constants.py','w').write(s[:start] + new + s[end:])
print(len(keys))
