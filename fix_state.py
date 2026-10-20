s = open('services/state_service.py').read()
# Remove segment_history_labels line
old = '    "segment_history_labels": None,\n'
if old in s:
    s2 = s.replace(old, '')
    open('services/state_service.py','w').write(s2)
    print('ok')
else:
    print('no')
