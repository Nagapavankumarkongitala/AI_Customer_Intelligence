s = open('services/dashboard_service.py').read()
old = "    df = pd.DataFrame(moves, columns=TRANSITION_COLUMNS)\n    if not df.empty and df['period_from'].nunique() == 1 and df['period_to'].nunique() == 1 and len(df) > 1:\n        # For compatibility with tests expecting single transition row, keep as-is but some tests check list equality\n        pass\n    return df"
new = "    df = pd.DataFrame(moves, columns=TRANSITION_COLUMNS)\n    return df"
if old not in s:
    print('no')
else:
    open('services/dashboard_service.py','w').write(s.replace(old,new))
    print('ok')
