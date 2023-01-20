'''
Remove the "Lightning" group of NetCDF file

Simpler method: ncks -O -x -C -g .?Lightning in.nc out.nc
'''

import os
from netCDF4 import Dataset

filename = '../data/lno2/S5P_LNO2_lifetime.nc'
rootGrp = Dataset(filename)

# get the group names of Cases
cases = list(sorted(rootGrp.groups.keys()))

grps = []
for case in cases:
    swaths = list(rootGrp[case].groups.keys())
    grp = [case+'/'+swath+'/Lightning' for swath in swaths]
    grps.extend(grp)

savename = filename.replace('.nc', '_subset.nc')
print(f'Deleted the "Lightning" subgroups and saved to {savename}')
os.system(f"ncks -O -x -C -g {','.join(grps)} {filename} {savename}")