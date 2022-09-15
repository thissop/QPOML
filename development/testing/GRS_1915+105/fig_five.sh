# good obs 
#91701-01-56-00,0.022759999999999447,2.13176,2.1090000000000004,0.17954999999999996,0.072,14.186560000000005,14.593

heainit
cd /mnt/c/Users/Research/Documents/GitHub/MAXI-J1535/final-push/data/sources/GRS_1915+105/qpo/external_PDS/91701-01-56-00

xspec
chatter 0
data power_0_249.pha

ignore **-1.0 10.0-**

model loren
0


freeze 1 
query yes 
fit

editmod loren+loren 
2.13176
0.17954999999999996
14.186560000000005

freeze 1-3

fit

cpd /ps
setplot energy
plot data
iplot 
label bottom Frequency (Hz)
label left Power (Leahy Normalized)
label top
font roman
t off
View 0.12 0.12
csize 2.0
rescale x 1. 15.
hard 91701-01-56-00[predicted-pds-data].ps/ps
quit 
plot model 
iplot
label top
font roman
t off
label X Frequency (Hz)
label Y Power (Leahy Normalized)
View 0.12 0.12
csize 2.0
rescale x 1. 15.
hard 91701-01-56-00[predicted-pds-model].ps/ps
quit
quit
y

ps2pdf 91701-01-56-00[predicted-pds-data].ps 91701-01-56-00[predicted-pds-data].pdf
ps2pdf 91701-01-56-00[predicted-pds-model].ps 91701-01-56-00[predicted-pds-model].pdf

python 
import os
import shutil  
os.remove('91701-01-56-00[predicted-pds-data].ps')
os.remove('91701-01-56-00[predicted-pds-model].ps')
fig_five_root = '/mnt/c/Users/Research/Documents/GitHub/MAXI-J1535/manuscript/figures/figure_five/'

for f in ['91701-01-56-00[predicted-pds-data].pdf', '91701-01-56-00[predicted-pds-model].pdf']:
    shutil.copyfile(f, f'{fig_five_root}{f}')
    os.remove(f)

quit()

# bad obs 
#80701-01-23-02,0.3066200000000001,2.5603800000000003,2.8670000000000004,0.26205,0.384,12.978379999999998,12.859

cd /mnt/c/Users/Research/Documents/GitHub/MAXI-J1535/final-push/data/sources/GRS_1915+105/qpo/external_PDS/80701-01-23-02

xspec
chatter 0
data power_0_249.pha

ignore **-1.0 10.0-**

model loren
0


freeze 1 
query yes 
fit

editmod loren+loren 
2.5603800000000003
0.26205
12.978379999999998

freeze 1-3

fit

cpd /ps
setplot energy
plot data
iplot 
label bottom Frequency (Hz)
label left Power (Leahy Normalized)
label top
font roman
t off
View 0.12 0.12
csize 2.0
rescale x 1. 15.
hard 80701-01-23-02[predicted-pds-data].ps/ps
quit 
plot model 
iplot
label top
font roman
t off
label X Frequency (Hz)
label Y Power (Leahy Normalized)
View 0.12 0.12
csize 2.0
rescale x 1. 15.
hard 80701-01-23-02[predicted-pds-model].ps/ps
quit
quit
y

ps2pdf 80701-01-23-02[predicted-pds-data].ps 80701-01-23-02[predicted-pds-data].pdf
ps2pdf 80701-01-23-02[predicted-pds-model].ps 80701-01-23-02[predicted-pds-model].pdf

python 
import os
import shutil  
os.remove('80701-01-23-02[predicted-pds-data].ps')
os.remove('80701-01-23-02[predicted-pds-model].ps')
fig_five_root = '/mnt/c/Users/Research/Documents/GitHub/MAXI-J1535/manuscript/figures/figure_five/'

for f in ['80701-01-23-02[predicted-pds-data].pdf', '80701-01-23-02[predicted-pds-model].pdf']:
    shutil.copyfile(f, f'{fig_five_root}{f}')
    os.remove(f)

quit()

# worst obs
# 90701-01-48-00,1.6607600000000013,3.482760000000001,1.8219999999999998,0.46360999999999986,0.19000000000000003,10.355650000000002,12.917
cd /mnt/c/Users/Research/Documents/GitHub/MAXI-J1535/final-push/data/sources/GRS_1915+105/qpo/external_PDS/90701-01-48-00

xspec
chatter 0
data power_0_249.pha

ignore **-1.0 10.0-**

model loren
0


freeze 1 
query yes 
fit

editmod loren+loren 
3.482760000000001
0.46360999999999986
10.355650000000002

freeze 1-3
fit

chatter 10
show param

cpd /ps
setplot energy
plot data
iplot 
label bottom Frequency (Hz)
label left Power (Leahy Normalized)
label top
font roman
t off
View 0.12 0.12
csize 2.0
rescale x 1. 15.
hard 90701-01-48-00[predicted-pds-data].ps/ps
quit 
plot model 
iplot
label top
font roman
t off
label X Frequency (Hz)
label Y Power (Leahy Normalized)
View 0.12 0.12
csize 2.0
rescale x 1. 15.
hard 90701-01-48-00[predicted-pds-model].ps/ps
quit
quit
y

ps2pdf 90701-01-48-00[predicted-pds-data].ps 90701-01-48-00[predicted-pds-data].pdf
ps2pdf 90701-01-48-00[predicted-pds-model].ps 90701-01-48-00[predicted-pds-model].pdf

python 
import os
import shutil  
os.remove('90701-01-48-00[predicted-pds-data].ps')
os.remove('90701-01-48-00[predicted-pds-model].ps')
fig_five_root = '/mnt/c/Users/Research/Documents/GitHub/MAXI-J1535/manuscript/figures/figure_five/'

for f in ['90701-01-48-00[predicted-pds-data].pdf', '90701-01-48-00[predicted-pds-model].pdf']:
    shutil.copyfile(f, f'{fig_five_root}{f}')
    os.remove(f)

quit()

