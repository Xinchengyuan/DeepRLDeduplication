# import subprocess

"""
path = '/Users/test/Documents/SegDedup/fs-hasher-0.9.5/hf-stat '
file = '/Users/test/Documents/SegDedup/data/user5/fslhomes-user005-2012-03-09.2kb.hash.anon'

cmd = path + '-h -w -f ' + file
out = subprocess.check_output(cmd, shell=True, text=True)
out = "".join([s for s in out.strip().splitlines(True) if s.strip()])
lns = out.split('\n')
print(lns[31])
"""
"""import os

path = '/Users/test/Documents/SegDedup/data/user5/'
files = sorted(os.listdir(path))

print(files)"""
"""
a = [1,2,3,4,5,6]
while a[0] < 5:
    del(a[0])
print(a)
"""
test = "0x9ba186"
hexn = hex(int(test, 16))
print (hexn)

