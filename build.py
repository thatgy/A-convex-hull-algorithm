from pathlib import Path
import os,sys,subprocess
here=Path(__file__).resolve().parent
name='_hull_scalable.dll' if os.name=='nt' else '_hull_scalable.dylib' if sys.platform=='darwin' else '_hull_scalable.so'
cxx=os.environ.get('CXX','g++')
flags=os.environ.get('CXXFLAGS','-O3 -march=native -flto -DNDEBUG').split()
cmd=[cxx,'-std=c++20',*flags,'-pthread','-shared',*([] if os.name=='nt' else ['-fPIC']),str(here/'hull_c_api.cpp'),'-o',str(here/name)]
print(' '.join(cmd),flush=True)
subprocess.run(cmd,check=True)
