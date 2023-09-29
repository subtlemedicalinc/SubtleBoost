# -*- mode: python -*-
from PyInstaller.utils.hooks import collect_submodules
import sys ; sys.setrecursionlimit(sys.getrecursionlimit() * 10)

block_cipher = None


a = Analysis(['infer.py'],
             pathex=['/project_dir/app', '.'],
             binaries=[],
             datas=[('/usr/local/lib/python3.10/site-packages/tensorflow', 'tensorflow'), ('/usr/local/lib/python3.10/site-packages/HD_BET', 'HD_BET')],
             hiddenimports=['pywt._extensions._cwt', 'wrapt', 'absl.*', 'gast', 'astor', 'termcolor', 'opt_einsum', 'google.protobuf.json_format', 'google.protobuf.wrappers_pb2', *collect_submodules('pydicom'), *collect_submodules('torch')],
             hookspath=['hooks'],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

b = Analysis(['validate_config.py'],
             pathex=['/project_dir/app', '.'],
             binaries=[],
             datas=[('/usr/local/lib/python3.10/site-packages/tensorflow', 'tensorflow'), ('/usr/local/lib/python3.10/site-packages/HD_BET', 'HD_BET')],
             hiddenimports=['pywt._extensions._cwt', 'wrapt', 'absl.*', 'gast', 'astor', 'termcolor', 'opt_einsum', 'google.protobuf.json_format', 'google.protobuf.wrappers_pb2', *collect_submodules('pydicom'), *collect_submodules('torch')],
             hookspath=['hooks'],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
a.binaries = a.binaries - TOC([('libcuda.so.1', None, None)])
b.binaries = b.binaries - TOC([('libcuda.so.1', None, None)])

MERGE((a, 'infer', 'infer'), (b, 'validate_config', 'validate_config'))

a_pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
b_pyz = PYZ(b.pure, b.zipped_data,
             cipher=block_cipher)
a_exe = EXE(a_pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='infer',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
b_exe = EXE(b_pyz,
          b.scripts,
          [],
          exclude_binaries=True,
          name='validate_config',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(a_exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               b_exe,
               b.binaries,
               b.zipfiles,
               b.datas,
               strip=False,
               upx=True,
               name='infer')
