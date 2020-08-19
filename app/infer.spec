# -*- mode: python -*-

block_cipher = None


a = Analysis(['infer.py'],
             pathex=['/project_dir/app', '.'],
             binaries=[],
             datas=[
                ('/usr/lib/python3.5/site-packages/tensorflow/contrib', 'tensorflow/contrib'),
                ('/usr/lib/python3.5/site-packages/deepbrain', 'deepbrain')
             ],
             hiddenimports=['tensorflow', 'tensorflow.contrib', 'tensorrt', 'pkg_resources', 'pkg_resources.py2_warn', 'pywt._extensions._cwt', 'gpustat'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
a.binaries = a.binaries - TOC([('libcuda.so.1', None, None)])
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='infer',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=True )
