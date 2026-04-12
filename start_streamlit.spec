# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['start_streamlit.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'PIL._tkinter_finder',
        'altair.vegalite.v5.api',
        'streamlit.runtime.scriptrunner.magic_funcs',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='TaxiStreamlit',
    debug=False,
    bootloader_ignore_signals=False,
    strip_debug=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip_debug=False,
    upx=True,
    upx_exclude=[],
    name='TaxiStreamlit',
)