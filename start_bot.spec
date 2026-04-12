# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['start_bot.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('.env', '.'),
    ],
    hiddenimports=[
        'PIL._tkinter_finder',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'pydantic',
        'pydantic.v2',
        'cryptography',
    ],
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
    name='TaxiBot',
    debug=False,
    bootloader_ignore_signals=False,
    strip_debug=False,
    upx=True,
    console=True,
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
    name='TaxiBot',
)