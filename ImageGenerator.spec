# -*- mode: python ; coding: utf-8 -*-

# ---НАЧАЛО ИЗМЕНЕНИЙ---

a = Analysis(
    ['image_generator_app.py'],
    pathex=[],
    binaries=[],
    # Добавляем каждый файл ресурса в корень сборки.
    # Точка '.' означает корневую директорию внутри .exe.
    datas=[
        ('color1.ttf', '.'),
        ('color2.otf', '.'),
        ('color3.otf', '.'),
        ('font1.ttf', '.'),
        ('font2.ttf', '.'),
        ('font3.ttf', '.'),
        ('font4.ttf', '.'),
        ('font5.ttf', '.'),
        ('font6.ttf', '.'),
        ('icon.ico', '.'),
        ('tutorial.png', '.')
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    onefile=True,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher_block_size=16,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ImageGenerator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # Указываем путь к иконке, которая теперь будет в корне.
    icon='icon.ico',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ImageGenerator',
)
# ---КОНЕЦ ИЗМЕНЕНИЙ---