# Установка LaTeX для VS Code в Ubuntu

## Основной этап

1. **Установка TinyTeX** (LaTeX-дистрибутив без root-прав):
    ```bash
    wget -qO- "https://yihui.org/tinytex/install-bin-unix.sh" | sh
    ```

2. **Добавление в PATH** (следуйте инструкции с [официального FAQ](https://yihui.org/tinytex/faq/#faq-7)):
    ```bash
    # Добавьте в ваш ~/.bashrc или ~/.zshrc
    export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
    ```

3. **Перезагрузите терминал** или выполните:
    ```bash
    source ~/.bashrc  # или source ~/.zshrc
    ```

## Установка дополнительных пакетов LaTeX

Установите необходимые пакеты командой:

```bash
tlmgr install collection-langcyrillic babel-russian
tlmgr install algorithm2e algorithmicx
tlmgr install enumitem cm-super
tlmgr install mathtools caption subcaption
tlmgr install wrapfig pgf
```

## Установка дополнительных инструментов

1. **Установка tex-fmt** (форматирование LaTeX):
    ```bash
    curl -L https://github.com/WGUNDERWOOD/tex-fmt/releases/download/v0.5.4/tex-fmt-x86_64-linux.tar.gz | sudo tar -xz -C /usr/local/bin
    ```

2. **Установка chktex** (проверка синтаксиса LaTeX):
    ```bash
    sudo apt install chktex
    ```

## Настройка VS Code

1. **Установите расширение LaTeX Workshop**:
    - Откройте VS Code
    - Перейдите в раздел расширений (Ctrl+Shift+X)
    - Найдите и установите "LaTeX Workshop" от James Yu

2. **Перезапустите VS Code** для применения изменений.

3. **Настройте LaTeX Workshop** (опционально):
    - Откройте настройки VS Code (Ctrl+,)
    - Найдите "LaTeX Workshop"
    - Настройте параметры, например, путь к `tex-fmt` или `chktex`

# Общие настройки

- Длина строки: 120 символов  
- tab: 2 пробела  
- Максимальный размер файла: 15 Мб
