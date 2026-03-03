# Professor Thiago Santos - UFOP, Brasil
# Utilitário para limpeza recursiva de pastas __pycache__

import os
import shutil
import sys
from pathlib import Path

def limpar_pycache(diretorio_raiz: str):
    """
    Percorre o diretório informado e remove recursivamente todas as pastas '__pycache__'.
    """
    path_raiz = Path(diretorio_raiz).resolve()
    print(f"--- Iniciando limpeza em: {path_raiz} ---")
    
    total_removido = 0
    erros = 0

    # rglob busca recursivamente por arquivos/pastas que combinem com o padrão
    for item in path_raiz.rglob('__pycache__'):
        if item.is_dir() and item.name == '__pycache__':
            try:
                print(f"Removendo: {item}")
                shutil.rmtree(item)
                total_removido += 1
            except Exception as e:
                print(f"Erro ao remover {item}: {e}")
                erros += 1

    print(f"\n--- Limpeza concluída! ---")
    print(f"Pastas removidas: {total_removido}")
    if erros > 0:
        print(f"Número de falhas: {erros}")

if __name__ == "__main__":
    # Define o diretório de execução como a raiz do projeto (onde o script está ou CWD)
    diretorio_alvo = os.getcwd()
    
    # Se houver um argumento na linha de comando, usa ele como diretório
    if len(sys.argv) > 1:
        diretorio_alvo = sys.argv[1]
        
    limpar_pycache(diretorio_alvo)
