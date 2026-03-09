# Sistema de Classificação de Tickets — POC

Este projeto implementa um sistema automático de classificação de tickets projetado para priorização, roteamento automático e identificação de criticidade em tickets de suporte ao cliente. O sistema utiliza um conjunto de dados de tickets para treinar um modelo de machine learning capaz de classificar tickets recebidos com base no seu conteúdo.

## Estrutura do Projeto

- **data/**: Contém o conjunto de dados usado para treinamento e testes do sistema de classificação.
  - `knowledge_base.json`: O dataset com detalhes dos tickets incluindo ticket_id, customer_segment, channel, text, category e priority.
  
- **src/**: Contém o código-fonte da aplicação.
  - `main.py`: Ponto de entrada da aplicação, orquestrando os processos de treinamento e inferência.
  - `pipeline.py`: Define o pipeline de processamento de dados para carregar e pré-processar os dados.
  - `trainer.py`: Contém a lógica para treinar o modelo de classificação.
  - `inferencer.py`: Lida com o processo de inferência para classificar novos tickets.
  - `api/`: Contém a configuração do servidor web.
    - `server.py`: Expõe endpoints para classificação de tickets.
  - `preprocessing.py`: Inclui funções de pré-processamento de texto.
  - `types/`: Define esquemas e tipos de dados.
    - `schemas.py`: Contém modelos de request e response para a API.

- **notebooks/**: Contém notebooks Jupyter para análise exploratória de dados.
  - `exploration.ipynb`: Usado para visualizações e insights sobre o dataset de tickets.

- **tests/**: Contém testes unitários para o sistema de classificação.
  - `test_classification.py`: Assegura que a lógica de classificação funciona conforme esperado.

- **lmproject.yaml**: Configurações do projeto LM Studio.# PowerShell / terminal do VS Code

- **requirements.txt**: Lista de dependências Python necessárias para o projeto.

- **.gitignore**: Especifica arquivos e diretórios a serem ignorados pelo versionamento.

## Instruções de Configuração

1. Clone o repositório para sua máquina local.
2. Navegue até o diretório do projeto.
3. Instale as dependências necessárias usando:
   ```
   pip install -r requirements.txt
   ```
4. Execute a aplicação com:
   ```
   python src/main.py
   ```

## Instruções rápidas (Windows PowerShell)
```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3 -m pip install --upgrade pip
py -3 -m pip install -r requirements.txt

# Treinar e salvar modelo (Logistic Regression):
py -3 .\scripts\train_and_save.py

# Rodar a API e frontend:
py -3 -m uvicorn src.api.server:app --reload --host 127.0.0.1 --port 8000
# abrir http://127.0.0.1:8000
```

## Guia de Uso

- Envie tickets através dos endpoints da API expostos pelo servidor para receber classificações.
- Explore o notebook Jupyter para obter insights e visualizações relacionados ao dataset.

## Visão Geral

Este sistema de classificação de tickets tem como objetivo otimizar operações de suporte ao cliente ao categorizar e priorizar automaticamente os tickets, permitindo um atendimento mais eficiente.