# ================================================================
#  GERADOR DE BASE SINTÉTICA DE TRANSACOES PIX — COMENTADO
#  Objetivo: criar uma base JSON com usuários e histórico de PIX
#  Explicações detalhadas em cada etapa, sem mudar a lógica original
# ================================================================

import json          # Biblioteca padrão para ler/gravar arquivos JSON
import random        # Geração de números aleatórios (escolhas, inteiros, floats)
from datetime import datetime, timedelta  # Manipulação de datas (criação e transações)
import numpy as np   # NumPy para estatística (distribuição Gama e quantis)
import pandas as pd  # Pandas (não é usado diretamente aqui, mas útil para inspeções futuras)

# ================================================================
#  CONFIGURAÇÕES GERAIS (CONSTANTES E CATÁLOGOS)
# ================================================================

# Quantidade de contas (usuários) a serem criadas
N_CONTAS = 50

# Quantidade de transações que cada usuário tentará realizar
# Obs.: "tentará" porque pode haver transações descartadas se excederem o saldo
TRANS_POR_CONTA = 100

# Lista de bancos simulados (apenas rótulos; sem integração real)
BANCOS = ["Banco do Brasil", "Caixa Econômica", "Itaú", "Bradesco", "Nubank", "Santander", "Inter", "BTG Pactual"]

# Lista de localidades do Distrito Federal (cidades/regiões administrativas)
# Essas localidades são usadas como "local_origem" para as transações e o "domicílio" do usuário
CIDADES = [
    "Luziânia", "Brasília (Plano Piloto)", "Ceilândia", "Taguatinga", "Samambaia", "Águas Claras", "Guará",
    "Gama", "Planaltina", "Sobradinho", "Sobradinho II", "Brazlândia", "Santa Maria",
    "São Sebastião", "Paranoá", "Itapoã", "Recanto das Emas", "Riacho Fundo I",
    "Riacho Fundo II", "Núcleo Bandeirante", "Candangolândia", "Cruzeiro", "Sudoeste/Octogonal",
    "Lago Sul", "Lago Norte", "Park Way", "SIA", "SCIA/Estrutural", "Varjão",
    "Jardim Botânico", "Vicente Pires", "Fercal", "Sol Nascente/Pôr do Sol", "Arniqueira"
]

# IMPORTANTE / IDEIA FUTURA:
# Da mesma forma que um usuário tem horários e valores preferidos,
# ele também poderia ter 2+ "locais de costume" (ex.: casa, trabalho, academia).
# Aqui mantemos um TODO para evolução futura:
# - Em uma aplicação real, isso viria de uma função que captura localização real (GPS/IP).
# - Neste gerador, poderíamos selecionar 2-3 CIDADES por usuário para simular "locais recorrentes".

# Tipos de chave Pix possíveis — a "chave_pix" é gerada em função do tipo
TIPOS_CHAVE = ["cpf", "email", "telefone", "aleatoria"]

# Canais disponíveis para execução das transações (aplicativo vs. internet banking web)
CANAIS = ["mobile", "internet banking"]

# Catálogo de descrições de pagamentos — rótulos para o campo "descricao_pagamento"
DESCRICOES_PIX = [
    "Pagamento de serviço", "Transferência entre contas", "Aluguel",
    "Pagamento de compra", "Empréstimo", "Repasse familiar",
    "Pix para amigo", "Pix mercado"
]

# ================================================================
#  FUNÇÕES PARA MONTAR O USUÁRIO (DADOS ESTÁTICOS / PERFIL ÚNICO)
# ================================================================

def gerar_chave_pix(tipo, nome):
    """
    Gera uma chave PIX compatível com o 'tipo' escolhido.
    - Se 'email': monta "nome@mail.com" (apenas exemplo; não é e-mail real).
    - Se 'telefone': gera um número no padrão +55 11 9XXXX-XXXX (intervalo simulado).
    - Se 'cpf': gera uma sequência numérica de 11 dígitos (não é CPF válido).
    - Se 'aleatoria': gera um hex aleatório de 32 caracteres.
    """
    if tipo == "email":
        return f"{nome.lower()}@mail.com"
    elif tipo == "telefone":
        # randint aqui pega um número grande dentro do DDD 11 (exemplo),
        # simulando telefones celulares. NÃO garante formato real.
        return f"+55{random.randint(11900000000, 11999999999)}"
    elif tipo == "cpf":
        # Gera um número de 11 dígitos (sem validação de dígitos verificadores)
        return f"{random.randint(10000000000, 99999999999)}"
    else:
        # Chave aleatória: 128 bits -> representação hexadecimal -> 32 chars
        return f"{random.getrandbits(128):032x}"[:32]

def gerar_data_criacao():
    """
    Define uma data de criação da conta entre 90 e 365 dias antes de 2024-01-01.
    - 'hoje' é fixado em 2024-01-01 para que todo o ano de 2024
      tenha transações (ver função gerar_data_transacao).
    """
    hoje = datetime(2024, 1, 1)
    # Intervalo de dias "atrás" aleatório: quanto tempo antes de 2024 a conta foi criada
    dias_atras = random.randint(90, 365)
    # Retorna no formato ISO 8601, ex.: "2023-04-15T10:30:00"
    return (hoje - timedelta(days=dias_atras)).isoformat()

def gerar_data_transacao(horas_pref):
    """
    Gera uma data/hora para uma transação:
    - Dia aleatório dentro do ano de 2024 (0..364 a partir de 2024-01-01).
    - 99% das vezes usa um 'hora' dentre os horários preferidos do usuário.
    - 1% das vezes usa um horário atípico (0..23 aleatório), simulando exceção.
    - Minuto e segundo são aleatórios (0..59).
    Retorna um objeto datetime.
    """
    # Dia do ano aleatório (0 = 1/jan, 364 ≈ 31/dez)
    dia_ano = random.randint(0, 364)
    data_base = datetime(2024, 1, 1) + timedelta(days=dia_ano)

    # Probabilidade de seguir os horários preferidos: 99%
    if random.random() < 0.99:
        # Escolhe um horário dentre os preferidos do usuário
        hora = random.choice(horas_pref)
    else:
        # 1%: horário fora do padrão, qualquer hora do dia
        hora = random.randint(0, 23)

    # Minuto e segundo aleatórios para dar naturalidade
    minuto, segundo = random.randint(0, 59), random.randint(0, 59)

    # Monta o datetime completo (data do dia_ano + hora/min/seg)
    return datetime(data_base.year, data_base.month, data_base.day, hora, minuto, segundo)

def gamma_params(mean, cv):
    """
    Converte 'média' (mean) e 'coeficiente de variação' (cv) para parâmetros
    'alpha' (forma) e 'theta' (escala) da distribuição Gama:
      - alpha = 1 / (cv^2)
      - theta = mean / alpha
    Esses parâmetros alimentam np.random.gamma(alpha, theta)
    para amostrar valores de transação realistas (positivos e assimétricos).
    """
    alpha = 1 / (cv ** 2)
    theta = mean / alpha
    return alpha, theta

# ================================================================
#  GERAR CONTAS E PERFIS (CADA USUÁRIO = COMPORTAMENTO ÚNICO)
# ================================================================











# 'usuarios' armazenará metadados de cada conta (para o arquivo JSON)
usuarios = []

# 'comportamentos' guarda parâmetros comportamentais por id_conta,
# reaproveitados durante a geração de transações.
comportamentos = {}

# Loop de criação de N_CONTAS usuários
for id_usuario in range(1, N_CONTAS + 1):
    # Identidade básica
    nome = f"Usuario_{id_usuario}"               # Nome sintético do usuário
    banco = random.choice(BANCOS)                # Banco aleatório da lista
    cidade = random.choice(CIDADES)              # Cidade (domicílio) do usuário
    tipo_chave = random.choice(TIPOS_CHAVE)      # Tipo de chave PIX (cpf/email/telefone/aleatória)
    chave_pix = gerar_chave_pix(tipo_chave, nome)  # Geração da chave em si

    # ---- Horários preferidos únicos por usuário ----
    # Seleciona um "bloco" coerente de horas (matutino/diurno/noturno/espalhado)
    bloco_horario = random.choice([
        range(6, 12),    # Janela mais matutina (06-11)
        range(11, 15),   # Meio do dia (11-14)
        range(17, 23),   # Final do dia/noite (17-22)
        range(7, 22)     # Janela ampla (07-21)
    ])




    # Dentro do bloco, escolhe entre 2 e 4 horários específicos
    horarios_pref = sorted(random.sample(list(bloco_horario), k=random.randint(2, 4)))
    #CAçar dados lek 




    # ---- Distribuição de valores por usuário ----
    # Define uma média de transação (R$) específica do usuário (30 a 2000)
    mean_valor = random.uniform(30, 2000)





    # Define quão dispersos são os valores (coeficiente de variação entre 0.2 e 1.0)
    cv_valor = random.uniform(0.2, 1.0)
    # Converte mean/cv em parâmetros da Gama
    alpha, theta = gamma_params(mean_valor, cv_valor)



    # ---- Canal preferido do usuário ----
    # Usa pesos 70/30 favorecendo 'mobile' (comportamento típico atual)
    canal_pref = random.choices(CANAIS, weights=[0.7, 0.3])[0]





    # ---- Probabilidades individuais por usuário ----
    # Chance de uma transação ter valor "atípico" (multiplicador 3..6) — 1% a 5%


    chance_anomalia_valor = random.uniform(0.01, 0.05)
    # Chance de a transação acontecer fora da cidade "domicílio" — 2% a 10% (viagem ou anomalia)
    chance_viagem = random.uniform(0.02, 0.1)





    # ---- Descrições favoritas desse usuário ----
    # Seleciona de 3 até todas as descrições possíveis, em ordem aleatória
    descricoes_pref = random.sample(DESCRICOES_PIX, k=random.randint(3, len(DESCRICOES_PIX)))

    # ---- Saldo inicial disponível para transações ao longo do ano ----
    saldo = round(random.uniform(5000, 100000), 2)

    # Montagem do dicionário do usuário que irá para o JSON
    usuario = {
        "id_conta": id_usuario,                       # Identificador único da conta
        "nome_titular": nome,                         # Nome sintético
        "banco": banco,                               # Banco
        "agencia": f"{random.randint(1000, 9999)}",   # Agência (quatro dígitos)
        "numero_conta": f"{random.randint(1000000, 9999999)}-{random.randint(0, 9)}",  # Conta com dígito
        "tipo_conta": random.choice(["corrente", "poupança"]),  # Tipo de conta
        "localizacao": cidade,                        # Cidade/base do usuário
        "tipo_chave_pix": tipo_chave,                 # Tipo da chave PIX
        "chave_pix": chave_pix,                       # Chave PIX gerada
        "data_criacao": gerar_data_criacao(),         # Data de criação da conta (antes de 2024)
        "saldo_inicial": saldo,                       # Saldo total de partida
        "horarios_preferidos": horarios_pref,         # Lista de horas "preferidas" para operar
        "perfil_valores": {"alpha": alpha, "theta": theta},  # Parâmetros da Gama (para valores)
        "canal_preferido": canal_pref                 # Canal preferido (mobile/web)
    }

    # Adiciona à lista global de usuários (para salvar no JSON)
    usuarios.append(usuario)

    # Salva o "perfil comportamental" interno para uso na geração das transações
    comportamentos[id_usuario] = {
        "alpha": alpha,                          # Parâmetro de forma Gama (valores)
        "theta": theta,                          # Parâmetro de escala Gama (valores)
        "horarios": horarios_pref,               # Horários preferidos
        "canal": canal_pref,                     # Canal preferido
        "cidade": cidade,                        # Cidade base (domicílio)
        "saldo": saldo,                          # Saldo disponível a consumir
        "chance_anomalia_valor": chance_anomalia_valor,  # Prob. de valor atípico
        "chance_viagem": chance_viagem,          # Prob. de transação fora da cidade
        "descricoes_pref": descricoes_pref       # Descrições mais prováveis para o usuário
    }













# ================================================================
#  GERAR TRANSACOES (DINÂMICA AO LONGO DO ANO)
# ================================================================

# 'transacoes' acumula todas as transações de todos os usuários
transacoes = []

# Percorre cada usuário para simular suas transações ao longo de 2024
for id_usuario in range(1, N_CONTAS + 1):
    # Resgata o "perfil comportamental" previamente calculado
    perfil = comportamentos[id_usuario]

    # 'saldo_disp' começa como o saldo do usuário e vai sendo reduzido a cada transação
    # Se uma transação tiver valor superior ao saldo restante, ela é descartada (continue)
    saldo_disp = perfil["saldo"]

    # Para cada usuário, tenta criar TRANS_POR_CONTA transações
    for _ in range(TRANS_POR_CONTA):
        # 1) SORTEIO DO VALOR
        # Amostra um valor da distribuição Gama com parâmetros do usuário
        valor = np.random.gamma(perfil["alpha"], perfil["theta"])

        # 2) EVENTUAL ANOMALIA DE VALOR (cauda pesada)
        # Com probabilidade individual, multiplica o valor por um fator 3..6
        if random.random() < perfil["chance_anomalia_valor"]:
            valor *= random.uniform(3, 6)

        # Arredonda o valor para 2 casas decimais (R$)
        valor = round(valor, 2)

        # 3) VALIDAÇÃO DE SALDO
        # Se o valor for <= 0 (muito improvável) ou exceder o saldo restante, descarta a transação
        if valor <= 0 or valor > saldo_disp:
            continue

        # 4) ESCOLHA DO DESTINO
        # Escolhe aleatoriamente outra conta (diferente do próprio usuário) como destino
        destino = random.choice([u for u in usuarios if u["id_conta"] != id_usuario])

        # 5) DATA/HORA DA OPERAÇÃO
        # Usa os horários preferidos (99% das vezes) e distribui ao longo de 2024
        data_op = gerar_data_transacao(perfil["horarios"])

        # 6) CANAL UTILIZADO
        # 85% das vezes usa o canal preferido; 15% pode ser o outro canal
        canal = random.choices(
            [perfil["canal"], random.choice(CANAIS)],
            weights=[0.85, 0.15]
        )[0]

        # 7) LOCAL DE ORIGEM
        # Com probabilidade 'chance_viagem', escolhe outra cidade (simulando viagem/anomalia)
        # Caso contrário, usa a cidade de domicílio do usuário
        local_origem = perfil["cidade"] if random.random() > perfil["chance_viagem"] else random.choice(CIDADES)

        # 8) REGRAS DE SUSPEIÇÃO (sinalização simples)
        flag_suspeita = False

        # 8a) Valor suspeito: maior que o quantil 99.5% de UMA amostra Gama de referência
        # Obs.: recalcula a amostra toda vez (mais "caro"); alternativa: pré-calcular por usuário.

        ########################################################

        # AQUI QUE E GERADO A TRANSAÇÂO SUSPEITA FULGO FLAG 
        ########################################################
        if valor > np.quantile(np.random.gamma(perfil["alpha"], perfil["theta"], 1000), 0.995):
            flag_suspeita = True

        # 8b) Horário fora dos preferidos => potencial anomalia
        if data_op.hour not in perfil["horarios"]:
            flag_suspeita = True

        # 8c) Local de origem diferente da cidade base => potencial anomalia/viagem
        if local_origem != perfil["cidade"]:
            flag_suspeita = True

        # 9) CONSTRUÇÃO DO REGISTRO DA TRANSAÇÃO
        transacao = {
            "id_conta": id_usuario,                              # Quem está realizando a transação
            "tipo_operacao": "PIX",                              # Tipo fixo "PIX" no nosso gerador
            "valor": valor,                                      # Valor em R$
            "data_operacao": data_op.isoformat(),                # Data/hora ISO 8601
            "descricao_pagamento": random.choice(perfil["descricoes_pref"]),  # Descrição adequada ao usuário
            "tipo_transacao": random.choice(["envio", "recebimento"]),        # Envio ou recebimento (simulado)
            "status_transacao": "concluida",                     # Assumimos concluída neste gerador
            "id_conta_origem": id_usuario,                       # Origem = o próprio usuário
            "id_conta_destino": destino["id_conta"],             # Destino = outra conta
            "chave_pix_destino": destino["chave_pix"],           # Chave do destinatário
            "tipo_chave_pix_destino": destino["tipo_chave_pix"], # Tipo de chave do destinatário
            "instituicao_destino": destino["banco"],             # Banco do destinatário
            "canal": canal,                                      # Canal usado
            "local_origem": local_origem,                        # Local de origem (pode ser diferente da base)
            "flag_suspeita": flag_suspeita                       # Sinalizador de suspeita (bool)
        }

        # 10) ADICIONA AO HISTÓRICO E ABATE DO SALDO
        transacoes.append(transacao)
        saldo_disp -= valor  # Abate o valor do saldo disponível para este usuário

# ================================================================
#  SALVAR BASE FINAL EM JSON
# ================================================================

# Monta o dicionário final com duas "tabelas":
# - contas_bancarias: metadados dos usuários/contas
# - historico_pix: lista de transações simuladas
dados_finais_2 = {
    "contas_bancarias": usuarios,
    "historico_pix": transacoes
}

# Grava o arquivo base_pix_simulada.json no diretório atual
with open("TesteParaVariancia.json", "w", encoding="utf-8") as f:
    # indent=4 para legibilidade; ensure_ascii=False para manter acentos
    json.dump(dados_finais_2, f, indent=4, ensure_ascii=False)

# Mensagem final no console com contagens geradas
print(f"✅ Base gerada com {len(usuarios)} contas e {len(transacoes)} transações.")

# ================================================================
#  NOTAS / DICAS:
#  - Se quiser reprodutibilidade, defina seeds:
#      random.seed(42); np.random.seed(42)
#  - Para performance maior, pode-se:
#      * pré-calcular o limiar de suspeita por usuário (quantil 99.5%);
#      * vetorização parcial de valores por usuário;
#      * cachear destinos ou sortear blocos em lote.
#  - Evolução sugerida (TODO do topo):
#      * dar a cada usuário 2-3 "locais de costume" e sortear local_origem
#        com peso maior nesses locais (casa/trabalho) e menor em outros.
# ================================================================
