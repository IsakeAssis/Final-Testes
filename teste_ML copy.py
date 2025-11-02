"""
Pipeline híbrido (IsolationForest + Supervisionado + Semi-supervisionado)
Entrada: base_pix_simulada.json (contas_bancarias + historico_pix)
Função principal: analisar_transacao(nova_tx) -> {decision, explanation, details}
"""


import os, json

def carregar_json_seguro(caminho):
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
    if os.path.getsize(caminho) == 0:
        raise ValueError(f"Arquivo JSON vazio: {caminho}")
    with open(caminho, "r", encoding="utf-8") as f:
        return json.load(f)




import json
import math
from datetime import datetime
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ---------------------------
# 1) Helpers: carregar dados
# ---------------------------
def carregar_base(caminho_json="base_pix_simulada.json"):
    with open(caminho_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    contas = pd.DataFrame(data["contas_bancarias"])
    historico = pd.DataFrame(data["historico_pix"])

    # Normaliza tipos de ID para string consistente (evita mismatch int/str)
    if "id_conta" in contas.columns:
        contas["id_conta"] = contas["id_conta"].astype(str).str.strip()
    # harmoniza possíveis nomes no historico
    if "id_conta_origem" in historico.columns: ############### concertar isso / não aparece no json
        historico["id_conta_origem"] = historico["id_conta_origem"].astype(str).str.strip()
    if "id_conta_destino" in historico.columns:
        historico["id_conta_destino"] = historico["id_conta_destino"].astype(str).str.strip()
    # normaliza chaves pix também (string)
    if "chave_pix" in contas.columns:
        contas["chave_pix"] = contas["chave_pix"].astype(str).str.strip()

    return contas, historico

# ---------------------------
# 2) Engenharia de features
def agregados_por_conta(features_df):
    """
    Calcula estatísticas por conta (média, desvio padrão, contagem, etc.)
    para caracterizar o histórico de transações de cada usuário.
    """
    import pandas as pd

    features_df["id_conta_origem"] = (
        pd.to_numeric(features_df["id_conta_origem"], errors="coerce")
        .fillna(-1)
        .astype(int)
    )

    agg = (
        features_df.groupby("id_conta_origem")
        .agg(
            cnt_tx=("valor", "count"),
            mean_valor=("valor", "mean"),
            std_valor=("valor", "std"),
            mean_time_since=("time_since_prev_h", "mean"),
            pct_hour_pref=("hour_pref_match", "mean"),
            pct_local_match=("local_match", "mean"),
            pct_canal_match=("canal_match", "mean")
        )
        .fillna(0)
    )

    return agg

# ---------------------------
def extrair_features_transacoes(historico_df, contas_df):
    """
    Extrai e normaliza as principais features por transação.
    Corrige datas e garante consistência no cálculo do tempo entre transações.
    """
    import pandas as pd
    import numpy as np

    df = historico_df.copy()

    # ---------------------------
    # Normaliza IDs
    # ---------------------------
    if "id_conta_origem" in df.columns:
        df["id_conta_origem"] = df["id_conta_origem"].astype(str).str.strip()

    # ---------------------------
    # ⏰ Tempo desde a última transação (versão robusta)
    # ---------------------------
    df["data_operacao"] = pd.to_datetime(
        df["data_operacao"]
        .astype(str)
        .str.replace("T", " ", regex=False)
        .str.replace("Z", "", regex=False)
        .str.split(".").str[0],
        errors="coerce"
    )

    df = df.sort_values(by=["id_conta_origem", "data_operacao"]).copy()

    df["time_since_prev_h"] = (
        df.groupby("id_conta_origem")["data_operacao"]
        .diff()
        .apply(lambda x: x.total_seconds() / 3600 if pd.notna(x) else np.nan)
    )

    df["time_since_prev_h"] = df["time_since_prev_h"].fillna(9999.0)

    # ---------------------------
    # Features temporais
    # ---------------------------
    df["hour"] = df["data_operacao"].dt.hour
    df["day_of_week"] = df["data_operacao"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # ---------------------------
    # Merge com dados de contas
    # ---------------------------
    contas_small = contas_df[[
        "id_conta",
        "localizacao",
        "horarios_preferidos",
        "canal_preferido"
    ]].copy()

    df = df.merge(
        contas_small,
        left_on="id_conta_origem",
        right_on="id_conta",
        how="left",
        suffixes=("", "_conta")
    )

    # Garante novamente o tipo datetime após o merge
    df["data_operacao"] = pd.to_datetime(df["data_operacao"], errors="coerce")

    # ---------------------------
    # Horário preferido do usuário
    # ---------------------------
    def hour_pref_match(row):
        prefs = row["horarios_preferidos"]
        if isinstance(prefs, list):
            return int(row["hour"] in prefs)
        return 0

    df["hour_pref_match"] = df.apply(hour_pref_match, axis=1)

    # ---------------------------
    # Local de origem confere?
    # ---------------------------
    df["local_match"] = (df["local_origem"] == df["localizacao"]).astype(int)

    # ---------------------------
    # Log do valor
    # ---------------------------
    df["valor_log"] = np.log1p(df["valor"].astype(float))

    # ---------------------------
    # Label supervisionado
    # ---------------------------
    

    if "flag_suspeita" in df.columns:
        df["label"] = df["flag_suspeita"].astype("boolean").astype("Int64")
        print(df[df["flag_suspeita"].isna()])


    else:
        df["label"] = np.nan

    # ---------------------------
    # Seleção das features
    # ---------------------------
    features = df[[
        "id_conta_origem",
        "valor",
        "valor_log",
        "hour",
        "hour_pref_match",
        "local_match",
        "time_since_prev_h",
        "canal",
        "canal_preferido",
        "label"
    ]].copy()

    # Canal preferido
    features["canal_match"] = (features["canal"] == features["canal_preferido"]).astype(int)
    features.drop(columns=["canal", "canal_preferido"], inplace=True)

    return features, df






# ---------------------------
# 3) Treinamento dos modelos
# ---------------------------
def treinar_modelos(features_df):
    # Prepara dataset para treinar: usaremos features contínuas + categorias já binárias
    X = features_df[["valor_log","hour","time_since_prev_h","hour_pref_match","local_match","canal_match"]].fillna(0)
    
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Isolation Forest (treina em todos os dados como detector de anomalia)
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    # n_estimators e a qtd de arvores, 100–200 árvores: equilíbrio entre velocidade e precisão. 500+ árvores: útil em datasets grandes e complexos
    # # Proporção estimada de amostras anômalas (fraudes) no dataset.  
    # random_stage    Define a semente aleatória (controla a aleatoriedade interna).

    iso.fit(Xs)

    # Supervisionado: usa label (flag_suspeita) quando disponível
    labeled_mask = features_df["label"].notna()
    clf = None
    if labeled_mask.sum() >= 10:  # precisa de pelo menos alguns exemplos
        X_lab = Xs[labeled_mask.values]
        y_lab = features_df.loc[labeled_mask, "label"].astype(int).values
        # Dividir para estimar performance (opcional)
        X_tr, X_te, y_tr, y_te = train_test_split(X_lab, y_lab, test_size=0.2, random_state=42, stratify=y_lab)
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X_tr, y_tr)
        # podemos avaliar com classification_report aqui se quiser
    # Semi-supervisionado: LabelPropagation
    # Prepara rótulos: -1 para unlabeled
    lp_labels = np.full(len(features_df), -1, dtype=int)
    labeled_indices = features_df["label"].notna()
    lp_labels[labeled_indices.values] = features_df.loc[labeled_indices, "label"].astype(int)
    lp = LabelPropagation()
    try:
        lp.fit(Xs, lp_labels)
    except Exception:
        # se falhar (poucos rótulos) deixa lp como None
        lp = None

    return {
        "scaler": scaler,
        "iso": iso,
        "clf": clf,
        "lp": lp
    }

# ---------------------------
# 4) Função que analisa nova transação
'''
# ---------------------------
def analisar_transacao(nova_tx: dict, contas_df, historico_df, models, agg_stats):
    """
    nova_tx: dicionário com keys compatíveis (id_conta_origem, valor, data_operacao, canal, local_origem, ...)
    Retorna: dict com decisão e explicação
    """
    # passo 1: extrair histórico do usuário
    idc = nova_tx["id_conta_origem"]
    hist_user = historico_df[historico_df["id_conta_origem"] == idc].copy()
    # monta features da nova transação no mesmo formato
    tx_time = pd.to_datetime(nova_tx["data_operacao"])
    hour = int(tx_time.hour)
    # time_since_prev: horas desde última transação do usuário
    if not hist_user.empty:
        last_ts = hist_user["data_operacao"].max()
        time_since_h = (tx_time - last_ts).total_seconds()/3600.0
    else:
        time_since_h = 9999.0
    # preferencia horário / local / canal da conta
    conta_info = contas_df[contas_df["id_conta"] == idc].squeeze() if idc in contas_df["id_conta"].values else None
    hour_pref_match = 0
    local_match = 0
    canal_match = 0
    if conta_info is not None:
        prefs = conta_info.get("horarios_preferidos", [])
        if isinstance(prefs, list):
            hour_pref_match = int(hour in prefs)
        local_match = int(nova_tx.get("local_origem") == conta_info.get("localizacao"))
        canal_match = int(nova_tx.get("canal") == conta_info.get("canal_preferido"))
    valor = float(nova_tx["valor"])
    valor_log = math.log1p(valor)

    X_new = np.array([[valor_log, hour, time_since_h, hour_pref_match, local_match, canal_match]])
    Xs_new = models["scaler"].transform(X_new)

    # 1) Isolation forest -> score and binary decision
    iso_score = models["iso"].decision_function(Xs_new)[0]  # quanto menor -> mais anômalo
    iso_pred = models["iso"].predict(Xs_new)[0]  # 1 normal / -1 outlier

    # 2) Supervisionado (probabilidade)
    clf = models.get("clf")
    sup_proba = None
    sup_pred = None
    if clf is not None:
        sup_proba = clf.predict_proba(Xs_new)[0,1]  # probabilidade de ser fraud (assumimos classe 1)
        sup_pred = int(clf.predict(Xs_new)[0])

    # 3) Semi-supervisionado
    lp = models.get("lp")
    lp_pred = None
    if lp is not None:
        lp_pred = int(lp.predict(Xs_new)[0])

    # Regras de combinação simples (exemplo):
    # - se Isolation detectou outlier (iso_pred == -1) e sup_proba > 0.4 -> sinaliza suspeita
    # - se sup_proba >= 0.8 -> suspeita forte
    # - se lp_pred == 1 -> soma como evidência
    score = 0.0
    reasons = []
    # ajustar iso_score para escala legível: iso_score menor => mais suspeito
    # convertendo iso_score para [0,1] inverso (aprox)
    iso_score_norm = 1.0 - (1.0 / (1.0 + math.exp(iso_score)))  # sigmoide invertida aprox
    score += (0.5 * iso_score_norm)
    reasons.append(f"IsolationForest score(raw)={iso_score:.4f}, norm={iso_score_norm:.3f} ({'outlier' if iso_pred==-1 else 'normal'})")

    if sup_proba is not None:
        score += 0.4 * sup_proba
        reasons.append(f"Supervised model prob(fraud)={sup_proba:.3f} (pred={sup_pred})")
    if lp_pred is not None:
        score += 0.2 * lp_pred
        reasons.append(f"LabelPropagation pred={lp_pred}")

    # features outliers: valor muito > mean histórico?
    ################################################################ Errro aqui 
    ag = None
    if idc in agg_stats.index:
        ag = agg_stats.loc[idc]
        if ag["cnt_tx"] >= 3: ####oque acontece se eu almentar esse numero? ou diminuir? estava em 3
            if valor > ag["mean_valor"] + 3*(ag["std_valor"] if ag["std_valor"]>0 else 1):
                score += 0.6
                reasons.append(f"Valor {valor:.2f} >> histórico mean {ag['mean_valor']:.2f} (3x std)")
            # hora fora do padrão
            if hour_pref_match == 0:
                reasons.append("Hora da transação fora do horário preferido do usuário")
    else:
        reasons.append("Pouco/nenhum histórico para essa conta (cold-start)")

    # normalize final score to 0..1
    final_score = 1/(1+math.exp(-score+0.5))  # sigmoide com shift
    decision = "normal"
    if final_score > 0.75:
        decision = "suspeita_alta"
    elif final_score > 0.45:
        decision = "suspeita_baixa"

    explanation = {
        "final_score": round(final_score,3),
        "decision": decision,
        "reasons": reasons,
        "iso_pred": int(iso_pred),
        "iso_score_raw": float(iso_score),
        "supervised_prob": None if sup_proba is None else float(sup_proba),
        "labelprop_pred": None if lp_pred is None else int(lp_pred),
        "historical_summary": ag.to_dict() if ag is not None else None
    }
    return explanation


'''

def analisar_transacao(nova_tx: dict, contas_df, historico_df, models, agg_stats):
    """
    Analisa uma nova transação e retorna um score de suspeita com explicações.

    Parâmetros:
        nova_tx: dict com os campos da transação (id_conta_origem, id_conta_destino, valor, data_operacao, etc.)
        contas_df: DataFrame com dados cadastrais das contas
        historico_df: DataFrame com o histórico de transações
        models: dicionário com modelos já treinados (scaler, iso, clf, lp)
        agg_stats: DataFrame agregado por conta (origem e/ou destino)

    Retorno:
        dict com decisão, score e motivos
    """
    import pandas as pd, numpy as np, math

    
    # ------------------------------------------------------------
    # 1️⃣ Identificar qual conta analisar (origem ou destino)
    # ------------------------------------------------------------
    tipo_tx = nova_tx.get("tipo_transacao", "").lower()
    if tipo_tx == "recebimento":
        idc = nova_tx.get("id_conta_destino")
    else:  # padrão: envio
        idc = nova_tx.get("id_conta_origem")

    import pandas as pd
    import numpy as np
    import math

    # ------------------------------------------------------------
    # 0️⃣ Normalizações de segurança (garante tipos coerentes)
    # ------------------------------------------------------------
    # garante que idc (id da transação) seja int
    try:
        # tenta obter id da transacao (origem ou destino já decididos acima)
        idc = int(idc)
    except Exception:
        # se não conseguir converter, marca como inválido e segue
        print(f"[ERROR] idc inválido: {idc}")
        idc = None

    # converte colunas de id do histórico para numérico (sem perder o df original)
    if "id_conta_origem" in historico_df.columns:
        historico_df["id_conta_origem"] = pd.to_numeric(historico_df["id_conta_origem"], errors="coerce")
    if "id_conta_destino" in historico_df.columns:
        historico_df["id_conta_destino"] = pd.to_numeric(historico_df["id_conta_destino"], errors="coerce")

    # ------------------------------------------------------------
    # 1️⃣ Extrair histórico dessa conta (origem OU destino)
    # ------------------------------------------------------------
    if idc is None:
        hist_user = pd.DataFrame(columns=historico_df.columns)
    else:
        # seleciona por igualdade numérica — evita problemas tipo '1' vs 1.0 vs '1.0'
        hist_user = historico_df[
            (historico_df["id_conta_origem"] == idc) |
            (historico_df["id_conta_destino"] == idc)
        ].copy()

    # debug básico se hist_user vazio
    if hist_user.empty:
        print(f"[DEBUG] Nenhum histórico encontrado para conta {idc}. "
              f"Exemplo índices históricos (primeiros 10): {historico_df[['id_conta_origem','id_conta_destino']].head(10).to_dict(orient='records')}")
    else:
        print(f"[DEBUG] {len(hist_user)} registros de histórico encontrados para conta {idc} (mostrando 3):")
        print(hist_user.head(3).to_dict(orient='records'))

    # ------------------------------------------------------------
    # 2️⃣ Converte a data da nova transação (tx_time) de forma robusta
    # ------------------------------------------------------------
    tx_time = pd.to_datetime(nova_tx.get("data_operacao"), errors="coerce")
    if pd.isna(tx_time):
        raise ValueError(f"Data inválida na transação: {nova_tx.get('data_operacao')}")
    hour = int(tx_time.hour)

    # ------------------------------------------------------------
    # 3️⃣ Tempo desde a última transação (com correção robusta)
    # ------------------------------------------------------------
        # ------------------------------------------------------------
    # 3️⃣ Tempo desde a última transação (versão à prova de tipo)
    # ------------------------------------------------------------
    time_since_h = 9999.0  # valor padrão

    try:
        # Força tx_time a datetime de forma absoluta
        tx_time = pd.to_datetime(str(nova_tx.get("data_operacao")), errors="coerce")

        if pd.isna(tx_time):
            raise ValueError(f"Data inválida na transação: {nova_tx.get('data_operacao')}")

        if not hist_user.empty:
            # Converte coluna de datas do histórico
            hist_user["data_operacao"] = pd.to_datetime(
                hist_user["data_operacao"].astype(str)
                .str.replace("T", " ", regex=False)
                .str.replace("Z", "", regex=False)
                .str.split(".").str[0],  # remove frações de segundos
                errors="coerce"
            )

            # Remove linhas sem data válida
            hist_user = hist_user[hist_user["data_operacao"].notna()].copy()

            if not hist_user.empty:
                # Garante que o máximo é datetime
                last_ts = pd.to_datetime(hist_user["data_operacao"].max(), errors="coerce")

                if pd.isna(last_ts):
                    print(f"[WARN] Última data inválida para conta {idc}: {last_ts}")
                    time_since_h = 9999.0
                else:
                    # Converte ambos para datetime no momento da subtração (blindagem total)
                    tx_time_dt = pd.to_datetime(tx_time, errors="coerce")
                    last_ts_dt = pd.to_datetime(last_ts, errors="coerce")

                    if pd.isna(tx_time_dt) or pd.isna(last_ts_dt):
                        print(f"[WARN] Falha na coerção final para datetime em conta {idc}.")
                        time_since_h = 9999.0
                    else:
                        delta_horas = (tx_time_dt - last_ts_dt).total_seconds() / 3600.0
                        if delta_horas < 0:
                            print(f"[WARN] Delta negativo ({delta_horas:.2f}h) — corrigindo com abs().")
                            delta_horas = abs(delta_horas)
                        time_since_h = delta_horas
            else:
                print(f"[INFO] Histórico sem datas válidas para conta {idc}.")
                time_since_h = 9999.0
        else:
            print(f"[INFO] Sem histórico para conta {idc}.")
            time_since_h = 9999.0

    except Exception as e:
        print(f"[ERRO] Falha geral no cálculo de time_since_h para conta {idc}: {e}")
        time_since_h = 9999.0

    print(f"[DEBUG] Conta {idc} | time_since_h calculado: {time_since_h:.2f}h")


    # ------------------------------------------------------------
    # 3️⃣ Preferências do usuário
    # ------------------------------------------------------------
    conta_info = contas_df[contas_df["id_conta"] == idc].squeeze() if idc in contas_df["id_conta"].values else None
    hour_pref_match = 0
    local_match = 0
    canal_match = 0
    if conta_info is not None:
        prefs = conta_info.get("horarios_preferidos", [])
        if isinstance(prefs, list):
            hour_pref_match = int(hour in prefs)
        local_match = int(nova_tx.get("local_origem") == conta_info.get("localizacao"))
        canal_match = int(nova_tx.get("canal") == conta_info.get("canal_preferido"))

    valor = float(nova_tx["valor"])
    valor_log = math.log1p(valor)

    # ------------------------------------------------------------
    # 4️⃣ Montar vetor de features e normalizar
    # ------------------------------------------------------------
    X_new = np.array([[valor_log, hour, time_since_h, hour_pref_match, local_match, canal_match]])
    Xs_new = models["scaler"].transform(X_new)

    # ------------------------------------------------------------
    # 5️⃣ Modelos de detecção
    # ------------------------------------------------------------
    iso_score = models["iso"].decision_function(Xs_new)[0]
    iso_pred = models["iso"].predict(Xs_new)[0]

    clf = models.get("clf")
    sup_proba, sup_pred = None, None
    if clf is not None:
        sup_proba = clf.predict_proba(Xs_new)[0, 1]
        sup_pred = int(clf.predict(Xs_new)[0])

    lp = models.get("lp")
    lp_pred = int(lp.predict(Xs_new)[0]) if lp is not None else None

    # ------------------------------------------------------------
    # 6️⃣ Combinar evidências e explicações
    # ------------------------------------------------------------
    score = 0.0
    reasons = []

    iso_score_norm = 1.0 - (1.0 / (1.0 + math.exp(iso_score)))  # sigmoide invertida
    score += (0.5 * iso_score_norm)
    reasons.append(
        f"IsolationForest score(raw)={iso_score:.4f}, norm={iso_score_norm:.3f} "
        f"({'outlier' if iso_pred == -1 else 'normal'})"
    )

    if sup_proba is not None:
        score += 0.4 * sup_proba
        reasons.append(f"Supervised model prob(fraud)={sup_proba:.3f} (pred={sup_pred})")
    if lp_pred is not None:
        score += 0.2 * lp_pred
        reasons.append(f"LabelPropagation pred={lp_pred}")

    # ------------------------------------------------------------
    # 7️⃣ Regras baseadas no histórico agregado
    # ------------------------------------------------------------
    ag = None
    if idc in agg_stats.index:
        ag = agg_stats.loc[idc]

        if ag["cnt_tx"] >= 3:
            lim = ag["mean_valor"] + 3 * (ag["std_valor"] if ag["std_valor"] > 0 else 1)
            if valor > lim:
                #score += 0.6
                excesso = valor / (ag["mean_valor"] + 1)
                score += min(1.5, 0.6 + math.log10(excesso))  # escala logarítmica
                reasons.append(
                f"Valor {valor:.2f} >> histórico mean {ag['mean_valor']:.2f} (3x std)"
               )
            if hour_pref_match == 0:
                reasons.append("Hora da transação fora do horário preferido do usuário")
    else:
        reasons.append("Pouco/nenhum histórico para essa conta (cold-start)")

    # ------------------------------------------------------------
    # 8️⃣ Normalizar score final e definir decisão
    # ------------------------------------------------------------
    final_score = 1 / (1 + math.exp(-score + 0.5))  # sigmoide ajustada
    '''
    if final_score > 0.75:
        decision = "suspeita_alta"
    elif final_score > 0.45:
        decision = "suspeita_baixa"
    else:
        decision = "normal"
    '''


    #################################################################################################################

    
    if final_score > 0.65:
        decision = "suspeita_alta"
    elif final_score > 0.4:
        decision = "suspeita_baixa"
    else:
        decision = "normal"

    # ------------------------------------------------------------
    # 🔚 Retornar resultado
    # ------------------------------------------------------------
    explanation = {
        "final_score": round(final_score, 3),
        "decision": decision,
        "reasons": reasons,
        "iso_pred": int(iso_pred),
        "iso_score_raw": float(iso_score),
        "supervised_prob": None if sup_proba is None else float(sup_proba),
        "labelprop_pred": None if lp_pred is None else int(lp_pred),
        "historical_summary": ag.to_dict() if ag is not None else None,
        "tipo_transacao": tipo_tx,
        "id_conta_analisada": idc
    }

    return explanation








'''import pandas as pd
contas, historico = carregar_base(carregar_base)
print("contas.columns:", contas.columns.tolist())
print("historico.columns:", historico.columns.tolist())
print("\ncontas sample:\n", contas.head(5).to_dict(orient='records'))
print("\nhistorico sample:\n", historico.head(5).to_dict(orient='records'))
'''





arquivo = "base_pix_simulada.json"

with open(arquivo, "r", encoding="utf-8") as f:
    linhas = f.readlines()

print("Total de linhas:", len(linhas))
erro_linha = 1737459

for i in range(erro_linha - 3, erro_linha + 3):
    if 0 <= i < len(linhas):
        print(f"{i+1:>8}: {linhas[i].rstrip()}")

# ---------------------------
# 5) Exemplo de uso (opcional para testes)
# ---------------------------
if __name__ == "__main__":
    # Teste rápido do pipeline completo
    contas, historico_raw = carregar_base("base_pix_simulada.json")
    historico_raw["data_operacao"] = pd.to_datetime(historico_raw["data_operacao"], errors="coerce")
    features, hist_with_feats = extrair_features_transacoes(historico_raw, contas)
    agg = agregados_por_conta(features)
    models = treinar_modelos(features)

    # Simula uma transação
    nova = {
        "id_conta_origem": 1,
        "valor": 8500.0,
        "data_operacao": "2024-12-01T03:30:00",
        "canal": "mobile",
        "local_origem": "Lago Norte",
        "tipo_transacao": "envio"
    }

    resultado = analisar_transacao(nova, contas, historico_raw, models, agg)
    print("Resultado da análise:", resultado)



''''''
#############################################################

import os
import time
import json
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


# Importa as funções do seu script principal de ML
from teste_ML import (
    carregar_base,
    extrair_features_transacoes,
    agregados_por_conta,
    treinar_modelos,
    analisar_transacao,
)

# -----------------------------
# 1️⃣ Caminho absoluto seguro
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAMINHO_JSON = os.path.join(BASE_DIR, "base_pix_simulada.json")

# Verifica se o arquivo existe
if not os.path.exists(CAMINHO_JSON):
    raise FileNotFoundError(f"[ERRO] Arquivo não encontrado: {CAMINHO_JSON}")

# Diretório do arquivo (para o watchdog observar)
path = os.path.dirname(CAMINHO_JSON)
if not os.path.isdir(path):
    raise FileNotFoundError(f"[ERRO] Diretório inválido: {path}")

print(f"[INFO] Arquivo localizado com sucesso:")
print(f"       {CAMINHO_JSON}\n")

# -----------------------------
# 2️⃣ Carrega a base inicial
# -----------------------------
print("[INFO] Carregando base inicial...")
contas, historico_raw = carregar_base(CAMINHO_JSON)
features, hist_with_feats = extrair_features_transacoes(historico_raw, contas)
agg = agregados_por_conta(features)
models = treinar_modelos(features)
ultimo_tamanho = len(historico_raw)

print(f"[READY] Monitoramento iniciado com {ultimo_tamanho} transações.\n")

# -----------------------------
# 3️⃣ Classe de monitoramento
# -----------------------------
class MonitorJSON(FileSystemEventHandler):
    def on_modified(self, event):
        """Executado automaticamente quando o arquivo JSON for modificado."""
        global ultimo_tamanho, contas, historico_raw, models, agg

        # Só age se o arquivo modificado for o JSON alvo
        if event.src_path.endswith("base_pix_simulada.json"):
            print("\n[EVENTO] O arquivo foi modificado. Verificando novas transações...")

            try:
                # Recarrega a base atualizada
                contas, novo_historico = carregar_base(CAMINHO_JSON)

                # Detecta novas transações
                if len(novo_historico) > ultimo_tamanho:
                    novas = novo_historico.iloc[ultimo_tamanho:]
                    qtd_novas = len(novas)
                    print(f"[DETECTADO] {qtd_novas} nova(s) transação(ões) encontrada(s).")

                    for _, tx in novas.iterrows():
                        tx_dict = tx.to_dict()
                        resultado = analisar_transacao(tx_dict, contas, historico_raw, models, agg)
                        print("####################################################")
                        print("\n🧾 Nova transação detectada:")
                        print(f"   ID Conta: {tx_dict.get('id_conta_origem', '?')}")
                        print(f"   ID Conta: {tx_dict.get('id_conta', '?')}")
                        print(f"   Valor: R$ {tx_dict.get('valor', 0):,.2f}")
                        print(f"   Data/Hora: {tx_dict.get('data_operacao', '?')}")
                        print(f"   Resultado: {resultado['decision'].upper()} (score={resultado['final_score']})")
                        print("   Motivos:")
                        for r in resultado["reasons"]:
                            print(f"     - {r}")
                        print("-" * 60)

                    # Atualiza histórico e contador
                    historico_raw = novo_historico
                    ultimo_tamanho = len(novo_historico)
                else:
                    print("[INFO] Nenhuma nova transação adicionada.")
            except Exception as e:
                print(f"[ERRO] Durante a análise: {e}")



##################################################################################



import pandas as pd
contas, historico = carregar_base(CAMINHO_JSON)
print("contas.columns:", contas.columns.tolist())
print("historico.columns:", historico.columns.tolist())
print("\ncontas sample:\n", contas.head(5).to_dict(orient='records'))
print("\nhistorico sample:\n", historico.head(5).to_dict(orient='records'))
historico["data_operacao"] = pd.to_datetime(historico["data_operacao"], errors="coerce")
features, hist_with_feats = extrair_features_transacoes(historico, contas)




# ---------------------------
# Helper: tenta resolver IDs de origem e destino de uma transação
# ---------------------------
def resolve_tx_ids(tx_dict, contas_df):
    print("##########################################################################################")
    """
    Tenta recuperar id_conta_origem e id_conta_destino de tx_dict.
    Estratégias (ordem):
      1) checar chaves comuns (vários nomes)
      2) forçar conversão int->str compatível com contas_df
      3) fallback por chave PIX (chave_pix / chave_pix_origem / chave_pix_destino)
      4) se nada, retorna (None, None)
    Retorna: (id_origem_str_or_None, id_destino_str_or_None, debug_info)
    """
    debug = {}
    # nomes possíveis
    poss_origem = ["id_conta_origem", "id_conta", "id_origem", "origem_id", "id_cliente_origem"]
    poss_destino = ["id_conta_destino", "id_destino", "destino_id", "id_conta_receb", "id_cliente_destino"]

    def pick_first(dct, candidates):
        for c in candidates:
            if c in dct and dct[c] not in (None, "", float("nan")):
                return dct[c], c
        return None, None

    o_val, o_key = pick_first(tx_dict, poss_origem)
    d_val, d_key = pick_first(tx_dict, poss_destino)
    debug["found_keys"] = {"orig_key": o_key, "dest_key": d_key}

    def normalize_id_val(v):
        # se é NaN do numpy/pandas, tratar como None
        print("######################################")
        try:
            if v is None:
                return None
            s = str(v)
            # remove .0 se for float representando inteiro
            if s.endswith(".0"):
                s = s[:-2]
            s = s.strip()
            if s == "nan" or s == "":
                return None
            return s
        except Exception:
            return None

    id_origem = normalize_id_val(o_val)
    id_destino = normalize_id_val(d_val)

    # Se ainda não tem id_origem, tenta casar pela chave PIX de origem
    if id_origem is None:
        print("####################################")
        # possíveis campos de chave pix
        poss_chaves_origem = ["chave_pix_origem", "chave_pix", "chave_origem"]
        for k in poss_chaves_origem:
            if k in tx_dict and tx_dict[k]:
                chave = str(tx_dict[k])
                debug["tried_chave_origem"] = chave
                # procura na tabela de contas
                matched = contas_df[contas_df["chave_pix"].astype(str) == chave]
                if len(matched) > 0:
                    id_origem = str(int(matched.iloc[0]["id_conta"])) if pd.notna(matched.iloc[0]["id_conta"]) else None
                    debug["matched_by_chave_origem"] = id_origem
                    break

    # Se ainda não tem id_destino, tenta casar pela chave_pix_destino
    if id_destino is None:
        print("#############################################################################")
        poss_chaves_dest = ["chave_pix_destino", "chave_pix", "chave_destino"]
        for k in poss_chaves_dest:
            if k in tx_dict and tx_dict[k]:
                chave = str(tx_dict[k])
                debug["tried_chave_destino"] = chave
                matched = contas_df[contas_df["chave_pix"].astype(str) == chave]
                if len(matched) > 0:
                    id_destino = str(int(matched.iloc[0]["id_conta"])) if pd.notna(matched.iloc[0]["id_conta"]) else None
                    debug["matched_by_chave_destino"] = id_destino
                    break

    # último recurso: se historico tem coluna 'id_conta' (parece ser redundante), use-a
    if id_origem is None and "id_conta" in tx_dict:
        print("###########################################")
        id_origem = normalize_id_val(tx_dict.get("id_conta"))
        

    debug["final"] = {"id_origem": id_origem, "id_destino": id_destino}
    return id_origem, id_destino, debug



# -----------------------------
# 4️⃣ Inicia o observador
# -----------------------------
observer = Observer()
event_handler = MonitorJSON()
observer.schedule(event_handler, path, recursive=False)
observer.start()

print("[MONITORANDO] O sistema está ativo. Adicione novas transações ao JSON para testar.\n")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n[ENCERRADO] Monitoramento finalizado pelo usuário.")
    observer.stop()
observer.join()



# ---------------------------
# 5️⃣ Execução principal
# ---------------------------
if __name__ == "__main__":
    print("[INFO] Iniciando análise e monitoramento...")

    # 1️⃣ Carrega base inicial
    contas, historico = carregar_base(CAMINHO_JSON)
    print("contas.columns:", contas.columns.tolist())
    print("historico.columns:", historico.columns.tolist())
    print("\ncontas sample:\n", contas.head(5).to_dict(orient='records'))
    print("\nhistorico sample:\n", historico.head(5).to_dict(orient='records'))

    # 2️⃣ Converte datas e extrai features
    historico["data_operacao"] = pd.to_datetime(historico["data_operacao"], errors="coerce")
    features, hist_with_feats = extrair_features_transacoes(historico, contas)
    agg = agregados_por_conta(features)
    models = treinar_modelos(features)

    # 3️⃣ Define helper para resolver IDs
    def resolve_tx_ids(tx_dict, contas_df):
        print("##########################################################################################")
        debug = {}
        poss_origem = ["id_conta_origem", "id_conta", "id_origem", "origem_id", "id_cliente_origem"]
        poss_destino = ["id_conta_destino", "id_destino", "destino_id", "id_conta_receb", "id_cliente_destino"]

        def pick_first(dct, candidates):
            for c in candidates:
                if c in dct and dct[c] not in (None, "", float("nan")):
                    return dct[c], c
            return None, None

        o_val, o_key = pick_first(tx_dict, poss_origem)
        d_val, d_key = pick_first(tx_dict, poss_destino)
        debug["found_keys"] = {"orig_key": o_key, "dest_key": d_key}

        def normalize_id_val(v):
            try:
                if v is None:
                    return None
                s = str(v)
                if s.endswith(".0"):
                    s = s[:-2]
                s = s.strip()
                if s == "nan" or s == "":
                    return None
                return s
            except Exception:
                return None

        id_origem = normalize_id_val(o_val)
        id_destino = normalize_id_val(d_val)

        if id_origem is None:
            poss_chaves_origem = ["chave_pix_origem", "chave_pix", "chave_origem"]
            for k in poss_chaves_origem:
                if k in tx_dict and tx_dict[k]:
                    chave = str(tx_dict[k])
                    matched = contas_df[contas_df["chave_pix"].astype(str) == chave]
                    if len(matched) > 0:
                        id_origem = str(int(matched.iloc[0]["id_conta"])) if pd.notna(matched.iloc[0]["id_conta"]) else None
                        debug["matched_by_chave_origem"] = id_origem
                        break

        if id_destino is None:
            poss_chaves_dest = ["chave_pix_destino", "chave_pix", "chave_destino"]
            for k in poss_chaves_dest:
                if k in tx_dict and tx_dict[k]:
                    chave = str(tx_dict[k])
                    matched = contas_df[contas_df["chave_pix"].astype(str) == chave]
                    if len(matched) > 0:
                        id_destino = str(int(matched.iloc[0]["id_conta"])) if pd.notna(matched.iloc[0]["id_conta"]) else None
                        debug["matched_by_chave_destino"] = id_destino
                        break

        if id_origem is None and "id_conta" in tx_dict:
            id_origem = normalize_id_val(tx_dict.get("id_conta"))

        debug["final"] = {"id_origem": id_origem, "id_destino": id_destino}
        return id_origem, id_destino, debug


    # 4️⃣ Inicia o monitoramento do JSON
    observer = Observer()
    event_handler = MonitorJSON()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()

    print("\n[MONITORANDO] O sistema está ativo. Adicione novas transações ao JSON para testar.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[ENCERRADO] Monitoramento finalizado pelo usuário.")
        observer.stop()
    observer.join()
