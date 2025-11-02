"""
Sistema de Detecção de Fraudes PIX - COM MONITORAMENTO AUTOMÁTICO
Monitora o arquivo JSON e analisa novas transações em tempo real

Instalação:
pip install numpy pandas scikit-learn xgboost watchdog

Uso:
python fraud_detection.py --modo monitorar
"""

import os
import json
import math
import logging
import time
import argparse
from datetime import datetime
from typing import Dict, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler



import json

caminho = "base_pix_simulada.json"

with open(caminho, "r", encoding="utf-8") as f:
    linhas = f.readlines()

print(f"Total de linhas: {len(linhas)}")
print(f"Analisando linha 1150668 (aproximadamente)...")

for i in range(1150650, 1150680):  # 30 linhas em torno do erro
    print(f"{i+1:>8}: {linhas[i].rstrip()}")





try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CARREGAMENTO E PREPARAÇÃO
# ============================================================================

def normalizar_ids(df: pd.DataFrame, colunas: list) -> pd.DataFrame:
    """Normaliza colunas de ID para int."""
    df = df.copy()
    for col in colunas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1).astype(int)
    return df


def carregar_base(caminho_json: str = "base_pix_simulada.json") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega e normaliza a base de dados PIX."""
    if not os.path.exists(caminho_json):
        raise FileNotFoundError(f"❌ Arquivo não encontrado: {caminho_json}")
    
    with open(caminho_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    contas = pd.DataFrame(data["contas_bancarias"])
    historico = pd.DataFrame(data["historico_pix"])
    
    contas = normalizar_ids(contas, ["id_conta"])
    historico = normalizar_ids(historico, ["id_conta", "id_conta_origem", "id_conta_destino"])
    
    return contas, historico


def extrair_features_transacoes(historico_df: pd.DataFrame, contas_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extrai features enriquecidas para detecção de fraude."""
    df = historico_df.copy()
    
    # Normaliza datas
    df["data_operacao"] = pd.to_datetime(
        df["data_operacao"].astype(str)
        .str.replace("T", " ", regex=False)
        .str.replace("Z", "", regex=False)
        .str.split(".").str[0],
        errors='coerce'
    )
    
    df = df[df["data_operacao"].notna()].copy()
    
    # Determina ID principal
    df["id_principal"] = df.apply(
        lambda row: row["id_conta_destino"] if row.get("tipo_transacao") == "recebimento" else row["id_conta_origem"],
        axis=1
    )
    
    # Tempo desde última transação
    df = df.sort_values(by=["id_principal", "data_operacao"]).copy()
    df["time_since_prev_h"] = (
        df.groupby("id_principal")["data_operacao"]
        .diff()
        .dt.total_seconds() / 3600.0
    ).fillna(9999.0)
    
    # Features temporais
    df["hour"] = df["data_operacao"].dt.hour
    df["day_of_week"] = df["data_operacao"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_night"] = df["hour"].isin(range(0, 6)).astype(int)
    
    # Merge com contas
    contas_info = contas_df[["id_conta", "localizacao", "horarios_preferidos", "canal_preferido"]].copy()
    df = df.merge(contas_info, left_on="id_principal", right_on="id_conta", how="left", suffixes=("", "_conta"))
    
    df["data_operacao"] = pd.to_datetime(df["data_operacao"], errors='coerce')
    
    # Match com preferências
    def calc_hour_pref_match(row):
        prefs = row.get("horarios_preferidos")
        if isinstance(prefs, list) and len(prefs) > 0:
            return int(row["hour"] in prefs)
        return 0
    
    df["hour_pref_match"] = df.apply(calc_hour_pref_match, axis=1)
    df["local_match"] = (df["local_origem"] == df["localizacao"]).astype(int)
    df["canal_match"] = (df["canal"] == df["canal_preferido"]).astype(int)
    
    # Transformações de valor
    df["valor_log"] = np.log1p(df["valor"].astype(float))
    df["valor_zscore"] = (df["valor"] - df["valor"].mean()) / (df["valor"].std() + 1e-8)
    
    # Label
    if "flag_suspeita" in df.columns:
        df["label"] = df["flag_suspeita"].astype(bool).astype(int)
    else:
        df["label"] = np.nan
    
    # Seleção de features
    feature_cols = [
        "id_principal", "valor", "valor_log", "valor_zscore",
        "hour", "day_of_week", "is_weekend", "is_night",
        "hour_pref_match", "local_match", "canal_match",
        "time_since_prev_h", "label"
    ]
    
    features = df[feature_cols].copy()
    
    return features, df


def agregados_por_conta(features_df: pd.DataFrame) -> pd.DataFrame:
    """Calcula estatísticas agregadas por conta."""
    agg = (
        features_df.groupby("id_principal")
        .agg(
            cnt_tx=("valor", "count"),
            mean_valor=("valor", "mean"),
            std_valor=("valor", "std"),
            max_valor=("valor", "max"),
            mean_time_since=("time_since_prev_h", "mean"),
            pct_hour_pref=("hour_pref_match", "mean"),
            pct_local_match=("local_match", "mean"),
            pct_canal_match=("canal_match", "mean")
        )
        .fillna(0)
    )
    return agg


# ============================================================================
# TREINAMENTO DOS MODELOS
# ============================================================================

def treinar_modelos(features_df: pd.DataFrame) -> dict:
    """Treina ensemble de modelos para detecção de fraude."""
    X_cols = ["valor_log", "valor_zscore", "hour", "day_of_week", "is_weekend", 
              "is_night", "time_since_prev_h", "hour_pref_match", "local_match", "canal_match"]
    X = features_df[X_cols].fillna(0).values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # IsolationForest
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    iso.fit(X_scaled)
    
    # Modelos supervisionados
    labeled_mask = features_df["label"].notna()
    rf_clf = None
    gb_clf = None
    
    if labeled_mask.sum() >= 20:
        X_labeled = X_scaled[labeled_mask]
        y_labeled = features_df.loc[labeled_mask, "label"].astype(int).values
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_labeled, y_labeled, test_size=0.2, random_state=42, stratify=y_labeled
        )
        
        # RandomForest
        rf_clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        rf_clf.fit(X_train, y_train)
        
        # XGBoost ou GradientBoosting
        if XGBOOST_AVAILABLE:
            gb_clf = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
        else:
            gb_clf = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        
        gb_clf.fit(X_train, y_train)
    
    # LabelPropagation
    lp_labels = np.full(len(features_df), -1, dtype=int)
    lp_labels[labeled_mask] = features_df.loc[labeled_mask, "label"].astype(int)
    
    lp = None
    try:
        lp = LabelPropagation(kernel='knn', n_neighbors=7)
        lp.fit(X_scaled, lp_labels)
    except Exception:
        pass
    
    return {
        "scaler": scaler,
        "iso": iso,
        "rf": rf_clf,
        "gb": gb_clf,
        "lp": lp,
        "feature_cols": X_cols
    }


# ============================================================================
# ANÁLISE DE TRANSAÇÃO
# ============================================================================

def analisar_transacao(
    nova_tx: dict,
    contas_df: pd.DataFrame,
    historico_df: pd.DataFrame,
    models: dict,
    agg_stats: pd.DataFrame
) -> dict:
    """Analisa uma nova transação usando ensemble de modelos."""
    tipo_tx = nova_tx.get("tipo_transacao", "").lower()
    id_conta = nova_tx.get("id_conta_destino" if tipo_tx == "recebimento" else "id_conta_origem")
    
    try:
        id_conta = int(id_conta)
    except:
        return {"decision": "erro", "final_score": 0.0, "reasons": ["ID inválido"]}
    
    hist_conta = historico_df[
        (historico_df["id_conta_origem"] == id_conta) |
        (historico_df["id_conta_destino"] == id_conta)
    ].copy()
    
    tx_time = pd.to_datetime(nova_tx.get("data_operacao"), errors='coerce')
    if pd.isna(tx_time):
        return {"decision": "erro", "final_score": 0.0, "reasons": ["Data inválida"]}
    
    hour = int(tx_time.hour)
    
    if not hist_conta.empty:
        hist_conta["data_operacao"] = pd.to_datetime(hist_conta["data_operacao"], errors='coerce')
        last_tx = hist_conta["data_operacao"].max()
        time_since_h = (tx_time - last_tx).total_seconds() / 3600.0 if pd.notna(last_tx) else 9999.0
    else:
        time_since_h = 9999.0
    
    conta_info = contas_df[contas_df["id_conta"] == id_conta]
    hour_pref_match = 0
    local_match = 0
    canal_match = 0
    
    if not conta_info.empty:
        conta_info = conta_info.iloc[0]
        prefs = conta_info.get("horarios_preferidos", [])
        if isinstance(prefs, list):
            hour_pref_match = int(hour in prefs)
        local_match = int(nova_tx.get("local_origem") == conta_info.get("localizacao"))
        canal_match = int(nova_tx.get("canal") == conta_info.get("canal_preferido"))
    
    valor = float(nova_tx["valor"])
    valor_log = np.log1p(valor)
    valor_zscore = (valor - historico_df["valor"].mean()) / (historico_df["valor"].std() + 1e-8)
    
    day_of_week = tx_time.dayofweek
    is_weekend = int(day_of_week in [5, 6])
    is_night = int(hour in range(0, 6))
    
    X_new = np.array([[
        valor_log, valor_zscore, hour, day_of_week, is_weekend,
        is_night, time_since_h, hour_pref_match, local_match, canal_match
    ]])
    
    X_scaled = models["scaler"].transform(X_new)
    
    scores = {}
    reasons = []
    
    # IsolationForest
    iso_score = models["iso"].decision_function(X_scaled)[0]
    iso_pred = models["iso"].predict(X_scaled)[0]
    iso_norm = 1.0 / (1.0 + np.exp(iso_score))
    scores["iso"] = iso_norm
    reasons.append(f"IsolationForest: {'⚠️ outlier' if iso_pred == -1 else '✅ normal'} (score={iso_norm:.3f})")
    
    # RandomForest
    if models["rf"] is not None:
        rf_proba = models["rf"].predict_proba(X_scaled)[0, 1]
        scores["rf"] = rf_proba
        reasons.append(f"RandomForest: prob(fraude)={rf_proba:.3f}")
    
    # GradientBoosting/XGBoost
    if models["gb"] is not None:
        gb_proba = models["gb"].predict_proba(X_scaled)[0, 1]
        scores["gb"] = gb_proba
        model_name = "XGBoost" if XGBOOST_AVAILABLE else "GradientBoosting"
        reasons.append(f"{model_name}: prob(fraude)={gb_proba:.3f}")
    
    # LabelPropagation
    if models["lp"] is not None:
        lp_pred = models["lp"].predict(X_scaled)[0]
        scores["lp"] = float(lp_pred)
        reasons.append(f"LabelPropagation: pred={lp_pred}")
    
    bonus_score = 0.0
    
    if id_conta in agg_stats.index:
        agg = agg_stats.loc[id_conta]
        
        if agg["cnt_tx"] >= 3:
            limite = agg["mean_valor"] + 3 * max(agg["std_valor"], 1)
            if valor > limite:
                excesso = valor / (agg["mean_valor"] + 1)
                bonus_score += min(0.4, 0.2 * np.log10(excesso))
                reasons.append(f"💰 Valor anômalo: R${valor:.2f} >> média R${agg['mean_valor']:.2f}")
            
            if hour_pref_match == 0 and agg["pct_hour_pref"] > 0.7:
                bonus_score += 0.15
                reasons.append(f"⏰ Horário atípico (fora do padrão do usuário)")
            
            if local_match == 0 and agg["pct_local_match"] > 0.8:
                bonus_score += 0.1
                reasons.append(f"📍 Local diferente do habitual")
    else:
        reasons.append("❓ Conta com pouco histórico (cold-start)")
    
    # Ensemble
    final_score = (
        scores.get("gb", 0) * 0.35 +
        scores.get("rf", 0) * 0.30 +
        scores.get("iso", 0) * 0.20 +
        scores.get("lp", 0) * 0.10 +
        bonus_score * 0.05
    )
    
    if final_score > 0.7:
        decision = "🚨 SUSPEITA ALTA"
        emoji = "🚨"
    elif final_score > 0.4:
        decision = "⚠️ SUSPEITA BAIXA"
        emoji = "⚠️"
    else:
        decision = "✅ NORMAL"
        emoji = "✅"
    
    return {
        "decision": decision,
        "emoji": emoji,
        "final_score": round(final_score, 3),
        "scores_individuais": {k: round(v, 3) for k, v in scores.items()},
        "reasons": reasons,
        "id_conta": id_conta,
        "tipo_transacao": tipo_tx,
        "valor": valor,
        "timestamp": tx_time.isoformat()
    }


# ============================================================================
# SISTEMA DE MONITORAMENTO (NOVO)
# ============================================================================

class MonitoradorFraudes(FileSystemEventHandler):
    """Classe para monitorar mudanças no arquivo JSON."""
    
    def __init__(self, caminho_json: str):
        self.caminho_json = caminho_json
        self.ultimo_tamanho = 0
        self.contas = None
        self.historico = None
        self.models = None
        self.agg = None
        self.inicializar()
    
    def inicializar(self):
        """Carrega base inicial e treina modelos."""
        logger.info("=" * 70)
        logger.info("🚀 INICIALIZANDO SISTEMA DE DETECÇÃO DE FRAUDES")
        logger.info("=" * 70)
        
        try:
            self.contas, self.historico = carregar_base(self.caminho_json)
            logger.info(f"✅ Base carregada: {len(self.contas)} contas, {len(self.historico)} transações")
            
            self.historico["data_operacao"] = pd.to_datetime(self.historico["data_operacao"], errors='coerce')
            
            features, _ = extrair_features_transacoes(self.historico, self.contas)
            self.agg = agregados_por_conta(features)
            
            logger.info("🔨 Treinando modelos de ML...")
            self.models = treinar_modelos(features)
            logger.info("✅ Modelos treinados com sucesso!")
            
            self.ultimo_tamanho = len(self.historico)
            
            logger.info("=" * 70)
            logger.info(f"👀 MONITORAMENTO ATIVO - Arquivo: {self.caminho_json}")
            logger.info(f"📊 Transações na base: {self.ultimo_tamanho}")
            logger.info("💡 Adicione novas transações ao JSON para análise automática")
            logger.info("⌨️  Pressione Ctrl+C para encerrar")
            logger.info("=" * 70)
            logger.info("")
            
        except Exception as e:
            logger.error(f"❌ Erro na inicialização: {e}")
            raise
    
    def on_modified(self, event):
        """Callback executado quando o arquivo é modificado."""
        if not event.src_path.endswith(os.path.basename(self.caminho_json)):
            return
        
        # Aguarda para garantir que o arquivo foi completamente escrito
        time.sleep(0.5)
        
        try:
            contas_new, historico_new = carregar_base(self.caminho_json)
            
            if len(historico_new) > self.ultimo_tamanho:
                qtd_novas = len(historico_new) - self.ultimo_tamanho
                
                logger.info("\n" + "🔔" * 35)
                logger.info(f"📥 {qtd_novas} NOVA(S) TRANSAÇÃO(ÕES) DETECTADA(S)!")
                logger.info("🔔" * 35 + "\n")
                
                # Analisa cada nova transação
                novas_txs = historico_new.iloc[self.ultimo_tamanho:]
                
                for idx, tx in novas_txs.iterrows():
                    tx_dict = tx.to_dict()
                    resultado = analisar_transacao(
                        tx_dict, 
                        self.contas, 
                        self.historico, 
                        self.models, 
                        self.agg
                    )
                    
                    self.exibir_resultado(resultado, idx + 1)
                
                # Atualiza histórico
                self.historico = historico_new
                self.ultimo_tamanho = len(historico_new)
                
                logger.info("\n" + "─" * 70 + "\n")
            
        except json.JSONDecodeError:
            logger.warning("⚠️ Arquivo JSON temporariamente inválido (aguardando gravação completa)...")
        except Exception as e:
            logger.error(f"❌ Erro ao processar novas transações: {e}")
    
    def exibir_resultado(self, resultado: dict, numero: int):
        """Exibe resultado formatado da análise."""
        print("┌" + "─" * 68 + "┐")
        print(f"│ 🧾 TRANSAÇÃO #{numero:04d}")
        print("├" + "─" * 68 + "┤")
        print(f"│ ID Conta: {resultado['id_conta']:<25} Tipo: {resultado['tipo_transacao']:<15} │")
        print(f"│ Valor: R$ {resultado['valor']:>10,.2f}{'':>35} │")
        print(f"│ Data/Hora: {resultado['timestamp']:<50} │")
        print("├" + "─" * 68 + "┤")
        print(f"│ {resultado['emoji']} DECISÃO: {resultado['decision']:<50} │")
        print(f"│ 📊 Score Final: {resultado['final_score']:.3f} (0=normal ➜ 1=fraude){'':>20} │")
        print("├" + "─" * 68 + "┤")
        print("│ 📈 Scores Individuais:                                              │")
        
        for modelo, score in resultado['scores_individuais'].items():
            nome_modelo = {
                'iso': 'IsolationForest',
                'rf': 'RandomForest',
                'gb': 'XGBoost/GradBoosting',
                'lp': 'LabelPropagation'
            }.get(modelo, modelo)
            print(f"│    • {nome_modelo:<20}: {score:.3f}{'':>30} │")
        
        print("├" + "─" * 68 + "┤")
        print("│ 💡 Explicações:                                                     │")
        
        for reason in resultado['reasons']:
            # Quebra linhas longas
            if len(reason) <= 60:
                print(f"│    {reason:<64} │")
            else:
                words = reason.split()
                line = "   "
                for word in words:
                    if len(line + " " + word) <= 60:
                        line += " " + word
                    else:
                        print(f"│ {line:<67} │")
                        line = "    " + word
                if line.strip():
                    print(f"│ {line:<67} │")
        
        print("└" + "─" * 68 + "┘")


def monitorar_arquivo(caminho_json: str = "base_pix_simulada.json"):
    """Inicia o monitoramento do arquivo JSON."""
    caminho_abs = os.path.abspath(caminho_json)
    
    if not os.path.exists(caminho_abs):
        logger.error(f"❌ Arquivo não encontrado: {caminho_abs}")
        return
    
    # Cria observador
    event_handler = MonitoradorFraudes(caminho_abs)
    observer = Observer()
    observer.schedule(event_handler, os.path.dirname(caminho_abs), recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\n\n🛑 Encerrando monitoramento...")
        observer.stop()
    
    observer.join()
    logger.info("✅ Sistema encerrado com sucesso!")


# ============================================================================
# INTERFACE DE LINHA DE COMANDO
# ============================================================================

def main():
    """Função principal com argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="🔍 Sistema de Detecção de Fraudes em Transações PIX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python fraud_detection.py --modo monitorar
  python fraud_detection.py --modo monitorar --arquivo minha_base.json
  python fraud_detection.py --modo testar
        """
    )
    
    parser.add_argument(
        "--modo",
        type=str,
        choices=["monitorar", "testar"],
        default="monitorar",
        help="Modo de operação (padrão: monitorar)"
    )
    
    parser.add_argument(
        "--arquivo",
        type=str,
        default="base_pix_simulada.json",
        help="Caminho do arquivo JSON (padrão: base_pix_simulada.json)"
    )
    
    args = parser.parse_args()
    
    if args.modo == "monitorar":
        monitorar_arquivo(args.arquivo)
    
    elif args.modo == "testar":
        # Modo de teste com transação fictícia
        logger.info("🧪 Modo de Teste - Analisando transação fictícia")
        
        transacao_teste = {
            "id_conta": 664,
            "valor": 8700.00,
            "data_operacao": "2024-11-26T03:30:00",
            "tipo_transacao": "envio",
            "id_conta_origem": 664,
            "id_conta_destino": 49,
            "canal": "web",
            "local_origem": "Local Desconhecido"
        }
        
        try:
            contas, historico = carregar_base(args.arquivo)
            historico["data_operacao"] = pd.to_datetime(historico["data_operacao"], errors='coerce')
            
            features, _ = extrair_features_transacoes(historico, contas)
            agg = agregados_por_conta(features)
            models = treinar_modelos(features)
            
            resultado = analisar_transacao(transacao_teste, contas, historico, models, agg)
            
            # Exibe resultado
            monitor = MonitoradorFraudes(args.arquivo)
            monitor.exibir_resultado(resultado, 1)
            
        except FileNotFoundError:
            logger.error(f"❌ Arquivo '{args.arquivo}' não encontrado!")
            logger.info("\n💡 Certifique-se de que o arquivo JSON existe no diretório atual")


if __name__ == "__main__":
    main()