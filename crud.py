import json
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")

# ====================================================== rodar no cmd streamlit run crud.py


ARQUIVO_JSON = "base_pix_simulada.json"

# ======================================================
# FUNÇÕES AUXILIARES
# ======================================================
def carregar_dados(caminho):
    """Lê o arquivo JSON e retorna dicionário + DataFrames"""
    try:
        with open(caminho, "r", encoding="utf-8") as f:
            dados = json.load(f)

        contas_df = pd.DataFrame(dados.get("contas_bancarias", []))
        trans_df = pd.DataFrame(dados.get("historico_pix", []))

        if not trans_df.empty and "data_operacao" in trans_df.columns:
            trans_df["data_operacao"] = pd.to_datetime(
                trans_df["data_operacao"], errors="coerce", infer_datetime_format=True
            )

        return dados, contas_df, trans_df

    except FileNotFoundError:
        st.error(f"Arquivo '{caminho}' não encontrado.")
        return {"contas_bancarias": [], "historico_pix": []}, pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo JSON: {e}")
        return {"contas_bancarias": [], "historico_pix": []}, pd.DataFrame(), pd.DataFrame()


def salvar_dados(caminho, dados):
    """Salva o dicionário de dados no JSON"""
    try:
        with open(caminho, "w", encoding="utf-8") as f:
            json.dump(dados, f, ensure_ascii=False, indent=4)
        st.success("✅ Dados salvos com sucesso!")
    except Exception as e:
        st.error(f"Erro ao salvar dados: {e}")


# ======================================================
# INTERFACE STREAMLIT
# ======================================================
st.set_page_config(page_title="CRUD e Visualizador de Transações PIX", layout="wide")
st.title("💳 Sistema PIX — CRUD e Visualização de Contas")

# Recarregar dados
if st.button("🔄 Recarregar dados do JSON"):
    st.cache_data.clear()

dados, contas_df, trans_df = carregar_dados(ARQUIVO_JSON)

if contas_df.empty:
    st.warning("Nenhuma conta encontrada no arquivo JSON.")
    st.stop()

# Barra lateral
aba = st.sidebar.radio(
    "Escolha uma funcionalidade:",
    ["📊 Visualizar Conta", "💸 Realizar Transação"],
)

# ======================================================
# ABA 1 - VISUALIZAÇÃO DE CONTA
# ======================================================
if aba == "📊 Visualizar Conta":
    st.header("📊 Visualização de Conta")

    conta_escolhida = st.selectbox("Selecione o ID da conta:", contas_df["id_conta"])

    trans_conta = trans_df[trans_df["id_conta"] == conta_escolhida].copy()
    conta_info = contas_df[contas_df["id_conta"] == conta_escolhida].iloc[0]

    # ------------------------
    # INFORMAÇÕES DA CONTA
    # ------------------------
    st.subheader("📌 Informações da Conta")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("👤 Titular", conta_info.get("nome_titular", "N/D"))
        st.metric("🏦 Banco", conta_info.get("banco", "N/D"))
    with col2:
        st.metric("📍 Localização", conta_info.get("localizacao", "N/D"))
        st.metric("💳 Canal Preferido", conta_info.get("canal_preferido", "N/D"))
    with col3:
        st.metric("💰 Saldo Atual", f"R$ {conta_info.get('saldo_inicial', 0):.2f}")
        horarios = conta_info.get("horarios_preferidos", [])
        st.metric("🕒 Horários Pref.", ", ".join(map(str, horarios)) if horarios else "N/D")

    # ------------------------
    # ESTATÍSTICAS
    # ------------------------
    if not trans_conta.empty:
        st.subheader("📈 Estatísticas")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nº Transações", len(trans_conta))
        with col2:
            st.metric("Valor Médio", f"R$ {trans_conta['valor'].mean():.2f}")
        with col3:
            st.metric("Maior Valor", f"R$ {trans_conta['valor'].max():.2f}")

        # ------------------------
        # GRÁFICOS
        # ------------------------
        st.subheader("📊 Visualizações")
        colA, colB = st.columns(2)

        # Série temporal
        with colA:
            fig1, ax1 = plt.subplots(figsize=(5, 3))
            if "flag_suspeita" in trans_conta.columns:
                normal = trans_conta[~trans_conta["flag_suspeita"].fillna(False)]
                suspeitas = trans_conta[trans_conta["flag_suspeita"].fillna(False)]
            else:
                normal = trans_conta
                suspeitas = pd.DataFrame()
            ax1.plot(normal["data_operacao"], normal["valor"], 'o', alpha=0.6, label="Normal")
            if not suspeitas.empty:
                ax1.plot(suspeitas["data_operacao"], suspeitas["valor"], 'ro', label="Suspeita", markersize=6)
            ax1.set_title("Série Temporal de Transações")
            ax1.set_xlabel("Data")
            ax1.set_ylabel("Valor (R$)")
            ax1.legend()
            st.pyplot(fig1)

        # Histograma
        with colB:
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            ax2.hist(trans_conta["valor"], bins=20, alpha=0.7, color="blue")
            ax2.set_title("Distribuição de Valores")
            ax2.set_xlabel("Valor (R$)")
            ax2.set_ylabel("Frequência")
            st.pyplot(fig2)

        # Horários e Locais
        colC, colD = st.columns(2)

        with colC:
            trans_conta["hora"] = trans_conta["data_operacao"].dt.hour
            fig3, ax3 = plt.subplots(figsize=(5, 3))
            ax3.hist(trans_conta["hora"], bins=24, alpha=0.7, color="green", rwidth=0.8)
            ax3.set_xticks(range(24))
            ax3.set_title("Horários das Transações")
            ax3.set_xlabel("Hora do Dia")
            ax3.set_ylabel("Nº Transações")
            st.pyplot(fig3)

        with colD:
            fig4, ax4 = plt.subplots(figsize=(5, 3))
            campo_local = None
            for candidato in ["local_origem", "local", "origem"]:
                if candidato in trans_conta.columns:
                    campo_local = candidato
                    break
            if campo_local:
                trans_conta[campo_local].value_counts().head(10).plot(kind="bar", ax=ax4, color="orange")
                ax4.set_title("Locais Mais Usados")
                ax4.set_ylabel("Nº Transações")
            else:
                ax4.text(0.5, 0.5, "Campo de local não encontrado", ha='center', va='center')
                ax4.set_axis_off()
            st.pyplot(fig4)

        # Canais usados
        st.subheader("📡 Canais Utilizados")
        if "canal" in trans_conta.columns:
            fig5, ax5 = plt.subplots(figsize=(5, 3))
            trans_conta["canal"].value_counts().plot(kind="pie", autopct='%1.1f%%', ax=ax5)
            ax5.set_ylabel("")
            ax5.set_title("Distribuição de Canais")
            st.pyplot(fig5)

        # Transações suspeitas
        st.subheader("⚠️ Transações Suspeitas")
        if not suspeitas.empty:
            st.dataframe(suspeitas.sort_values("data_operacao"))
        else:
            st.info("Nenhuma transação suspeita encontrada.")

    else:
        st.warning("Essa conta ainda não possui transações registradas.")


# ======================================================
# ABA 2 - REALIZAR TRANSAÇÃO ENTRE CONTAS
# ======================================================
elif aba == "💸 Realizar Transação":
    st.header("💸 Nova Transação entre Contas")

    col1, col2 = st.columns(2)
    with col1:
        origem_id = st.selectbox("Conta de origem", contas_df["id_conta"])
    with col2:
        destino_id = st.selectbox("Conta de destino", contas_df["id_conta"])

    valor = st.number_input("Valor da transação (R$)", min_value=0.01, step=0.01)
    canal = st.selectbox("Canal", ["App", "Internet Banking", "ATM", "Outros"])
    local = st.text_input("Local de origem", "Online")

    if st.button("🚀 Executar Transação"):
        if origem_id == destino_id:
            st.error("A conta de origem e destino não podem ser iguais.")
        else:
            conta_origem = next(c for c in dados["contas_bancarias"] if c["id_conta"] == origem_id)
            conta_destino = next(c for c in dados["contas_bancarias"] if c["id_conta"] == destino_id)

            if conta_origem["saldo_inicial"] < valor:
                st.error("❌ Saldo insuficiente na conta de origem.")
            else:
                # Atualiza saldos
                conta_origem["saldo_inicial"] -= valor
                conta_destino["saldo_inicial"] += valor

                # Cria registro de transação
                nova_transacao = {
                    "id_conta": origem_id,
                    "id_destino": destino_id,
                    "data_operacao": datetime.now().isoformat(),
                    "valor": valor,
                    "canal": canal,
                    "local_origem": local,
                    "flag_suspeita": False,
                    "id_conta_origem": origem_id

                }

                # Grava no JSON
                dados["historico_pix"].append(nova_transacao)
                salvar_dados(ARQUIVO_JSON, dados)

                st.success(f"✅ Transação de R$ {valor:.2f} realizada com sucesso!")
                st.balloons()
