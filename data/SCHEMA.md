# Esquema atual (baseado nos dados presentes no repositório)

Campos principais (obrigatórios para o pipeline de treino/avaliação):
- customer_segment: str  
  - Ex.: "SME", "Startup", "Enterprise", "Consumer"
- channel: str  
  - Ex.: "email", "chat", "phone", "portal"
- preprocessed_text: str  (campo textual principal — texto normalizado pré‑computado; obrigatório)
  - Ex.: "meu sistema esta fora do ar desde ontem e nao consigo emitir notas"
- category: str  (rótulo alvo — obrigatório)
  - Ex.: "technical_issue", "billing", "cancellation", "account_management", "complaint"
- priority: str  (feature categórica — obrigatório na maioria dos registros; valores: "high","medium","low")

Campos opcionais / históricos (podem existir em alguns registros):
- text: str (texto bruto original — pode existir; quando presente, preprocessed_text deve conter a versão normalizada)
- product: str
- invoice_amount: number | null
- service_down: bool
- affected_users_count: int | null
- intent: str
- entities: object (JSON)
- label_confidence: float (0.0-1.0)
- metadata (version, source_system, attachments_reference, etc.)

Observações e regras práticas
- A pipeline de produção e avaliação deve priorizar preprocessed_text como entrada textual. Se apenas "text" existir, gere preprocessed_text com o mesmo pré‑processamento adotado no projeto (lowercase, remoção de acentos e pontuação, colapso de espaços).
- Ordem recomendada ao serializar JSONs de saída (para legibilidade/consistência): customer_segment, channel, text (se existir), preprocessed_text, category, priority.
- Evitar vazamento: ao criar splits de treino/teste use GroupShuffleSplit por preprocessed_text ou remova duplicatas exatas antes do split.
- Versione o pré‑processamento (campo ou nota) para garantir reprodutibilidade do modelo.