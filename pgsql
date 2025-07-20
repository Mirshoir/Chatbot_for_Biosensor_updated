                       ┌──────────────────────────┐
                       │   👤 User Interface       │
                       │ (Streamlit Frontend UI)  │
                       └────────────┬─────────────┘
                                    │
                                    ▼
                       ┌──────────────────────────┐
                       │ 🔤 User Input (Text Box)  │
                       └────────────┬─────────────┘
                                    │
                                    ▼
        ┌──────────────────────────────────────────────────────┐
        │ 📊 Biometric Data Processor (cached every 5s)         │
        │ ┌──────────────────────┐  ┌────────────────────────┐ │
        │ │ Emotional State File │  │ ECG Stress Level File  │ │
        │ └──────────────────────┘  └────────────────────────┘ │
        │ ┌────────────────────────────┐                       │
        │ │ EMG Muscle Activation File │                       │
        │ └────────────────────────────┘                       │
        └────────────────────┬────────────────────────────────┘
                             │
                             ▼
           ┌────────────────────────────────────┐
           │ 🧠 Prompt Template Generator         │
           │ ├── General Chat                    │
           │ ├── Cognitive Load Analysis         │
           │ ├── Differential Diagnosis          │
           │ └── Clinic Plan                     │
           └────────────────────┬───────────────┘
                                │
                                ▼
                ┌────────────────────────────────┐
                │ 🤖 Mistral API Client           │
                │ ├── mistral-large-latest       │
                │ └── mistral-small-latest       │
                └────────────────┬───────────────┘
                                 │
                                 ▼
              ┌─────────────────────────────────────┐
              │ 💬 Bot Response (Natural Language)   │
              └────────────────┬────────────────────┘
                               │
                ┌──────────────▼─────────────┐
                │ 🗂 Save to ChromaDB         │
                │ (duckdb+parquet backend)   │
                │ ├── User Message Document  │
                │ ├── Bot Response Document  │
                │ └── Metadata + UUID        │
                └──────────────┬─────────────┘
                               │
                  ┌────────────▼─────────────┐
                  │ 🕘 Chat History Retrieval │
                  └────────────┬─────────────┘
                               │
                               ▼
              ┌────────────────────────────────────┐
              │ 📜 Display Past Chats (Markdown)   │
              └────────────────────────────────────┘
