graph TD
    subgraph "Phase 1: Offline Training Pipeline (Proj (1).ipynb)"
        direction TB
        
        %% --- 1. Data Input ---
        A[movies_metadata.csv\n(45,000+ Movies)]
        
        %% --- 2. Cleaning ---
        A --> B(Data Cleaning & Filtering\n- Runtime 60-200 min\n- Popularity < 12\n- Drop Duplicates)
        B --> C[32,993 Clean Movies]

        %% --- 3. Parallel Pipelines ---
        C --> D(Tabular Data Pipeline)
        C --> E(Text Data Pipeline)

        %% --- 4. Tabular Path ---
        D --> F(Run 'build_preprocessing_assets.py')
        F --> G[language_columns.json]
        F --> H[genre_columns.json]
        
        D --> I(Build Preprocessor)
        I --> J[preprocessor.pkl]
        
        D --> K(Train Autoencoder)
        K --> L[encoder_model_last.keras]
        
        D --> M(Run 'convert_df_to_json.py')
        M --> N[movie_details.json]

        %% --- 5. Text Path ---
        E --> O(Run SBERT Model\n'all-MiniLM-L6-v2')
        O --> P[384-dim Text Embeddings]

        %% --- 6. Combine & Create Final Assets ---
        L --> Q(Get 32-dim Tabular Embeddings)
        P & Q --> R(Combine Embeddings)
        R --> S[416-dim Hybrid Embeddings]
        
        S & C --> T(Build Dataframe)
        T --> U[embeddings.csv]
        
        S --> V(Build FAISS Index)
        V --> W[prebuilt_index.faiss]
    end

    %% --- Styling ---
    style A fill:#ffe0e0,stroke:#333,stroke-width:2px
    style C fill:#d4f7d4,stroke:#333,stroke-width:2px
    style G fill:#f9f,stroke:#333
    style H fill:#f9f,stroke:#333
    style J fill:#f9f,stroke:#333
    style L fill:#f9f,stroke:#333
    style N fill:#f9f,stroke:#333
    style U fill:#f9f,stroke:#333
    style W fill:#f9f,stroke:#333


graph TD
    subgraph "Phase 2: Live Streamlit App (app.py)"
        direction TB

        %% --- 1. App Start ---
        AA(App Start: load_resources)
        AA --> A[Load All Assets\n- encoder.keras\n- sbert_model\n- preprocessor.pkl\n- prebuilt_index.faiss\n- movie_details.json\n- embeddings.csv]
        
        A --> B(User Input: 'Movie Title')
        
        %% --- 2. Decision ---
        B --> C{Is Movie in DB?}
        
        %% --- 3. Path A: New Movie ---
        C -- No --> D(Scrap2.py: fetch_movie_data_for_processing)
        D --> E(Live Pipeline: auto_processing_for_movie\n- Dummies.py\n- preprocessor.pkl\n- encoder.keras\n- sbert_model)
        E --> F[New 416-dim Embedding]
        
        D --> G(Scrap2.py: fetch_movie_data_for_display)
        G --> H[New Display Data]
        
        F & H --> I(Update In-Memory DB\n- Add to FAISS index\n- Add to 'all_titles'\n- Add to 'all_movie_details')
        I --> J(Get FAISS Recommendations)
        
        %% --- 4. Path B: Existing Movie ---
        C -- Yes --> K(Get Pre-computed Embedding\nfrom 'title_to_embedding' map)
        K --> J

        %% --- 5. Display ---
        J --> L(Get Reco Details\nfrom 'all_movie_details')
        L --> M{Poster/Rating Missing?}
        M -- Yes --> N(Scrap2.py: Patch Data\n(with dataset_year_hint))
        M -- No --> O(Display Results)
        N --> O
    end

    %% --- Styling ---
    style A fill:#d4f7d4,stroke:#333,stroke-width:2px
    style C fill:#ffedc7,stroke:#333,stroke-width:2px
    style E fill:#e0e8ff,stroke:#333,stroke-width:2px
    style F fill:#d4f7d4,stroke:#333
    style J fill:#e0e8ff,stroke:#333,stroke-width:2px
    style O fill:#d4f7d4,stroke:#333,stroke-width:2px
