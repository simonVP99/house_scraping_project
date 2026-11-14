"""
Embed property descriptions using multilingual-e5-small and PCA-reduce to 50 dims.
Produces property_data_with_embeddings.csv for model training.
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
import joblib
import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_NAME = 'intfloat/multilingual-e5-small'
PCA_COMPONENTS = 50
BATCH_SIZE = 32
MAX_LENGTH = 512


def load_and_join_descriptions():
    """Load preprocessed data and join descriptions from parsed CSV."""
    print("Loading preprocessed data...")
    df_prep = pd.read_csv(os.path.join(OUTPUT_DIR, 'property_data_preprocessed.csv'))
    print(f"  Preprocessed: {len(df_prep):,} rows x {len(df_prep.columns)} columns")

    print("Loading descriptions from parsed data...")
    df_desc = pd.read_csv(
        os.path.join(OUTPUT_DIR, 'property_data_parsed.csv'),
        usecols=['property_id', 'description'],
        dtype={'property_id': str, 'description': str},
    )
    print(f"  Parsed descriptions: {len(df_desc):,} rows")

    # Dedup on property_id, keep last occurrence
    df_desc = df_desc.drop_duplicates(subset='property_id', keep='last')
    print(f"  After dedup: {len(df_desc):,} unique property_ids")

    # Join
    df_prep['property_id'] = df_prep['property_id'].astype(str)
    df = df_prep.merge(df_desc, on='property_id', how='left')
    matched = df['description'].notna().sum()
    print(f"  Matched {matched:,}/{len(df):,} rows with descriptions ({matched/len(df)*100:.1f}%)")

    return df


def clean_descriptions(df):
    """Clean description text."""
    # Replace language-only markers with empty string
    language_markers = re.compile(
        r'^(English|Français|Frans|French|Nederlands|Dutch|Deutsch|German)$',
        re.IGNORECASE
    )

    def clean(text):
        if pd.isna(text) or not isinstance(text, str):
            return ''
        text = text.strip()
        if language_markers.match(text):
            return ''
        return text

    df['description'] = df['description'].apply(clean)

    # Add binary flag for meaningful descriptions (>50 chars)
    df['has_meaningful_description'] = (df['description'].str.len() > 50).astype(int)
    meaningful = df['has_meaningful_description'].sum()
    print(f"  Meaningful descriptions (>50 chars): {meaningful:,}/{len(df):,} ({meaningful/len(df)*100:.1f}%)")

    return df


def embed_texts(texts, model, tokenizer, device):
    """Embed a list of texts using E5 model with mean pooling."""
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        # E5 requires "passage: " prefix for document encoding
        batch = [f"passage: {t}" if t else "passage: " for t in batch]

        encoded = tokenizer(
            batch,
            max_length=MAX_LENGTH,
            padding=True,
            truncation=True,
            return_tensors='pt',
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded)

        # Mean pooling over non-padding tokens
        attention_mask = encoded['attention_mask']
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1).clamp(min=1e-9)

        all_embeddings.append(embeddings.cpu().numpy())

        if (i // BATCH_SIZE + 1) % 50 == 0 or i + BATCH_SIZE >= len(texts):
            print(f"    Embedded {min(i + BATCH_SIZE, len(texts)):,}/{len(texts):,} texts")

    return np.vstack(all_embeddings)


def main():
    print("=" * 70)
    print("DESCRIPTION EMBEDDING PIPELINE")
    print("=" * 70)

    # Step 1: Load and join
    df = load_and_join_descriptions()

    # Step 2: Clean descriptions
    print("\nCleaning descriptions...")
    df = clean_descriptions(df)

    # Step 3: Set up model
    print("\nLoading embedding model...")
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"  Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    print(f"  Model loaded: {MODEL_NAME}")

    # Step 4: Embed
    print("\nEmbedding descriptions...")
    texts = df['description'].fillna('').tolist()
    embeddings = embed_texts(texts, model, tokenizer, device)
    print(f"  Raw embeddings shape: {embeddings.shape}")

    # Step 5: PCA reduce
    print(f"\nPCA reducing {embeddings.shape[1]} dims -> {PCA_COMPONENTS} components...")
    pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
    embeddings_reduced = pca.fit_transform(embeddings)
    variance_explained = pca.explained_variance_ratio_.sum() * 100
    print(f"  Variance explained: {variance_explained:.1f}%")

    # Save PCA model
    pca_path = os.path.join(MODEL_DIR, 'description_pca.joblib')
    joblib.dump(pca, pca_path)
    print(f"  PCA model saved to {pca_path}")

    # Step 6: Append embedding columns
    emb_cols = [f'desc_emb_{i}' for i in range(PCA_COMPONENTS)]
    emb_df = pd.DataFrame(embeddings_reduced, columns=emb_cols, index=df.index)
    df = pd.concat([df, emb_df], axis=1)

    # Drop raw description column
    df = df.drop(columns=['description'])

    print(f"\nFinal dataframe: {len(df):,} rows x {len(df.columns)} columns")

    # Step 7: Save
    output_path = os.path.join(OUTPUT_DIR, 'property_data_with_embeddings.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    print("\n" + "=" * 70)
    print("EMBEDDING PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
