def normalize_columns(df):
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]