import pandas as pd

def guess_target_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if "target" in col.lower() or "label" in col.lower() or "quality" in col.lower():
            return col
    return df.columns[-1]  # fallback to last column

def detect_task_type(df: pd.DataFrame, target_column: str) -> str:
    nunique = df[target_column].nunique()
    dtype = df[target_column].dtype

    if nunique <= 20 and dtype in ['int64', 'object']:
        return "classification"
    elif pd.api.types.is_numeric_dtype(dtype):
        return "regression"
    else:
        return "unknown"