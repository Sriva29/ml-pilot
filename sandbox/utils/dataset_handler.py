import pandas as pd
from io import BytesIO

class DatasetHandler:
    def load(self, uploaded_file: BytesIO) -> pd.DataFrame:
        """Reads an uploaded CSV/Excel/JSON into a DataFrame."""
        pass
