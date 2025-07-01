from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
import io, os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy.stats import zscore

app = FastAPI()

# CORS setup for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload/")
async def upload_file(
    file: UploadFile = File(...),
    handleMissing: bool = Form(...),
    missingMethod: str = Form(None),
    encodeCategorical: bool = Form(...),
    encodingMethod: str = Form(None),
    scaleNumeric: bool = Form(...),
    scaleMethod: str = Form(None),
    handleOutliers: bool = Form(...),
    outlierMethod: str = Form(None),
):
    contents = await file.read()
    filename = file.filename

    # Read into DataFrame
    if filename.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(contents))
    else:
        df = pd.read_excel(io.BytesIO(contents))

    # Missing values
    if handleMissing:
        if missingMethod == "drop":
            df = df.dropna()
        elif missingMethod == "mean":
            df = df.fillna(df.mean(numeric_only=True))
        elif missingMethod == "median":
            df = df.fillna(df.median(numeric_only=True))
        elif missingMethod == "mode":
            df = df.fillna(df.mode().iloc[0])
        elif missingMethod == "constant":
            df = df.fillna(0)

    # Categorical encoding
    if encodeCategorical:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        if encodingMethod == "onehot":
            df = pd.get_dummies(df, columns=cat_cols)
        elif encodingMethod == "label":
            for col in cat_cols:
                df[col] = df[col].astype("category").cat.codes

    # Scaling
    if scaleNumeric:
        num_cols = df.select_dtypes(include="number").columns
        scaler = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler()
        }.get(scaleMethod, StandardScaler())
        df[num_cols] = scaler.fit_transform(df[num_cols])

    # Outlier handling
    if handleOutliers:
        num_cols = df.select_dtypes(include="number").columns
        if outlierMethod == "zscore":
            z_scores = df[num_cols].apply(zscore)
            df = df[(z_scores < 3).all(axis=1)]
        elif outlierMethod == "iqr":
            Q1 = df[num_cols].quantile(0.25)
            Q3 = df[num_cols].quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df[num_cols] < (Q1 - 1.5 * IQR)) | (df[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

    # Save processed file
    os.makedirs("processed", exist_ok=True)
    output_name = f"processed_{filename.split('.')[0]}.csv"
    output_path = f"processed/{output_name}"
    df.to_csv(output_path, index=False)

    return {"message": "Processed successfully", "download_url": f"/download/{output_name}"}

@app.get("/download/{filename}")
def download_file(filename: str):
    file_path = f"processed/{filename}"
    return FileResponse(file_path, filename=filename)
