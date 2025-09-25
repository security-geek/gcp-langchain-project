# GCP LangChain AI Project

This repository demonstrates an end-to-end AI pipeline using **Vertex AI**, **BigQuery**, and **LangChain** for embedding PDFs, storing embeddings, and querying them interactively.  

---

## **Files**

### 1. `qa_vertexai.py`
- Takes a PDF file already in **GCPBucket**.
- Allows a user to ask questions interactively.
- Retrieves the top-k relevant chunks using in-memory embeddings.

### 2. `qa_vertexai1.py`
- Similar functionality as `qa_vertexai.py`.
- Additionally **creates a `.json` file locally** to inspect the embeddings.

### 3. `qa_vertexai_bigquery.py`
- Reads PDF files from a **GCS bucket**.
- Splits text into chunks and computes embeddings using **Vertex AI** (`text-embedding-004`).
- Stores chunk text and normalized embeddings in **BigQuery** (`FLOAT64 REPEATED` column).
- Supports multiple PDF files.

---

## **Pipeline Overview**

1. **Input:** PDF documents (from GCS or BigQuery).  
2. **Processing:**  
   - Text extraction and chunking  
   - Embedding computation using Vertex AI  
3. **Storage:**  
   - Store chunks and embeddings in BigQuery  
4. **Query:**  
   - Retrieve top-k chunks for a user question  
   - Optional: Create local JSON of embeddings  

---

## **Prerequisites**

- Python 3.10+  
- Google Cloud SDK and service account with BigQuery & Storage access  
- `vertexai` Python library  
- `PyPDF2` and `numpy`  
- BigQuery table schema:  

| Column       | Type   | Mode      |
|--------------|--------|-----------|
| chunk_id     | INT64  | REQUIRED  |
| chunk_text   | STRING | NULLABLE  |
| embedding    | FLOAT  | REPEATED  |

---

## **Usage**

1. Place your PDFs in GCS or BigQuery.  
2. Update configuration variables in the scripts (`PROJECT_ID`, `BUCKET`, `PDF_FILES`, `BQ_DATASET`, `BQ_TABLE`).  
3. Run the scripts:  

```bash
python3 qa_vertexai_bigquery.py   # compute & store embeddings along with Interactive Q&A
python3 qa_vertexai1.py           # create embeddings JSON locally and Interactive Q&A
python3 qa_vertexai.py            # interactive Q&A

Notes

qa_vertexai_bigquery.py appends embeddings; it does not delete existing rows.

Ensure embedding column in BigQuery is REPEATED FLOAT64 to store vectors.

Supports multiple PDFs with chunking and normalized embeddings for similarity search.

License

This project is for learning and training purposes.
