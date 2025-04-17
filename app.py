from flask import Flask, request, jsonify
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
import pinecone
import pandas as pd
import re
from langdetect import detect
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
from uuid import uuid4
from llama_index.core import Settings

app = Flask(__name__)

# Load and preprocess data
df = pd.read_excel("Espace_Original_Batch1 (1) (1) (1) (2) (1).xlsx")
df = df.drop(0)
df = df.rename(columns={
    'exportateur': 'exporter',
    'total_facture': 'total_invoice',
    'montant_total_facture': 'total_invoice_amount',
    'total_quantite Calculated': 'total_quantity',
    'expediteur': 'sender',
    'destinataire': 'recipient',
    'poids_brut_total_pesee': 'gross_weight_kg',
    'adresse_expediteur': 'sender_address',
    'montant_fret': 'freight_amount',
    'origine': 'origin',
    'adresse_destinataire': 'recipient_address',
    'poids_net_kg': 'net_weight_kg',
    'poids_net': 'net_weight',
    'devise': 'currency',
    'paiement': 'payment_terms',
    'importateur': 'importer',
    'client': 'client',
    'conditions_livraison': 'delivery_terms',
    'accords': 'agreements',
})

def clean_number(x):
    if pd.isna(x): return None
    s = str(x)
    s = re.sub(r"[^\d\.\-]", "", s)
    try: return float(s)
    except: return None

for col in ['total_invoice', 'total_invoice_amount', 'total_quantity',
            'gross_weight_kg', 'freight_amount', 'net_weight_kg']:
    df[col] = df[col].apply(clean_number)

text_cols = ['exporter','sender','recipient','origin','importer','client',
             'sender_address','recipient_address','delivery_terms','agreements']

for col in text_cols:
    df[col] = df[col].fillna("").astype(str).str.replace(r"[\r\n]+"," ", regex=True).str.strip()

df.drop_duplicates(inplace=True)

def make_doc(row):
    def safe_int(val):
        if pd.isna(val):
            return 0
        try:
            return int(val)
        except:
            return 0

    return (
        f"Exporter: {row.exporter}\n"
        f"Invoice Total: {row.total_invoice} {row.currency}\n"
        f"Quantity: {safe_int(row.total_quantity)}\n"
        f"Sender: {row.sender} ({row.sender_address})\n"
        f"Recipient: {row.recipient} ({row.recipient_address})\n"
        f"Gross Weight (kg): {row.gross_weight_kg or 0}\n"
        f"Net Weight (kg): {row.net_weight_kg or 0}\n"
        f"Freight Amount: {row.freight_amount or 0}\n"
        f"Origin: {row.origin}\n"
        f"Payment Terms: {row.payment_terms}\n"
        f"Delivery Terms: {row.delivery_terms}\n"
        f"Agreements: {row.agreements}"
    )

df['document'] = df.apply(make_doc, axis=1)

# Build documents
documents = []
for _, row in df.iterrows():
    txt = row.document
    lang = detect(txt)
    documents.append(Document(text=txt, metadata={**row.drop("document").to_dict(), "lang": lang}))

# Initialize Pinecone
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = pinecone.Pinecone(api_key=pinecone_api_key)
index_name = "multilingual-e5-large"
pinecone_index = pc.Index(index_name)

# Setup LlamaIndex
parser = SimpleNodeParser.from_defaults()
nodes = parser.get_nodes_from_documents(documents)
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-large")
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
llm = OpenAI(model="gpt-4o-mini")
query_engine = index.as_query_engine(llm=llm, response_mode="compact")
Settings.embed_model = embed_model



def build_document(data, record_id=None):
    for col in ['total_invoice', 'total_invoice_amount', 'total_quantity', 'gross_weight_kg', 'freight_amount', 'net_weight_kg']:
        data[col] = clean_number(data.get(col))

    for col in text_cols:
        val = data.get(col, "")
        data[col] = str(val).strip()

    record_id = record_id or str(uuid4())
    doc_text = make_doc(pd.Series(data))
    lang = detect(doc_text)

    return Document(text=doc_text, id_=record_id, metadata={**data, "lang": lang})

@app.route("/insert", methods=["POST"])
def insert_data():
    try:
        data = request.get_json()
        doc = build_document(data)
        index.insert(doc)
        return jsonify({"id": doc.id_, "message": "Document inserted successfully"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/update/<doc_id>", methods=["PUT"])
def update_data(doc_id):
    try:
        data = request.get_json()
        doc = build_document(data, record_id=doc_id)
        index.update_ref_doc(doc)
        return jsonify({"id": doc_id, "message": "Document updated successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/delete/<doc_id>", methods=["DELETE"])
def delete_data(doc_id):
    try:
        index.delete_ref_doc(doc_id, delete_from_docstore=True)
        return jsonify({"id": doc_id, "message": "Document deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/query', methods=['POST'])
def query_data():
    data = request.get_json()
    user_query = data.get("query", "")
    if not user_query:
        return jsonify({"error": "Query not provided"}), 400
    try:
        response = query_engine.query(user_query)
        return jsonify({
            "markdown_response": f"```\n{response.response.strip()}\n```"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
