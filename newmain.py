from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import requests
import json
from io import BytesIO
from pdfminer.high_level import extract_text
import pandas as pd
import re
from typing import List, Dict, Any
import logging
import traceback
import numpy as np

# Hierarchical Retrieval imports
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Starting application...")

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", include_in_schema=False)
async def serve_ui():
    return FileResponse("static/index.html")

# Groq credentials
GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Global storage for SOP and dataset
current_sop_text = ""
current_sop_steps = []
current_sop_domain = ""
current_dataset = None
current_dataset_domain = ""
current_violations = []
domain_compatibility_score = 0.0

# Hierarchical Retrieval System
embedder = SentenceTransformer('all-MiniLM-L6-v2')
sop_chunks: List[str] = []
sop_embeddings: np.ndarray = None
cluster_model: KMeans = None
cluster_assignments: np.ndarray = None
NUM_CLUSTERS = 5

# Single RAG system for dataset
current_dataset_embeddings: np.ndarray = None
nn_index: NearestNeighbors = None

def build_hierarchical_tree(text: str):
    """Build hierarchical retrieval tree from SOP text"""
    global sop_chunks, sop_embeddings, cluster_model, cluster_assignments
    
    if not text.strip():
        return
    
    try:
        # Create overlapping chunks for better context
        sop_chunks = []
        chunk_size = 500
        overlap = 100
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                sop_chunks.append(chunk.strip())
            if i + chunk_size >= len(text):
                break
        
        if not sop_chunks:
            return
            
        # Generate embeddings
        sop_embeddings = embedder.encode(sop_chunks, convert_to_numpy=True)
        
        # Cluster the chunks
        k = min(NUM_CLUSTERS, len(sop_chunks))
        if k > 1:
            cluster_model = KMeans(n_clusters=k, random_state=42)
            cluster_assignments = cluster_model.fit_predict(sop_embeddings)
        else:
            cluster_model = None
            cluster_assignments = np.zeros(len(sop_chunks))
            
        print(f"✅ Built hierarchical tree with {len(sop_chunks)} chunks and {k} clusters")
        
    except Exception as e:
        print(f"❌ Error building hierarchical tree: {e}")
        sop_chunks = []

def get_relevant_rows(query: str, k: int = 5) -> pd.DataFrame:
    """
    Embed the query, run a nearest-neighbors search on the precomputed index,
    and return the top-k most relevant rows from current_dataset.
    """
    global nn_index, current_dataset_embeddings, current_dataset
    
    if nn_index is None or current_dataset is None:
        return pd.DataFrame()  # no data yet
    
    try:
        # 1. Embed query - ensure 2D array
        q_emb = embedder.encode([query], convert_to_numpy=True)  # shape = (1, D)
        
        # 2. Find nearest neighbors
        dists, idxs = nn_index.kneighbors(q_emb, n_neighbors=min(k, len(current_dataset)))
        
        # 3. Return those rows
        return current_dataset.iloc[idxs[0]]
    except Exception as e:
        logger.error(f"Error retrieving relevant rows: {str(e)}")
        return pd.DataFrame()

def get_relevant_chunks(query: str, top_k: int = 3) -> List[str]:
    """Retrieve most relevant chunks for a query"""
    if not sop_chunks or sop_embeddings is None:
        return []
    
    try:
        # Encode query
        q_emb = embedder.encode([query], convert_to_numpy=True)[0]
        
        if cluster_model is not None:
            # Find best cluster
            centers = cluster_model.cluster_centers_
            cluster_distances = np.linalg.norm(centers - q_emb, axis=1)
            best_cluster = int(np.argmin(cluster_distances))
            
            # Get chunks from best cluster
            cluster_indices = [i for i, c in enumerate(cluster_assignments) if c == best_cluster]
            
            if cluster_indices:
                # Calculate distances within cluster
                cluster_embeddings = sop_embeddings[cluster_indices]
                distances = np.linalg.norm(cluster_embeddings - q_emb, axis=1)
                best_in_cluster = np.argsort(distances)[:top_k]
                return [sop_chunks[cluster_indices[i]] for i in best_in_cluster]
        
        # Fallback: simple similarity search
        distances = np.linalg.norm(sop_embeddings - q_emb, axis=1)
        best_indices = np.argsort(distances)[:top_k]
        return [sop_chunks[i] for i in best_indices]
        
    except Exception as e:
        print(f"❌ Error retrieving chunks: {e}")
        return sop_chunks[:top_k] if sop_chunks else []

def is_aggregate_query(q: str) -> bool:
    """Check if query is asking for aggregate/statistical information"""
    return any(tok in q.lower() for tok in ("count", "sum", "average", "mean", "max", "min", "total", "how many"))

def handle_aggregate_query(question: str) -> str:
    """Handle simple aggregate queries directly with pandas"""
    if current_dataset is None:
        return "No dataset is currently loaded."
    
    question_lower = question.lower()
    
    # Count queries
    if "count" in question_lower or "how many" in question_lower:
        if "row" in question_lower or "record" in question_lower:
            return f"Total rows in dataset: {len(current_dataset)}"
        elif "column" in question_lower:
            return f"Total columns in dataset: {len(current_dataset.columns)}"
    
    # Basic stats
    if "total" in question_lower and "row" in question_lower:
        return f"Total rows: {len(current_dataset)}"
    
    return None  # Not an aggregate query we can handle

def identify_domain(text: str, is_sop: bool = True) -> str:
    """Identify the domain/industry of SOP or dataset using AI"""
    
    if is_sop:
        prompt = f"""
Analyze this SOP document and identify its primary domain/industry. 

SOP TEXT (first 2000 characters):
{text[:2000]}

Based on the content, terminology, processes, and requirements mentioned, identify the PRIMARY domain from these categories:
- Healthcare/Medical
- Manufacturing/Production
- Finance/Banking
- Sales/Marketing
- HR/Human Resources  
- IT/Technology
- Quality Assurance/Testing
- Research/Laboratory
- Legal/Compliance
- Supply Chain/Logistics
- Education/Training
- Food & Safety
- Construction/Engineering
- Retail/E-commerce
- Other

Return only the domain name (e.g., "Healthcare" or "Manufacturing"). Be specific and choose the most accurate match.
"""
    else:
        # For dataset, analyze column names and sample data
        columns = text if isinstance(text, list) else []
        prompt = f"""
Analyze this dataset and identify its primary domain/industry based on the column names.

DATASET COLUMNS:
{', '.join(columns)}

Based on the column names, data types, and fields present, identify the PRIMARY domain from these categories:
- Healthcare/Medical
- Manufacturing/Production  
- Finance/Banking
- Sales/Marketing
- HR/Human Resources
- IT/Technology
- Quality Assurance/Testing
- Research/Laboratory
- Legal/Compliance
- Supply Chain/Logistics
- Education/Training
- Food & Safety
- Construction/Engineering
- Retail/E-commerce
- Other

Return only the domain name (e.g., "Healthcare" or "Sales"). Be specific and choose the most accurate match.
"""

    messages = [
        {"role": "system", "content": "You are a domain classification expert. Analyze the provided content and return only the primary domain/industry name."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        domain = call_groq_api(messages, temperature=0.1)
        # Clean up the response to get just the domain name
        domain_clean = domain.strip().replace('"', '').replace("'", '')
        # Extract first line if multiple lines returned
        domain_clean = domain_clean.split('\n')[0].strip()
        return domain_clean
    except Exception as e:
        print(f"❌ Domain identification error: {e}")
        return "Unknown"

def check_domain_compatibility(sop_domain: str, dataset_domain: str) -> tuple[bool, float, str]:
    """Check if SOP and dataset domains are compatible"""
    
    prompt = f"""
Analyze domain compatibility between an SOP and a dataset.

SOP DOMAIN: {sop_domain}
DATASET DOMAIN: {dataset_domain}

Determine if these domains are compatible for compliance analysis:

1. DIRECT MATCH: Same domain (e.g., Healthcare SOP + Healthcare dataset) = 100% compatible
2. RELATED DOMAINS: Related/overlapping domains (e.g., Quality Assurance + Manufacturing) = 60-80% compatible  
3. INCOMPATIBLE: Completely different domains (e.g., Sales SOP + Healthcare dataset) = 0% compatible

Return a JSON response with:
{{
    "compatible": true/false,
    "compatibility_score": 0-100,
    "explanation": "detailed explanation of why they are/aren't compatible"
}}

Rules:
- If compatibility_score < 50, set compatible = false
- If compatibility_score >= 50, set compatible = true
- Be strict: only allow compatibility when domains actually make sense together
"""

    messages = [
        {"role": "system", "content": "You are a domain compatibility expert. Return only valid JSON with compatibility analysis."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = call_groq_api(messages, temperature=0.1)
        
        # Try to extract JSON from response
        json_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", response)
        if not json_match:
            json_match = re.search(r'\{(?:[^{}]|\{[^{}]*\})*\}', response)
            
        if json_match:
            try:
                raw = json_match.group(1) if json_match.lastindex else json_match.group(0)
                compatibility_data = json.loads(raw)
                
                compatible = compatibility_data.get("compatible", False)
                score = float(compatibility_data.get("compatibility_score", 0))
                explanation = compatibility_data.get("explanation", "No explanation provided")
                
                return compatible, score, explanation
            except (json.JSONDecodeError, ValueError) as e:
                print(f"JSON parsing error in compatibility check: {e}")
        
        # Fallback: simple string matching
        sop_lower = sop_domain.lower()
        dataset_lower = dataset_domain.lower()
        
        if sop_lower == dataset_lower:
            return True, 100.0, "Exact domain match"
        elif any(word in dataset_lower for word in sop_lower.split()) or any(word in sop_lower for word in dataset_lower.split()):
            return True, 75.0, "Related domains with overlapping keywords"
        else:
            return False, 0.0, f"Incompatible domains: {sop_domain} SOP cannot be applied to {dataset_domain} dataset"
            
    except Exception as e:
        print(f"❌ Domain compatibility check error: {e}")
        return False, 0.0, "Error checking domain compatibility"

class ViolationResponse(BaseModel):
    violations: List[Dict[str, Any]]
    summary: Dict[str, Any]

class ViolationDetailRequest(BaseModel):
    violation_index: int

def call_groq_api(messages: List[Dict], temperature: float = 0.3) -> str:
    """Helper function to call Groq API"""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": GROQ_MODEL,
        "temperature": temperature,
        "messages": messages
    }
    
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"❌ Groq API error: {e}")
        return ""

@app.post("/upload-sop/")
async def upload_sop(file: UploadFile = File(...)):
    """Upload and process SOP document with domain identification"""
    global current_sop_text, current_sop_steps, current_sop_domain
    
    try:
        pdf_bytes = await file.read()
        current_sop_text = extract_text(BytesIO(pdf_bytes))
        print(f"📄 Extracted SOP text length: {len(current_sop_text)}")

        # Identify SOP domain
        current_sop_domain = identify_domain(current_sop_text, is_sop=True)
        print(f"🎯 Identified SOP domain: {current_sop_domain}")

        # Build hierarchical retrieval tree
        build_hierarchical_tree(current_sop_text)

        prompt = f"""
Extract all actionable procedural steps, rules, and requirements from this SOP document.
Return a JSON object with the following structure:
{{
    "steps": ["step 1", "step 2", ...],
    "rules": ["rule 1", "rule 2", ...],
    "requirements": ["requirement 1", "requirement 2", ...]
}}

Focus on:
- Specific procedures that must be followed
- Compliance requirements
- Quality standards
- Safety protocols
- Data validation rules
- Approval processes

--- START OF DOCUMENT ---
{current_sop_text[:5000]}  
--- END OF DOCUMENT (truncated if longer) ---
"""

        messages = [
            {"role": "system", "content": "You are an SOP analyzer that extracts procedural steps, rules, and requirements as structured JSON."},
            {"role": "user", "content": prompt}
        ]
        
        content = call_groq_api(messages)
        
        # Try to extract JSON from response
        json_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", content)
        if not json_match:
            json_match = re.search(r'\{(?:[^{}]|\{[^{}]*\})*\}', content)
        
        if json_match:
            try:
                raw = json_match.group(1) if json_match.lastindex else json_match.group(0)
                sop_data = json.loads(raw)
                current_sop_steps = (
                    sop_data.get("steps", []) +
                    sop_data.get("rules", []) +
                    sop_data.get("requirements", [])
                )
                return {
                    "success": True,
                    "sop_data": sop_data,
                    "total_items": len(current_sop_steps),
                    "domain": current_sop_domain
                }
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
        
        # Fallback: extract any list-like content
        lines = content.split('\n')
        extracted_items = []
        for line in lines:
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or re.match(r'^\d+\.', line)):
                clean_line = re.sub(r'^[-•\d\.\s]+', '', line).strip()
                if clean_line:
                    extracted_items.append(clean_line)
        
        if not extracted_items:
            # Final fallback: simple sentence splitting
            sentences = [s.strip() for s in current_sop_text.split('.') if len(s.strip()) > 20]
            extracted_items = sentences[:20]  # Limit to first 20 sentences
        
        current_sop_steps = extracted_items
        return {
            "success": True,
            "sop_data": {"steps": extracted_items},
            "total_items": len(extracted_items),
            "domain": current_sop_domain
        }
        
    except Exception as e:
        print(f"❌ SOP upload error: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.post("/upload-dataset/")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload and process dataset with domain identification and build single RAG index"""
    global current_dataset, current_dataset_domain, current_dataset_embeddings, nn_index
    
    try:
        file_content = await file.read()
        
        # Determine file type and read accordingly
        if file.filename.endswith('.csv'):
            current_dataset = pd.read_csv(BytesIO(file_content))
        elif file.filename.endswith(('.xlsx', '.xls')):
            current_dataset = pd.read_excel(BytesIO(file_content))
        else:
            return {"success": False, "error": "Unsupported file format. Please use CSV or Excel files."}
        
        # Identify dataset domain based on column names
        current_dataset_domain = identify_domain(current_dataset.columns.tolist(), is_sop=False)
        print(f"🎯 Identified dataset domain: {current_dataset_domain}")
        
        # Build single RAG index for dataset
        if len(current_dataset) > 0:
            # Convert each row to a single string
            row_texts = current_dataset.astype(str).agg(' '.join, axis=1).tolist()
            
            # Compute all embeddings once
            current_dataset_embeddings = embedder.encode(row_texts, convert_to_numpy=True)
            
            # Build a fast nearest-neighbors index
            nn_index = NearestNeighbors(n_neighbors=min(10, len(current_dataset)), metric='cosine')
            nn_index.fit(current_dataset_embeddings)
            
            print(f"✅ Built RAG index for {len(current_dataset)} rows")
        else:
            current_dataset_embeddings = None
            nn_index = None
            print("⚠️ Dataset is empty; skipping embedding/indexing.")

        # Basic dataset info
        dataset_info = {
            "rows": len(current_dataset),
            "columns": len(current_dataset.columns),
            "column_names": current_dataset.columns.tolist(),
            "sample_data": current_dataset.head(3).to_dict('records'),
            "domain": current_dataset_domain
        }
        
        return {
            "success": True,
            "dataset_info": dataset_info
        }
        
    except Exception as e:
        print(f"❌ Dataset upload error: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}
    
@app.post("/analyze-violations/")
async def analyze_violations():
    """Analyze dataset for SOP violations with domain compatibility check"""
    global current_sop_steps, current_dataset, current_sop_text, current_violations
    global current_sop_domain, current_dataset_domain, domain_compatibility_score
    
    if not current_sop_text:
        return {"success": False, "error": "No SOP uploaded. Please upload an SOP first."}
    
    if current_dataset is None:
        return {"success": False, "error": "No dataset uploaded. Please upload a dataset first."}
    
    try:
        # Check domain compatibility first
        is_compatible, compatibility_score, compatibility_explanation = check_domain_compatibility(
            current_sop_domain, current_dataset_domain
        )
        
        domain_compatibility_score = compatibility_score
        
        print(f"🔍 Domain Compatibility Check:")
        print(f"   SOP Domain: {current_sop_domain}")
        print(f"   Dataset Domain: {current_dataset_domain}")
        print(f"   Compatible: {is_compatible}")
        print(f"   Score: {compatibility_score}%")
        print(f"   Explanation: {compatibility_explanation}")
        
        # If domains are incompatible, return incompatibility status
        if not is_compatible:
            incompatible_analysis = {
                "violations": [],
                "summary": {
                    "total_violations": 0,
                    "critical": 0,
                    "high": 0,
                    "medium": 0,
                    "low": 0,
                    "compliant_rows": 0,
                    "compliance_rate": -1.0,  # -1 indicates incompatibility
                    "domain_compatibility": {
                        "compatible": False,
                        "score": compatibility_score,
                        "sop_domain": current_sop_domain,
                        "dataset_domain": current_dataset_domain,
                        "explanation": compatibility_explanation
                    }
                },
                "domain_mismatch": True,
                "domain_explanation": compatibility_explanation,
                "status": "INCOMPATIBLE_DOMAINS"
            }
            
            current_violations = []
            
            return {
                "success": True,
                "analysis": incompatible_analysis
            }
        
        # Proceed with normal violation analysis for compatible domains
        dataset_sample = current_dataset.head(20).to_string()
        dataset_columns = current_dataset.columns.tolist()
        
        prompt = f"""
You are a compliance analyst. Analyze the provided dataset against the SOP requirements to identify violations.

DOMAIN CONTEXT:
- SOP Domain: {current_sop_domain}
- Dataset Domain: {current_dataset_domain}
- Compatibility Score: {compatibility_score}%

SOP REQUIREMENTS:
{chr(10).join([f"- {step}" for step in current_sop_steps[:20]])}

DATASET COLUMNS:
{', '.join(dataset_columns)}

DATASET SAMPLE (first 20 rows):
{dataset_sample}

ANALYSIS INSTRUCTIONS:
1. Check each row against SOP requirements
2. Identify specific violations (missing data, invalid values, non-compliance)
3. Categorize violations by severity (Critical, High, Medium, Low)
4. Provide row numbers where violations occur

Return a JSON object with this structure:
{{
    "violations": [
        {{
            "row_number": 1,
            "column": "column_name",
            "violation_type": "missing_required_field",
            "severity": "Critical",
            "sop_requirement": "specific requirement violated",
            "current_value": "actual value in dataset",
            "expected_value": "what should be there",
            "description": "detailed explanation",
            "rule_violated": "specific rule name",
            "field_name": "field that has issue",
            "remediation_suggestion": "how to fix this violation"
        }}
    ],
    "summary": {{
        "total_violations": 10,
        "critical": 2,
        "high": 3,
        "medium": 4,
        "low": 1,
        "compliant_rows": 15,
        "compliance_rate": 60.0
    }}
}}
"""

        messages = [
            {"role": "system", "content": "You are a compliance analyst that identifies SOP violations in datasets. Always return valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        content = call_groq_api(messages, temperature=0.1)
        
        # Try to extract JSON from response
        json_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", content)
        if not json_match:
            json_match = re.search(r'\{(?:[^{}]|\{[^{}]*\})*\}', content)
            
        if json_match:
            try:
                raw = json_match.group(1) if json_match.lastindex else json_match.group(0)
                violation_data = json.loads(raw)
                current_violations = violation_data.get("violations", [])
                
                # Add domain compatibility info to summary
                if "summary" in violation_data:
                    violation_data["summary"]["domain_compatibility"] = {
                        "compatible": is_compatible,
                        "score": compatibility_score,
                        "sop_domain": current_sop_domain,
                        "dataset_domain": current_dataset_domain,
                        "explanation": compatibility_explanation
                    }
                    # Add status based on violations found
                    if len(current_violations) == 0:
                        violation_data["status"] = "FULLY_COMPLIANT"
                    else:
                        violation_data["status"] = "VIOLATIONS_FOUND"
                
                return {
                    "success": True,
                    "analysis": violation_data
                }
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
        
        # Fallback: create basic analysis
        total_rows = len(current_dataset)
        missing_data_count = current_dataset.isnull().sum().sum()
        
        fallback_violations = []
        if missing_data_count > 0:
            fallback_violations.append({
                "row_number": "Multiple",
                "column": "Various",
                "violation_type": "missing_data",
                "severity": "Medium",
                "sop_requirement": "Data completeness",
                "current_value": "NULL/Empty",
                "expected_value": "Valid data",
                "description": f"Found {missing_data_count} missing values across dataset",
                "rule_violated": "Data Completeness Rule",
                "field_name": "Multiple fields",
                "remediation_suggestion": "Fill missing values with appropriate data or mark as N/A"
            })
        
        fallback_analysis = {
            "violations": fallback_violations,
            "summary": {
                "total_violations": len(fallback_violations),
                "critical": 0,
                "high": 0,
                "medium": len(fallback_violations),
                "low": 0,
                "compliant_rows": max(0, total_rows - len(fallback_violations)),
                "compliance_rate": max(0, (total_rows - len(fallback_violations)) / total_rows * 100) if total_rows > 0 else 100,
                "domain_compatibility": {
                    "compatible": is_compatible,
                    "score": compatibility_score,
                    "sop_domain": current_sop_domain,
                    "dataset_domain": current_dataset_domain,
                    "explanation": compatibility_explanation
                }
            },
            "status": "FULLY_COMPLIANT" if len(fallback_violations) == 0 else "VIOLATIONS_FOUND"
        }
        
        current_violations = fallback_violations
        
        return {
            "success": True,
            "analysis": fallback_analysis
        }
        
    except Exception as e:
        print(f"❌ Violation analysis error: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.post("/violation-details/")
async def get_violation_details(request: ViolationDetailRequest):
    """Get detailed explanation of a specific violation using Groq API"""
    global current_violations, current_sop_steps, current_dataset
    
    try:
        if request.violation_index >= len(current_violations):
            return {"success": False, "error": "Invalid violation index"}
        
        violation = current_violations[request.violation_index]
        
        # Get relevant dataset context if available
        dataset_context = ""
        if current_dataset is not None:
            row_num = violation.get("row_number", "N/A")
            if row_num != "N/A" and row_num != "Multiple" and str(row_num).isdigit():
                try:
                    row_idx = int(row_num) - 1
                    if 0 <= row_idx < len(current_dataset):
                        row_data = current_dataset.iloc[row_idx].to_dict()
                        dataset_context = f"\nActual row data: {row_data}"
                except (ValueError, IndexError):
                    pass
        
        prompt = f"""
Provide a detailed technical explanation of this SOP compliance violation:

VIOLATION SUMMARY:
- Rule Violated: {violation.get('rule_violated', 'N/A')}
- Severity: {violation.get('severity', 'N/A')}
- Description: {violation.get('description', 'N/A')}
- Field: {violation.get('field_name', 'N/A')}
- Column: {violation.get('column', 'N/A')}
- Row: {violation.get('row_number', 'N/A')}
- Current Value: {violation.get('current_value', 'N/A')}
- Expected Value: {violation.get('expected_value', 'N/A')}
- SOP Requirement: {violation.get('sop_requirement', 'N/A')}
{dataset_context}

RELEVANT SOP REQUIREMENTS:
{chr(10).join([f"- {step}" for step in current_sop_steps[:10]])}

Please provide:
1. **Root Cause Analysis**: Why this violation occurred
2. **Business Impact**: What problems this could cause
3. **Technical Details**: Specific technical issues
4. **Step-by-Step Remediation**: How to fix this violation
5. **Prevention Measures**: How to prevent similar violations

Format your response clearly with headers and bullet points.
"""

        messages = [
            {"role": "system", "content": "You are a compliance expert providing detailed technical analysis of SOP violations. Be thorough and practical."},
            {"role": "user", "content": prompt}
        ]
        
        detailed_explanation = call_groq_api(messages, temperature=0.2)
        
        if not detailed_explanation:
            detailed_explanation = "Unable to generate detailed explanation at this time. Please try again."
        
        return {
            "success": True,
            "detailed_explanation": detailed_explanation,
            "violation_summary": violation
        }
        
    except Exception as e:
        print(f"❌ Violation details error: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.get("/status/")
async def get_status():
    """Get current status of uploaded files with domain information"""
    return {
        "sop_uploaded": len(current_sop_steps) > 0,
        "dataset_uploaded": current_dataset is not None,
        "sop_items_count": len(current_sop_steps),
        "dataset_rows": len(current_dataset) if current_dataset is not None else 0,
        "dataset_columns": current_dataset.columns.tolist() if current_dataset is not None else [],
        "hierarchical_chunks": len(sop_chunks),
        "sop_domain": current_sop_domain,
        "dataset_domain": current_dataset_domain,
        "domain_compatibility_score": domain_compatibility_score
    }

class QARequest(BaseModel):
    question: str
    context_type: str = "sop"

@app.post("/ask/")
def ask_question(request: QARequest):
    """Enhanced Q&A that can answer questions about SOP, dataset, or violations"""
    
    context = ""
    if request.context_type in ["sop", "both"]:
        context += f"SOP REQUIREMENTS ({current_sop_domain} domain):\n{chr(10).join(current_sop_steps[:10])}\n\n"
    
    if request.context_type in ["violations", "both"] and current_dataset is not None:
        context += f"DATASET INFO ({current_dataset_domain} domain):\nRows: {len(current_dataset)}, Columns: {current_dataset.columns.tolist()}\n"
        context += f"Sample data:\n{current_dataset.head(5).to_string()}\n\n"
    
    prompt = f"""Answer the user's question based on the following context:

{context}

Question: {request.question}

Provide a clear, concise, and helpful answer based on the available information."""

    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions about SOPs, datasets, and compliance violations."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        content = call_groq_api(messages)
        return {"answer": content}
    except Exception as e:
        print(f"❌ Q&A error: {e}")
        return {"answer": "Sorry, I couldn't process your question at the moment."}

from sentence_transformers import SentenceTransformer

def get_topk_relevant_dataset_rows(query, k=5):
    """Returns the k most relevant dataset rows (as text) to the query."""
    if DATASET_ROW_EMBEDDINGS is None or DATASET_NN_INDEX is None:
        return []
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        q_emb = model.encode([query])[0].reshape(1, -1)
        dists, idxs = DATASET_NN_INDEX.kneighbors(q_emb, n_neighbors=min(k, len(DATASET_ROW_TEXTS)))
        return [DATASET_ROW_TEXTS[i] for i in idxs[0]]
    except Exception as e:
        logger.error(f"Error retrieving top-k dataset rows: {str(e)}")
        return []
    
class ChatRequest(BaseModel):
    question: str
@app.post("/chat/")
async def chat_endpoint(request: ChatRequest):
    """
    Enhanced chat endpoint with hierarchical retrieval for SOP compliance
    """
    try:
        logger.info(f"Received chat request: {request.question}")
        
        question = request.question
        
        # 0. If it's an aggregate/statistical question, run pandas and return
        if is_aggregate_query(question) and current_dataset is not None:
            aggregate_answer = handle_aggregate_query(question)
            if aggregate_answer:
                return {"response": aggregate_answer}
        
        # Get relevant SOP chunks using hierarchical retrieval
        try:
            relevant_chunks = get_relevant_chunks(question, top_k=3)
            logger.info(f"Retrieved {len(relevant_chunks) if relevant_chunks else 0} relevant chunks")
        except Exception as e:
            logger.error(f"Error getting relevant chunks: {str(e)}")
            relevant_chunks = []
        
        # Build comprehensive context
        context = "You are a SOP Compliance Assistant with access to relevant SOP sections."
        
        # Add domain compatibility information
        try:
            context += f"\n\nDOMAIN ANALYSIS:"
            context += f"\n- SOP Domain: {current_sop_domain if current_sop_domain else 'Not available'}"
            context += f"\n- Dataset Domain: {current_dataset_domain if current_dataset_domain else 'Not available'}"
            context += f"\n- Compatibility Score: {domain_compatibility_score:.2f if domain_compatibility_score else 'Not calculated'}"
            
            if domain_compatibility_score and domain_compatibility_score < 0.3:
                context += f"\n- WARNING: Low domain compatibility detected!"
        except Exception as e:
            logger.error(f"Error building domain analysis: {str(e)}")
            context += f"\n\nDOMAIN ANALYSIS: Not available"
        
        if relevant_chunks:
            context += f"\n\nRELEVANT SOP SECTIONS:\n"
            for i, chunk in enumerate(relevant_chunks, 1):
                context += f"\n--- Section {i} ---\n{chunk}\n"
        
        try:
            if current_sop_steps:
                context += f"\n\nKEY SOP REQUIREMENTS ({len(current_sop_steps)} total):\n"
                context += "\n".join(f"{i+1}. {step}" for i, step in enumerate(current_sop_steps[:5]))
                if len(current_sop_steps) > 5:
                    context += f"\n... and {len(current_sop_steps) - 5} more requirements."
        except Exception as e:
            logger.error(f"Error adding SOP steps: {str(e)}")
        
        try:
            if current_dataset is not None:
                context += f"\n\nCURRENT DATASET INFO:\n"
                context += f"- Rows: {len(current_dataset)}\n"
                context += f"- Columns: {list(current_dataset.columns)}\n"
                context += f"- Domain: {current_dataset_domain if current_dataset_domain else 'Not available'}\n"
                context += f"- Domain Compatibility: {domain_compatibility_score:.2f if domain_compatibility_score else 'Not calculated'}"
        except Exception as e:
            logger.error(f"Error adding dataset info: {str(e)}")
        
        # --- NEW BLOCK: retrieve only the top-5 most relevant rows ---
        try:
            relevant_rows = get_relevant_rows(question, k=5)
            if not relevant_rows.empty:
                rows_str = relevant_rows.to_string(index=False)
                context += f"\n\nRELEVANT DATASET ROWS (top 5 most similar to your question):\n{rows_str}\n"
                logger.info(f"Retrieved {len(relevant_rows)} relevant dataset rows")
        except Exception as e:
            logger.error(f"Error retrieving relevant dataset rows: {str(e)}")
        # -------------------------------------------------------------
        
        try:
            if current_violations:
                context += f"\n\nCURRENT VIOLATIONS FOUND ({len(current_violations)} total):\n"
                for i, violation in enumerate(current_violations[:3]):
                    context += f"- Row {violation.get('row_number', 'N/A')}: {violation.get('description', 'No description')}\n"
                if len(current_violations) > 3:
                    context += f"... and {len(current_violations) - 3} more violations.\n"
        except Exception as e:
            logger.error(f"Error adding violations info: {str(e)}")
        
        context += "\n\nPlease provide helpful, specific answers about SOP compliance, violations, and remediation suggestions based on the relevant sections provided. Consider domain compatibility in your responses."
        
        messages = [
            {"role": "system", "content": context},
            {"role": "user", "content": question}
        ]
        
        logger.info("Calling Groq API...")
        try:
            answer = call_groq_api(messages)
            logger.info(f"Groq API response received: {answer[:100]}..." if len(answer) > 100 else f"Groq API response: {answer}")
            return {"response": answer}
        except Exception as e:
            logger.error(f"❌ Groq API error: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"response": "Sorry, I'm having trouble connecting to the AI service right now. Please try again."}
            
    except Exception as e:
        logger.error(f"❌ Chat endpoint error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"response": "Sorry, I'm having trouble processing your question right now. Please try again."}
    

@app.post("/reset/")
async def reset_system():
    """Reset all uploaded data and analysis results"""
    global current_sop_text, current_sop_steps, current_dataset, current_violations
    global current_sop_domain, current_dataset_domain, domain_compatibility_score
    global sop_chunks, sop_embeddings, cluster_model, cluster_assignments
    global DATASET_ROW_EMBEDDINGS, DATASET_ROW_TEXTS, DATASET_NN_INDEX
    global current_dataset_embeddings, nn_index  # Add these globals
    
    try:
        # Reset SOP data
        current_sop_text = ""
        current_sop_steps = []
        current_sop_domain = ""
        
        # Reset dataset data
        current_dataset = None
        current_dataset_domain = ""
        
        # Reset analysis results
        current_violations = []
        domain_compatibility_score = 0.0
        
        # Reset hierarchical retrieval
        sop_chunks = []
        sop_embeddings = None
        cluster_model = None
        cluster_assignments = None
        
        # Reset existing embeddings
        DATASET_ROW_EMBEDDINGS = None
        DATASET_ROW_TEXTS = None
        DATASET_NN_INDEX = None
        
        # Reset new RAG embeddings & index
        current_dataset_embeddings = None
        nn_index = None
        
        return {
            "success": True,
            "message": "System reset successfully. All data cleared."
        }
        
    except Exception as e:
        print(f"❌ Reset error: {e}")
        return {"success": False, "error": str(e)}
    
@app.get("/domain-info/")
async def get_domain_info():
    """Get detailed domain analysis information"""
    return {
        "sop_domain": sop_domain,
        "dataset_domain": dataset_domain,
        "compatibility_score": domain_compatibility_score,
        "compatibility_status": (
            "High" if domain_compatibility_score >= 0.7 else
            "Medium" if domain_compatibility_score >= 0.3 else
            "Low"
        ),
        "is_compatible": domain_compatibility_score >= 0.3,
        "recommendation": (
            "Domains are highly compatible - proceed with analysis" if domain_compatibility_score >= 0.7 else
            "Domains are moderately compatible - analysis may have limitations" if domain_compatibility_score >= 0.3 else
            "Domains are incompatible - consider using matching SOP and dataset"
        )
    }

@app.post("/suggest-improvements/")
async def suggest_improvements():
    """Generate suggestions for improving compliance based on violations"""
    global current_violations, current_sop_steps, current_dataset
    
    if not current_violations:
        return {"success": False, "error": "No violations found. Run analysis first."}
    
    try:
        # Analyze violation patterns
        violation_types = {}
        severity_counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
        
        for violation in current_violations:
            v_type = violation.get("violation_type", "unknown")
            severity = violation.get("severity", "Unknown")
            
            violation_types[v_type] = violation_types.get(v_type, 0) + 1
            if severity in severity_counts:
                severity_counts[severity] += 1
        
        # Generate improvement suggestions
        improvement_prompt = f"""
Based on the following compliance violations analysis, provide actionable improvement recommendations:

VIOLATION SUMMARY:
- Total Violations: {len(current_violations)}
- Critical: {severity_counts['Critical']}
- High: {severity_counts['High']}
- Medium: {severity_counts['Medium']}
- Low: {severity_counts['Low']}

VIOLATION TYPES:
{chr(10).join([f"- {v_type}: {count} occurrences" for v_type, count in violation_types.items()])}

DOMAIN COMPATIBILITY: {domain_compatibility_score:.2f} (SOP: {sop_domain}, Dataset: {dataset_domain})

TOP VIOLATIONS:
{chr(10).join([f"- {v.get('description', 'No description')}" for v in current_violations[:5]])}

Please provide:
1. **Priority Actions**: Most critical fixes needed immediately
2. **Process Improvements**: Changes to prevent future violations  
3. **Training Recommendations**: Areas where staff need better guidance
4. **System/Tool Suggestions**: Technical solutions to automate compliance
5. **Monitoring Strategy**: How to track compliance improvements over time

Focus on practical, actionable recommendations that address the root causes.
"""

        messages = [
            {"role": "system", "content": "You are a compliance improvement specialist. Provide specific, actionable recommendations to improve SOP compliance based on violation patterns. Consider domain compatibility in your suggestions."},
            {"role": "user", "content": improvement_prompt}
        ]
        
        suggestions = call_groq_api(messages, temperature=0.2)
        
        return {
            "success": True,
            "improvement_suggestions": suggestions,
            "violation_summary": {
                "total_violations": len(current_violations),
                "severity_breakdown": severity_counts,
                "violation_types": violation_types,
                "domain_compatibility": domain_compatibility_score
            }
        }
        
    except Exception as e:
        print(f"❌ Improvement suggestions error: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.post("/export-report/")
async def export_compliance_report():
    """Generate a comprehensive compliance report"""
    global current_violations, current_sop_steps, current_dataset
    global sop_domain, dataset_domain, domain_compatibility_score
    
    if not current_violations and current_dataset is not None:
        return {"success": False, "error": "No analysis results available. Run violation analysis first."}
    
    try:
        # Calculate summary statistics
        total_rows = len(current_dataset) if current_dataset is not None else 0
        violation_count = len(current_violations)
        
        severity_counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
        for violation in current_violations:
            severity = violation.get("severity", "Unknown")
            if severity in severity_counts:
                severity_counts[severity] += 1
        
        # Calculate compliance rate considering domain compatibility
        base_compliance_rate = max(0, (total_rows - violation_count) / total_rows * 100) if total_rows > 0 else 100
        adjusted_compliance_rate = base_compliance_rate * domain_compatibility_score if domain_compatibility_score > 0 else 0
        
        # Generate detailed report
        report = {
            "report_metadata": {
                "generated_at": pd.Timestamp.now().isoformat(),
                "sop_domain": sop_domain,
                "dataset_domain": dataset_domain,
                "domain_compatibility_score": domain_compatibility_score,
                "domain_compatibility_status": (
                    "High" if domain_compatibility_score >= 0.7 else
                    "Medium" if domain_compatibility_score >= 0.3 else
                    "Low"
                )
            },
            "executive_summary": {
                "total_records_analyzed": total_rows,
                "total_violations_found": violation_count,
                "raw_compliance_rate": round(base_compliance_rate, 2),
                "adjusted_compliance_rate": round(adjusted_compliance_rate, 2),
                "domain_adjusted": domain_compatibility_score < 1.0,
                "severity_breakdown": severity_counts,
                "overall_status": (
                    "COMPLIANT" if adjusted_compliance_rate >= 95 else
                    "MOSTLY_COMPLIANT" if adjusted_compliance_rate >= 80 else
                    "NON_COMPLIANT" if adjusted_compliance_rate >= 50 else
                    "CRITICAL_NON_COMPLIANCE"
                )
            },
            "sop_analysis": {
                "total_requirements": len(current_sop_steps),
                "domain": sop_domain,
                "requirements_sample": current_sop_steps[:5] if current_sop_steps else []
            },
            "dataset_analysis": {
                "total_rows": total_rows,
                "total_columns": len(current_dataset.columns) if current_dataset is not None else 0,
                "columns": current_dataset.columns.tolist() if current_dataset is not None else [],
                "domain": dataset_domain
            },
            "violations_detail": current_violations[:50],  # Limit to first 50 violations
            "recommendations": {
                "immediate_actions": [],
                "long_term_improvements": [],
                "domain_compatibility_note": (
                    "Domain compatibility is low. Consider using an SOP designed for this dataset's domain." 
                    if domain_compatibility_score < 0.3 else
                    "Domain compatibility is acceptable for analysis."
                )
            }
        }
        
        # Add specific recommendations based on violations
        if severity_counts["Critical"] > 0:
            report["recommendations"]["immediate_actions"].append(
                f"Address {severity_counts['Critical']} critical violations immediately"
            )
        
        if domain_compatibility_score < 0.3:
            report["recommendations"]["immediate_actions"].append(
                "Review domain compatibility - current SOP may not be suitable for this dataset"
            )
        
        return {
            "success": True,
            "compliance_report": report
        }
        
    except Exception as e:
        print(f"❌ Report export error: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}

from fastapi import Request

@app.post("/api/llm")
async def api_llm_proxy(request: Request):
    data = await request.json()
    # Try to extract the text/prompt that needs completion.
    # The optimizer might send {"prompt": "..."} or {"question": "..."} or even {"messages": [...]}
    question = (
        data.get("prompt")
        or data.get("question")
        or (
            data.get("messages", [{}])[-1].get("content")
            if data.get("messages")
            else ""
        )
    )
    if not question:
        return {"response": ""}

    print(f"🟡 [PROXY] /api/llm received: {question!r}")

    # Proxy to your existing chat logic (reuse your ChatRequest model/class)
    from pydantic import ValidationError
    try:
        chat_req = ChatRequest(question=question)
        chat_response = await chat_endpoint(chat_req)
    except ValidationError:
        return {"response": ""}

    # Normalize to always return {'response': ...}
    if isinstance(chat_response, dict):
        # If your /chat/ returns {'response': ...}
        return {"response": chat_response.get("response", "")}
    # If your /chat/ returns a Response object, convert as needed
    return {"response": str(chat_response)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)