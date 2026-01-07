"""
NEXUS CDM - Data Platform
=========================
Port 8503

Clinical data ingestion, schema extraction, and transformation pipeline.
"""

import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime
from typing import Dict, List, Any
import sys


from config import DATA_BASE_PATH, FILE_PATTERNS, STUDY_NAMES
from agents.tools.data_tools import get_data_loader

# Page config
st.set_page_config(
    page_title="NEXUS CDM - Data Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS - avoid universal selector to prevent font rendering issues
st.markdown("""
<style>
.stApp { background-color: #1a1a1a; }
h1, h2, h3, p, span, div, label { font-family: 'SF Mono', 'Consolas', monospace; }
h1, h2, h3 { color: #dcdcaa; }
.stMarkdown, .stText, .stCode { font-family: 'SF Mono', 'Consolas', monospace; }
[data-testid="stMetricValue"] { font-family: 'SF Mono', monospace; }
.file-card {
    background-color: #252526;
    padding: 10px;
    border-radius: 5px;
    margin: 5px 0;
    border-left: 3px solid #6a9955;
}
.schema-card {
    background-color: #1e1e1e;
    padding: 15px;
    border-radius: 5px;
    border: 1px solid #404040;
}
</style>
""", unsafe_allow_html=True)


def get_study_folders() -> List[Dict]:
    """Get actual study folders from disk"""
    folders = []
    if os.path.exists(DATA_BASE_PATH):
        for item in os.listdir(DATA_BASE_PATH):
            full_path = os.path.join(DATA_BASE_PATH, item)
            if os.path.isdir(full_path):
                # Count files
                files = [f for f in os.listdir(full_path) if f.endswith('.xlsx')]
                total_size = sum(os.path.getsize(os.path.join(full_path, f)) for f in files)
                folders.append({
                    "name": item,
                    "path": full_path,
                    "file_count": len(files),
                    "total_size_mb": round(total_size / (1024 * 1024), 2)
                })
    return sorted(folders, key=lambda x: x['name'])


def get_study_files(study_path: str) -> List[Dict]:
    """Get actual files from a study folder"""
    files = []
    if os.path.exists(study_path):
        for f in os.listdir(study_path):
            if f.endswith('.xlsx'):
                full_path = os.path.join(study_path, f)
                stat = os.stat(full_path)
                files.append({
                    "name": f,
                    "path": full_path,
                    "size_kb": round(stat.st_size / 1024, 2),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
    return sorted(files, key=lambda x: x['name'])


def parse_excel_schema(file_path: str) -> Dict:
    """Parse Excel file and extract schema"""
    try:
        # Read all sheets
        xl = pd.ExcelFile(file_path)
        sheets = {}

        for sheet_name in xl.sheet_names:
            df = pd.read_excel(xl, sheet_name=sheet_name, nrows=100)  # Sample for schema
            sheets[sheet_name] = {
                "columns": list(df.columns),
                "dtypes": {col: str(df[col].dtype) for col in df.columns},
                "row_count": len(df),
                "null_counts": {col: int(df[col].isnull().sum()) for col in df.columns},
                "sample_values": {col: str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else "N/A"
                                 for col in df.columns}
            }

        return {"success": True, "sheets": sheets}
    except Exception as e:
        return {"success": False, "error": str(e)}


def render_source_browser():
    """Render source file browser section"""
    st.header("1. SOURCE FILE BROWSER")

    st.subheader("Data Source Location")
    st.code(f"BASE_PATH: {DATA_BASE_PATH}", language=None)

    # Get real folders
    folders = get_study_folders()

    if not folders:
        st.error(f"No study folders found at {DATA_BASE_PATH}")
        return

    st.subheader(f"Study Folders ({len(folders)} found)")

    # Summary metrics
    total_files = sum(f['file_count'] for f in folders)
    total_size = sum(f['total_size_mb'] for f in folders)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Studies", len(folders))
    with col2:
        st.metric("Excel Files", total_files)
    with col3:
        st.metric("Total Size", f"{total_size:.2f} MB")

    st.markdown("---")

    # Folder listing
    folder_text = """
STUDY FOLDER            | FILES | SIZE (MB) | PATH
------------------------|-------|-----------|-----"""

    for folder in folders:
        folder_text += f"\n{folder['name']:<23} | {folder['file_count']:<5} | {folder['total_size_mb']:<9} | {folder['path']}"

    st.code(folder_text, language=None)

    # File browser
    st.markdown("---")
    st.subheader("File Explorer")

    selected_study = st.selectbox("Select Study Folder", [f['name'] for f in folders])

    if selected_study:
        selected_folder = next(f for f in folders if f['name'] == selected_study)
        files = get_study_files(selected_folder['path'])

        st.markdown(f"**Files in {selected_study}:**")

        file_text = """
FILENAME                                          | SIZE (KB) | MODIFIED
--------------------------------------------------|-----------|------------------------"""

        for file in files:
            file_text += f"\n{file['name'][:49]:<49} | {file['size_kb']:<9} | {file['modified']}"

        st.code(file_text, language=None)


def render_schema_extraction():
    """Render schema extraction demo"""
    st.header("2. SCHEMA EXTRACTION")

    st.markdown("""
    **Live Excel Parsing** - Select a file to extract its schema in real-time.
    """)

    # Get folders and files
    folders = get_study_folders()
    if not folders:
        st.error("No study folders found")
        return

    col1, col2 = st.columns(2)
    with col1:
        selected_study = st.selectbox("Study", [f['name'] for f in folders], key="schema_study")

    if selected_study:
        selected_folder = next(f for f in folders if f['name'] == selected_study)
        files = get_study_files(selected_folder['path'])

        with col2:
            selected_file = st.selectbox("File", [f['name'] for f in files], key="schema_file")

    if st.button("Extract Schema", type="primary"):
        if selected_study and selected_file:
            selected_folder = next(f for f in folders if f['name'] == selected_study)
            file_path = os.path.join(selected_folder['path'], selected_file)

            # Show parsing code
            st.subheader("Parsing Code")
            st.code(f"""
import pandas as pd

# Actual file path
file_path = "{file_path}"

# Parse Excel file
xl = pd.ExcelFile(file_path)
print(f"Sheets found: {{xl.sheet_names}}")

for sheet_name in xl.sheet_names:
    df = pd.read_excel(xl, sheet_name=sheet_name)
    print(f"Sheet: {{sheet_name}}")
    print(f"Columns: {{list(df.columns)}}")
    print(f"Rows: {{len(df)}}")
            """, language="python")

            # Execute parsing
            with st.spinner("Parsing Excel file..."):
                result = parse_excel_schema(file_path)

            if result['success']:
                st.success(f"Parsed {len(result['sheets'])} sheet(s)")

                for sheet_name, schema in result['sheets'].items():
                    with st.expander(f"Sheet: {sheet_name} ({schema['row_count']} rows)", expanded=True):
                        st.markdown("**Column Schema:**")

                        schema_text = """
COLUMN NAME                  | DTYPE      | NULLS | SAMPLE VALUE
-----------------------------|------------|-------|----------------------------------"""

                        for col in schema['columns']:
                            dtype = schema['dtypes'][col][:10]
                            nulls = schema['null_counts'][col]
                            sample = str(schema['sample_values'][col])[:32]
                            schema_text += f"\n{str(col)[:28]:<28} | {dtype:<10} | {nulls:<5} | {sample}"

                        st.code(schema_text, language=None)
            else:
                st.error(f"Parse error: {result['error']}")


def render_transformation_trace():
    """Render data transformation trace"""
    st.header("3. DATA TRANSFORMATION PIPELINE")

    st.markdown("""
    **ETL Pipeline** - Shows how raw Excel data is transformed into unified study structure.
    """)

    loader = get_data_loader()

    # Pipeline steps
    st.subheader("Transformation Pipeline")

    pipeline_text = """
STAGE 1: FILE DISCOVERY
-----------------------
  Input: DATA_BASE_PATH
  Action: os.listdir() to find study folders
  Output: List of study directories

STAGE 2: EXCEL PARSING (per study)
----------------------------------
  Input: Study folder path
  Action: pd.read_excel() for each file pattern:
    - EDC_Metrics*.xlsx -> patient enrollment
    - Missing_Pages*.xlsx -> CRF status
    - SAE*.xlsx -> safety data
    - MedDRA*.xlsx / WHODD*.xlsx -> coding
  Output: Raw DataFrames per file type

STAGE 3: DATA UNIFICATION
-------------------------
  Input: Raw DataFrames
  Action: Aggregate metrics by site:
    - Count patients per site
    - Sum open queries
    - Sum missing pages
    - Extract query details
  Output: Unified study structure

STAGE 4: QUALITY CALCULATION
----------------------------
  Input: Unified study data
  Action: Calculate DQI per site:
    DQI = 100 - SUM(Wi x Di)
  Output: Site-level quality scores
    """

    st.code(pipeline_text, language=None)

    st.markdown("---")

    # Live transformation
    st.subheader("Live Transformation Demo")

    folders = get_study_folders()
    if not folders:
        st.error("No study folders found")
        return

    selected_study = st.selectbox("Select Study for Transformation", [f['name'] for f in folders], key="transform_study")

    if st.button("Execute Transformation", type="primary"):
        selected_folder = next(f for f in folders if f['name'] == selected_study)

        st.markdown("### Stage 1: File Discovery")
        files = get_study_files(selected_folder['path'])
        st.code(f"""
os.listdir("{selected_folder['path']}")
# Result: {len(files)} Excel files found
        """, language="python")

        for f in files[:5]:
            st.code(f"  - {f['name']} ({f['size_kb']} KB)", language=None)
        if len(files) > 5:
            st.code(f"  ... and {len(files) - 5} more files", language=None)

        st.markdown("### Stage 2: Excel Parsing")

        # Parse first file as demo
        if files:
            first_file = files[0]
            st.code(f"""
df = pd.read_excel("{first_file['path']}")
print(f"Loaded {{len(df)}} rows, {{len(df.columns)}} columns")
            """, language="python")

            try:
                df = pd.read_excel(first_file['path'], nrows=5)
                st.markdown(f"**Sample from {first_file['name']}:**")
                st.dataframe(df.head(), use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

        st.markdown("### Stage 3: Data Unification")

        # Get unified data
        all_studies = loader.get_all_studies()

        # Find matching study
        study_key = None
        for key in all_studies.keys():
            if selected_study.lower() in key.lower() or key.lower() in selected_study.lower():
                study_key = key
                break

        if study_key:
            study_data = all_studies[study_key]

            st.code(f"""
# Unified structure for {study_key}
study_data = {{
    "patients": [...],     # {len(study_data['patients'])} records
    "sites": {{...}},       # {len(study_data['sites'])} sites
    "queries": [...],      # {len(study_data['queries'])} queries
}}
            """, language="python")

            st.markdown("### Stage 4: Quality Metrics")

            # Calculate summary stats
            total_queries = sum(s['open_queries'] for s in study_data['sites'].values())
            total_missing = sum(s['missing_pages'] for s in study_data['sites'].values())

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Patients", len(study_data['patients']))
            with col2:
                st.metric("Sites", len(study_data['sites']))
            with col3:
                st.metric("Open Queries", total_queries)
            with col4:
                st.metric("Missing Pages", total_missing)
        else:
            st.warning(f"Study {selected_study} not found in loaded data")


def render_data_explorer():
    """Render interactive data explorer"""
    st.header("4. DATA EXPLORER")

    st.markdown("""
    **Direct Data Access** - Query loaded study data interactively.
    """)

    loader = get_data_loader()
    all_studies = loader.get_all_studies()

    # Study selector
    selected_study = st.selectbox("Select Study", list(all_studies.keys()), key="explorer_study")

    if selected_study:
        study_data = all_studies[selected_study]

        # Data sections
        tab1, tab2, tab3, tab4 = st.tabs(["Patients", "Sites", "Queries", "Summary"])

        with tab1:
            st.subheader("Patient Data")
            if study_data['patients']:
                df = pd.DataFrame(study_data['patients'])
                st.markdown(f"**{len(df)} patients**")
                st.dataframe(df.head(50), use_container_width=True, hide_index=True)

                # Show code
                with st.expander("View Loading Code"):
                    st.code("""
# From data_tools.py - ClinicalDataLoader
def load_patient_data(self, study_path):
    edc_file = self.find_file(study_path, "EDC_Metrics")
    df = pd.read_excel(edc_file)
    patients = []
    for _, row in df.iterrows():
        patients.append({
            "patient_id": row.get("SUBJID", row.get("Subject", "")),
            "site_id": row.get("SITEID", row.get("Site", "")),
            "status": row.get("Status", "Active"),
            ...
        })
    return patients
                    """, language="python")

        with tab2:
            st.subheader("Site Metrics")
            site_data = []
            for site_id, data in study_data['sites'].items():
                site_data.append({
                    "Site ID": site_id,
                    "Patients": len(data['patients']),
                    "Open Queries": data['open_queries'],
                    "Missing Pages": data['missing_pages'],
                    "Lab Issues": data['lab_issues']
                })

            df = pd.DataFrame(site_data)
            st.markdown(f"**{len(df)} sites with aggregated metrics**")
            st.dataframe(df.head(50), use_container_width=True, hide_index=True)

        with tab3:
            st.subheader("Query Data")
            if study_data['queries']:
                df = pd.DataFrame(study_data['queries'])
                st.markdown(f"**{len(df)} queries**")

                # Filter options
                col1, col2 = st.columns(2)
                with col1:
                    status_filter = st.selectbox("Filter by Status", ["All"] + list(df['status'].unique()))
                with col2:
                    if status_filter != "All":
                        df = df[df['status'] == status_filter]

                st.dataframe(df.head(50), use_container_width=True, hide_index=True)

                # Query aging distribution
                if 'days_open' in df.columns:
                    st.markdown("**Query Aging Distribution:**")
                    bins = [0, 7, 15, 30, 60, float('inf')]
                    labels = ['0-7d', '8-15d', '16-30d', '31-60d', '>60d']
                    df['aging_bucket'] = pd.cut(df['days_open'], bins=bins, labels=labels)
                    aging_dist = df['aging_bucket'].value_counts().sort_index()
                    st.bar_chart(aging_dist)

        with tab4:
            st.subheader("Study Summary")

            summary = {
                "Study": selected_study,
                "Total Patients": len(study_data['patients']),
                "Total Sites": len(study_data['sites']),
                "Total Queries": len(study_data['queries']),
                "Open Queries": sum(s['open_queries'] for s in study_data['sites'].values()),
                "Missing Pages": sum(s['missing_pages'] for s in study_data['sites'].values()),
                "Aging Queries (>15d)": len([q for q in study_data['queries']
                                             if q['status'] == 'Open' and q['days_open'] > 15])
            }

            st.code(json.dumps(summary, indent=2), language="json")


def render_architecture():
    """Render architecture diagram - addresses 'integrated data flow model' requirement"""
    st.header("NEXUS CDM - INTEGRATED DATA FLOW MODEL")
    st.markdown("*Addressing problem statement: 'integrated solution that can ingest and harmonize clinical and operational data'*")

    # CSS for architecture boxes
    st.markdown("""
    <style>
    .arch-layer {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        border: 2px solid #4a90d9;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        text-align: center;
    }
    .arch-layer-title {
        color: #7cb9e8;
        font-size: 14px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .arch-box {
        background: #252526;
        border: 1px solid #569cd6;
        border-radius: 6px;
        padding: 8px 12px;
        margin: 4px;
        display: inline-block;
        color: #dcdcaa;
        font-size: 12px;
    }
    .arch-box-agent {
        background: linear-gradient(135deg, #2d4a3e 0%, #3d6a5e 100%);
        border: 1px solid #4ec9b0;
    }
    .arch-box-analytics {
        background: linear-gradient(135deg, #4a3d2d 0%, #6a5d3d 100%);
        border: 1px solid #dcdcaa;
    }
    .arch-box-output {
        background: linear-gradient(135deg, #3d2d4a 0%, #5d3d6a 100%);
        border: 1px solid #c586c0;
    }
    .arch-arrow {
        color: #569cd6;
        font-size: 24px;
        text-align: center;
        margin: 5px 0;
    }
    .arch-flow {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Layer 1: Data Sources
    st.markdown("""
    <div class="arch-layer">
        <div class="arch-layer-title">DATA SOURCES (Heterogeneous)</div>
        <div class="arch-flow">
            <span class="arch-box">EDC/eCRF<br/><small>Rave EDC</small></span>
            <span class="arch-box">Lab LIMS<br/><small>LabWare</small></span>
            <span class="arch-box">SAE System<br/><small>Argus</small></span>
            <span class="arch-box">Coding<br/><small>MedDRA/WHO-Drug</small></span>
            <span class="arch-box">Site Ops<br/><small>CTMS</small></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="arch-arrow">▼</div>', unsafe_allow_html=True)

    # Layer 2: Integration
    st.markdown("""
    <div class="arch-layer">
        <div class="arch-layer-title">NEXUS INTEGRATION LAYER</div>
        <div class="arch-flow">
            <span class="arch-box">Excel Ingestion<br/><small>pandas</small></span>
            <span style="color:#569cd6;">→</span>
            <span class="arch-box">Schema Extraction<br/><small>dtype inference</small></span>
            <span style="color:#569cd6;">→</span>
            <span class="arch-box">Data Harmonization<br/><small>normalization</small></span>
            <span style="color:#569cd6;">→</span>
            <span class="arch-box">Unification<br/><small>merge</small></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="arch-arrow">▼</div>', unsafe_allow_html=True)

    # Layer 3: AI Agents
    st.markdown("""
    <div class="arch-layer">
        <div class="arch-layer-title">AI AGENT LAYER (GPT-5.2 Responses API)</div>
        <div class="arch-flow">
            <span class="arch-box arch-box-agent">ORCHESTRATOR<br/><small>Query routing & synthesis</small></span>
            <span style="color:#4ec9b0;">↔</span>
            <span class="arch-box arch-box-agent">DATA STEWARD<br/><small>Query/DQI analysis</small></span>
            <span style="color:#4ec9b0;">↔</span>
            <span class="arch-box arch-box-agent">MEDICAL MONITOR<br/><small>Safety/SAE analysis</small></span>
            <span style="color:#4ec9b0;">↔</span>
            <span class="arch-box arch-box-agent">REGULATORY<br/><small>Compliance assessment</small></span>
        </div>
        <div style="color:#4ec9b0; margin-top:10px; font-size:12px;">15 Specialized Tools (Data, Safety, Compliance)</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="arch-arrow">▼</div>', unsafe_allow_html=True)

    # Layer 4: Analytics
    st.markdown("""
    <div class="arch-layer">
        <div class="arch-layer-title">ANALYTICS ENGINE</div>
        <div class="arch-flow">
            <span class="arch-box arch-box-analytics">DQI Calculator<br/><small>ICH E6(R3) - 9 components</small></span>
            <span class="arch-box arch-box-analytics">KRI Detection<br/><small>RBQM QTLs - 5 metrics</small></span>
            <span class="arch-box arch-box-analytics">Proactive Alerts<br/><small>Real-time issue detection</small></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="arch-arrow">▼</div>', unsafe_allow_html=True)

    # Layer 5: Outputs
    st.markdown("""
    <div class="arch-layer">
        <div class="arch-layer-title">ACTIONABLE OUTPUTS</div>
        <div class="arch-flow">
            <span class="arch-box arch-box-output">Proactive Alerts<br/><small>CRITICAL/HIGH/MEDIUM</small></span>
            <span class="arch-box arch-box-output">Auto-Generated<br/><small>Queries & Reports</small></span>
            <span class="arch-box arch-box-output">Dashboards<br/><small>CTT/CRA/Site views</small></span>
            <span class="arch-box arch-box-output">Regulatory<br/><small>FDA/EMA submissions</small></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Architecture metrics
    st.subheader("Architecture Metrics")
    cols = st.columns(5)
    with cols[0]:
        st.metric("Data Sources", "9 systems", help="EDC, Lab, SAE, Coding, CTMS, etc.")
    with cols[1]:
        st.metric("Processing", "<2 seconds", help="Near real-time ingestion")
    with cols[2]:
        st.metric("AI Agents", "4 specialized", help="Orchestrator + 3 domain experts")
    with cols[3]:
        st.metric("Tools", "15 functions", help="Data, Safety, Compliance")
    with cols[4]:
        st.metric("KRIs Monitored", "5 metrics", help="Per RBQM framework")


def main():
    st.title("NEXUS CDM - Data Platform")
    st.markdown("**Clinical Data Ingestion and Transformation**")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.markdown("## Navigation")
        section = st.radio(
            "Section",
            ["Architecture", "Source Browser", "Schema Extraction", "Transformation", "Data Explorer"]
        )

        st.markdown("---")
        st.markdown("### Data Source")
        st.code(f"""
Path: {DATA_BASE_PATH[:40]}...
        """, language=None)

        # Stats
        folders = get_study_folders()
        total_files = sum(f['file_count'] for f in folders)

        st.markdown("### Source Statistics")
        st.code(f"""
Study Folders: {len(folders)}
Excel Files: {total_files}
        """, language=None)

        st.markdown("### File Patterns")
        pattern_text = ""
        for key, pattern in FILE_PATTERNS.items():
            pattern_text += f"{key}: {pattern}*\n"
        st.code(pattern_text, language=None)

        # Impact metrics
        st.markdown("---")
        st.markdown("### NEXUS IMPACT")
        st.metric("Time Saved", "47 hrs/week", help="vs manual review")
        st.metric("Early Detection", "12 days avg", help="Issues found earlier")
        st.metric("Query Resolution", "2.8 days", "-0.4d", help="Benchmark: 3.2d")

    # Main content
    if section == "Architecture":
        render_architecture()
    elif section == "Source Browser":
        render_source_browser()
    elif section == "Schema Extraction":
        render_schema_extraction()
    elif section == "Transformation":
        render_transformation_trace()
    else:
        render_data_explorer()


if __name__ == "__main__":
    main()
