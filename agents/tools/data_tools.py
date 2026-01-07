"""
NEXUS CDM Data Tools
====================
Tools for clinical trial data analysis and query management.

These tools implement Key Risk Indicators (KRIs) and support
Risk-Based Quality Management (RBQM) per ICH E6(R3).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import os
import sys

from config import DATA_BASE_PATH, QTL_CONFIG, DQI_WEIGHTS


class ClinicalDataLoader:
    """Singleton data loader for clinical trial data"""
    _instance = None
    _data_cache = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_all_data()
        return cls._instance

    def _load_all_data(self):
        """Load all study data into memory"""
        self._data_cache = {
            'studies': {},
            'summary': {
                'total_patients': 0,
                'total_sites': 0,
                'total_queries_open': 0,
                'total_missing_pages': 0,
                'total_lab_issues': 0
            }
        }

        studies = [d for d in os.listdir(DATA_BASE_PATH) if os.path.isdir(os.path.join(DATA_BASE_PATH, d))]

        for study_folder in studies:
            study_name = study_folder.split('_')[0].replace('STUDY', 'Study')
            study_path = os.path.join(DATA_BASE_PATH, study_folder)
            files = os.listdir(study_path)

            study_data = {
                'name': study_name,
                'patients': {},
                'sites': {},
                'queries': [],
                'missing_pages': [],
                'lab_issues': [],
                'sae_issues': [],
                'coding_issues': []
            }

            # Load EDC Metrics
            edc_file = [f for f in files if 'EDC_Metrics' in f]
            if edc_file:
                self._load_edc_metrics(study_path, edc_file[0], study_data)

            # Load Query data
            if edc_file:
                self._load_query_data(study_path, edc_file[0], study_data)

            # Load Missing Pages
            pages_file = [f for f in files if 'Missing_Pages' in f]
            if pages_file:
                self._load_missing_pages(study_path, pages_file[0], study_data)

            # Load Lab Issues
            lab_file = [f for f in files if 'Lab_Name' in f or 'Missing_Lab' in f]
            if lab_file:
                self._load_lab_issues(study_path, lab_file[0], study_data)

            # Load SAE data
            sae_file = [f for f in files if 'SAE' in f or 'eSAE' in f]
            if sae_file:
                self._load_sae_data(study_path, sae_file[0], study_data)

            # Load Coding data
            medra_file = [f for f in files if 'MedDRA' in f]
            if medra_file:
                self._load_coding_data(study_path, medra_file[0], study_data)

            self._data_cache['studies'][study_name] = study_data

            # Update summary
            self._data_cache['summary']['total_patients'] += len(study_data['patients'])
            self._data_cache['summary']['total_sites'] += len(study_data['sites'])

    def _load_edc_metrics(self, study_path: str, filename: str, study_data: Dict):
        """Load EDC metrics data"""
        try:
            filepath = os.path.join(study_path, filename)
            df = pd.read_excel(filepath, sheet_name='Subject Level Metrics', header=None, skiprows=4)

            for _, row in df.iterrows():
                patient_id = str(row[4]) if pd.notna(row[4]) else None
                if not patient_id or patient_id == 'nan':
                    continue

                site_id = str(row[3]) if pd.notna(row[3]) else 'Unknown'
                region = str(row[1]) if pd.notna(row[1]) else 'Unknown'
                country = str(row[2]) if pd.notna(row[2]) else 'Unknown'
                status = str(row[6]) if pd.notna(row[6]) else 'Unknown'

                study_data['patients'][patient_id] = {
                    'patient_id': patient_id,
                    'site_id': site_id,
                    'region': region,
                    'country': country,
                    'status': status,
                    'expected_visits': int(row[16]) if pd.notna(row[16]) else 0,
                    'pages_entered': int(row[17]) if pd.notna(row[17]) else 0,
                    'non_conformant': int(row[18]) if pd.notna(row[18]) else 0,
                    'missing_visits': 0,
                    'missing_pages': 0,
                    'open_queries': 0,
                    'lab_issues': 0,
                    'sae_issues': 0,
                    'uncoded_terms': 0
                }

                if site_id not in study_data['sites']:
                    study_data['sites'][site_id] = {
                        'site_id': site_id,
                        'region': region,
                        'country': country,
                        'patients': [],
                        'total_queries': 0,
                        'open_queries': 0,
                        'missing_pages': 0,
                        'lab_issues': 0,
                        'sae_issues': 0
                    }
                study_data['sites'][site_id]['patients'].append(patient_id)
        except Exception as e:
            print(f"Error loading EDC metrics: {e}")

    def _load_query_data(self, study_path: str, filename: str, study_data: Dict):
        """Load query report data"""
        try:
            filepath = os.path.join(study_path, filename)
            df = pd.read_excel(filepath, sheet_name='Query Report - Cumulative', header=None, skiprows=1)

            for _, row in df.iterrows():
                if len(row) < 15:
                    continue

                query = {
                    'site_id': str(row[3]) if pd.notna(row[3]) else 'Unknown',
                    'patient_id': str(row[4]) if pd.notna(row[4]) else 'Unknown',
                    'folder': str(row[5]) if pd.notna(row[5]) else '',
                    'form': str(row[6]) if pd.notna(row[6]) else '',
                    'status': str(row[10]) if pd.notna(row[10]) else 'Unknown',
                    'action_owner': str(row[11]) if pd.notna(row[11]) else '',
                    'open_date': row[13] if pd.notna(row[13]) else None,
                    'days_open': int(row[15]) if pd.notna(row[15]) else 0
                }
                study_data['queries'].append(query)

                # Update patient and site counts
                if query['status'] == 'Open':
                    if query['patient_id'] in study_data['patients']:
                        study_data['patients'][query['patient_id']]['open_queries'] += 1
                    if query['site_id'] in study_data['sites']:
                        study_data['sites'][query['site_id']]['open_queries'] += 1
        except Exception as e:
            print(f"Error loading query data: {e}")

    def _load_missing_pages(self, study_path: str, filename: str, study_data: Dict):
        """Load missing pages data"""
        try:
            filepath = os.path.join(study_path, filename)
            df = pd.read_excel(filepath)

            for _, row in df.iterrows():
                patient_col = 'SubjectName' if 'SubjectName' in df.columns else df.columns[3]
                site_col = 'SiteNumber' if 'SiteNumber' in df.columns else df.columns[2]

                days_val = row.get('No. #Days Page Missing', 0) if 'No. #Days Page Missing' in df.columns else 0
                page = {
                    'patient_id': str(row[patient_col]) if pd.notna(row[patient_col]) else 'Unknown',
                    'site_id': str(row[site_col]) if pd.notna(row[site_col]) else 'Unknown',
                    'form_name': str(row.get('FormName', '')) if 'FormName' in df.columns else '',
                    'days_missing': int(days_val) if pd.notna(days_val) else 0
                }
                study_data['missing_pages'].append(page)

                # Update counts
                if page['patient_id'] in study_data['patients']:
                    study_data['patients'][page['patient_id']]['missing_pages'] += 1
                if page['site_id'] in study_data['sites']:
                    study_data['sites'][page['site_id']]['missing_pages'] += 1
        except Exception as e:
            print(f"Error loading missing pages: {e}")

    def _load_lab_issues(self, study_path: str, filename: str, study_data: Dict):
        """Load lab issues data"""
        try:
            filepath = os.path.join(study_path, filename)
            df = pd.read_excel(filepath)

            for _, row in df.iterrows():
                issue = {
                    'patient_id': str(row.get('Subject', '')) if 'Subject' in df.columns else '',
                    'site_id': str(row.get('Site number', '')) if 'Site number' in df.columns else '',
                    'issue_type': str(row.get('Issue', '')) if 'Issue' in df.columns else '',
                    'lab_category': str(row.get('Lab category', '')) if 'Lab category' in df.columns else ''
                }
                study_data['lab_issues'].append(issue)

                # Update counts
                if issue['patient_id'] in study_data['patients']:
                    study_data['patients'][issue['patient_id']]['lab_issues'] += 1
                if issue['site_id'] in study_data['sites']:
                    study_data['sites'][issue['site_id']]['lab_issues'] += 1
        except Exception as e:
            print(f"Error loading lab issues: {e}")

    def _load_sae_data(self, study_path: str, filename: str, study_data: Dict):
        """Load SAE dashboard data"""
        try:
            filepath = os.path.join(study_path, filename)
            xl = pd.ExcelFile(filepath)

            for sheet in xl.sheet_names:
                df = pd.read_excel(xl, sheet_name=sheet)
                for _, row in df.iterrows():
                    if 'Review Status' in df.columns:
                        status = str(row.get('Review Status', ''))
                        if 'Pending' in status:
                            issue = {
                                'patient_id': str(row.get('Patient ID', '')) if 'Patient ID' in df.columns else '',
                                'site_id': str(row.get('Site', '')) if 'Site' in df.columns else '',
                                'status': status,
                                'type': sheet
                            }
                            study_data['sae_issues'].append(issue)
        except Exception as e:
            print(f"Error loading SAE data: {e}")

    def _load_coding_data(self, study_path: str, filename: str, study_data: Dict):
        """Load coding status data"""
        try:
            filepath = os.path.join(study_path, filename)
            df = pd.read_excel(filepath)

            if 'Coding Status' in df.columns:
                uncoded = df[df['Coding Status'] == 'UnCoded Term']
                for _, row in uncoded.iterrows():
                    issue = {
                        'patient_id': str(row.get('Subject', '')) if 'Subject' in df.columns else '',
                        'form': str(row.get('Form', '')) if 'Form' in df.columns else '',
                        'status': 'UnCoded Term'
                    }
                    study_data['coding_issues'].append(issue)

                    if issue['patient_id'] in study_data['patients']:
                        study_data['patients'][issue['patient_id']]['uncoded_terms'] += 1
        except Exception as e:
            print(f"Error loading coding data: {e}")

    def get_study_data(self, study_name: str) -> Optional[Dict]:
        """Get data for a specific study"""
        return self._data_cache['studies'].get(study_name)

    def get_all_studies(self) -> Dict:
        """Get all study data"""
        return self._data_cache['studies']

    def get_summary(self) -> Dict:
        """Get summary statistics"""
        return self._data_cache['summary']


# Initialize global data loader
_data_loader = None

def get_data_loader() -> ClinicalDataLoader:
    global _data_loader
    if _data_loader is None:
        _data_loader = ClinicalDataLoader()
    return _data_loader


# =============================================================================
# TOOL FUNCTIONS
# =============================================================================

def get_study_overview(study_name: str = None) -> Dict:
    """
    Get overview of study(s) with key metrics.

    KRI: Study-level performance indicators
    """
    loader = get_data_loader()

    if study_name:
        study = loader.get_study_data(study_name)
        if not study:
            return {"error": f"Study '{study_name}' not found"}

        return {
            "study": study_name,
            "total_patients": len(study['patients']),
            "total_sites": len(study['sites']),
            "total_queries": len(study['queries']),
            "open_queries": sum(1 for q in study['queries'] if q['status'] == 'Open'),
            "missing_pages": len(study['missing_pages']),
            "lab_issues": len(study['lab_issues']),
            "sae_pending": len(study['sae_issues']),
            "uncoded_terms": len(study['coding_issues'])
        }
    else:
        all_studies = loader.get_all_studies()
        result = {"studies": []}
        for name, study in all_studies.items():
            result["studies"].append({
                "study": name,
                "patients": len(study['patients']),
                "sites": len(study['sites']),
                "open_queries": sum(1 for q in study['queries'] if q['status'] == 'Open'),
                "missing_pages": len(study['missing_pages'])
            })
        result["total_studies"] = len(all_studies)
        result["total_patients"] = sum(len(s['patients']) for s in all_studies.values())
        result["total_sites"] = sum(len(s['sites']) for s in all_studies.values())
        return result


def get_query_aging_report(study_name: str, threshold_days: int = 15) -> Dict:
    """
    Get query aging report per RBQM KRI requirements.

    Industry benchmark: High-performing sites resolve queries in <3.2 days.
    QTL thresholds: Warning at 15 days, Critical at 30 days.
    """
    loader = get_data_loader()

    # Support "all" for cross-study analysis
    if study_name.lower() == "all":
        all_studies = loader.get_all_studies()
        combined_result = {
            "analysis_type": "cross_study",
            "studies_analyzed": len(all_studies),
            "total_open_queries": 0,
            "total_queries_over_threshold": 0,
            "aging_distribution": {"0-5_days": 0, "5-15_days": 0, "15-30_days_WARNING": 0, "over_30_days_CRITICAL": 0},
            "study_breakdown": [],
            "worst_sites_across_studies": []
        }
        all_aging_sites = []
        for name, study in all_studies.items():
            queries = study['queries']
            open_queries = [q for q in queries if q['status'] == 'Open']
            aging_over = [q for q in open_queries if q['days_open'] > threshold_days]
            combined_result["total_open_queries"] += len(open_queries)
            combined_result["total_queries_over_threshold"] += len(aging_over)
            combined_result["aging_distribution"]["0-5_days"] += len([q for q in open_queries if q['days_open'] <= 5])
            combined_result["aging_distribution"]["5-15_days"] += len([q for q in open_queries if 5 < q['days_open'] <= 15])
            combined_result["aging_distribution"]["15-30_days_WARNING"] += len([q for q in open_queries if 15 < q['days_open'] <= 30])
            combined_result["aging_distribution"]["over_30_days_CRITICAL"] += len([q for q in open_queries if q['days_open'] > 30])
            combined_result["study_breakdown"].append({
                "study": name, "open_queries": len(open_queries), "over_threshold": len(aging_over)
            })
            # Collect worst sites
            for q in open_queries:
                if q['days_open'] > threshold_days:
                    all_aging_sites.append({"study": name, "site_id": q['site_id'], "days": q['days_open']})
        # Get top 10 worst sites across all studies
        site_counts = {}
        for s in all_aging_sites:
            key = f"{s['study']}:{s['site_id']}"
            if key not in site_counts:
                site_counts[key] = {"study": s['study'], "site_id": s['site_id'], "count": 0, "max_days": 0}
            site_counts[key]["count"] += 1
            site_counts[key]["max_days"] = max(site_counts[key]["max_days"], s['days'])
        combined_result["worst_sites_across_studies"] = sorted(site_counts.values(), key=lambda x: x['count'], reverse=True)[:10]
        return combined_result

    study = loader.get_study_data(study_name)

    if not study:
        return {"error": f"Study '{study_name}' not found"}

    queries = study['queries']
    open_queries = [q for q in queries if q['status'] == 'Open']

    # Categorize by aging
    aging_0_5 = [q for q in open_queries if q['days_open'] <= 5]
    aging_5_15 = [q for q in open_queries if 5 < q['days_open'] <= 15]
    aging_15_30 = [q for q in open_queries if 15 < q['days_open'] <= 30]
    aging_over_30 = [q for q in open_queries if q['days_open'] > 30]

    # Get sites with most aging queries
    site_aging = {}
    for q in open_queries:
        if q['days_open'] > threshold_days:
            site = q['site_id']
            if site not in site_aging:
                site_aging[site] = {'count': 0, 'queries': []}
            site_aging[site]['count'] += 1
            site_aging[site]['queries'].append({
                'patient': q['patient_id'],
                'form': q['form'],
                'days': q['days_open']
            })

    # Sort sites by aging query count
    worst_sites = sorted(site_aging.items(), key=lambda x: x[1]['count'], reverse=True)[:5]

    return {
        "study": study_name,
        "total_open_queries": len(open_queries),
        "aging_distribution": {
            "0-5_days": len(aging_0_5),
            "5-15_days": len(aging_5_15),
            "15-30_days_WARNING": len(aging_15_30),
            "over_30_days_CRITICAL": len(aging_over_30)
        },
        "queries_over_threshold": len([q for q in open_queries if q['days_open'] > threshold_days]),
        "worst_performing_sites": [
            {
                "site_id": site,
                "queries_aging": data['count'],
                "sample_queries": data['queries'][:3]
            }
            for site, data in worst_sites
        ],
        "kri_status": "CRITICAL" if aging_over_30 else "WARNING" if aging_15_30 else "GOOD",
        "benchmark": "Industry target: <3.2 days average resolution"
    }


def get_missing_pages_by_site(study_name: str) -> Dict:
    """
    Get missing pages report organized by site.

    KRI: Data completeness indicator
    QTL: >2% missing pages triggers review
    """
    loader = get_data_loader()

    # Support "all" for cross-study analysis
    if study_name.lower() == "all":
        all_studies = loader.get_all_studies()
        combined_result = {
            "analysis_type": "cross_study",
            "studies_analyzed": len(all_studies),
            "total_missing_pages": 0,
            "total_patients": 0,
            "study_breakdown": [],
            "worst_sites_across_studies": []
        }
        all_site_missing = []
        for name, study in all_studies.items():
            missing = len(study['missing_pages'])
            patients = len(study['patients'])
            combined_result["total_missing_pages"] += missing
            combined_result["total_patients"] += patients
            combined_result["study_breakdown"].append({
                "study": name, "missing_pages": missing, "patients": patients,
                "rate": round(missing / max(patients, 1) * 100, 2)
            })
            # Collect site data
            for site_id, site_data in study['sites'].items():
                if site_data['missing_pages'] > 0:
                    all_site_missing.append({
                        "study": name, "site_id": site_id,
                        "missing": site_data['missing_pages'], "patients": len(site_data['patients'])
                    })
        combined_result["overall_missing_rate"] = round(
            combined_result["total_missing_pages"] / max(combined_result["total_patients"], 1) * 100, 2
        )
        combined_result["worst_sites_across_studies"] = sorted(
            all_site_missing, key=lambda x: x['missing'], reverse=True
        )[:10]
        return combined_result

    study = loader.get_study_data(study_name)

    if not study:
        return {"error": f"Study '{study_name}' not found"}

    missing = study['missing_pages']

    # Group by site
    site_missing = {}
    for page in missing:
        site = page['site_id']
        if site not in site_missing:
            site_missing[site] = {'count': 0, 'forms': [], 'patients': set()}
        site_missing[site]['count'] += 1
        site_missing[site]['forms'].append(page['form_name'])
        site_missing[site]['patients'].add(page['patient_id'])

    # Calculate rates
    total_patients = len(study['patients'])
    total_missing = len(missing)

    # Get worst sites
    worst_sites = sorted(site_missing.items(), key=lambda x: x[1]['count'], reverse=True)[:5]

    return {
        "study": study_name,
        "total_missing_pages": total_missing,
        "total_patients": total_patients,
        "missing_rate": round(total_missing / max(total_patients, 1) * 100, 2),
        "sites_with_missing": len(site_missing),
        "worst_sites": [
            {
                "site_id": site,
                "missing_count": data['count'],
                "patients_affected": len(data['patients']),
                "common_forms": list(set(data['forms']))[:3]
            }
            for site, data in worst_sites
        ],
        "qtl_status": "BREACH" if (total_missing / max(total_patients, 1)) > 0.02 else "WITHIN_LIMIT"
    }


def get_site_dqi_scores(study_name: str) -> Dict:
    """
    Calculate Data Quality Index (DQI) for all sites in a study.

    DQI Formula: 100 - Σ(Wi × Di)
    Weights based on ICH E6(R3) regulatory impact.
    """
    loader = get_data_loader()

    # Support "all" for cross-study analysis
    if study_name.lower() == "all":
        all_studies = loader.get_all_studies()
        combined_result = {
            "analysis_type": "cross_study",
            "studies_analyzed": len(all_studies),
            "total_sites": 0,
            "overall_average_dqi": 0,
            "total_at_risk": 0,
            "study_breakdown": [],
            "worst_sites_across_studies": []
        }
        all_site_scores = []
        all_dqi_scores = []
        for name, study in all_studies.items():
            study_sites = []
            for site_id, site_data in study['sites'].items():
                num_patients = len(site_data['patients'])
                if num_patients == 0:
                    continue
                issues = site_data['open_queries'] + site_data['missing_pages'] + site_data['lab_issues']
                dqi = max(0, 100 - (issues / num_patients) * 20)
                all_dqi_scores.append(dqi)
                status = "AT_RISK" if dqi < 70 else "NEEDS_ATTENTION" if dqi < 85 else "GOOD"
                site_info = {"study": name, "site_id": site_id, "dqi_score": round(dqi, 2), "status": status, "patients": num_patients}
                study_sites.append(site_info)
                all_site_scores.append(site_info)
            avg_dqi = np.mean([s['dqi_score'] for s in study_sites]) if study_sites else 0
            at_risk = len([s for s in study_sites if s['status'] == 'AT_RISK'])
            combined_result["study_breakdown"].append({
                "study": name, "sites": len(study_sites), "avg_dqi": round(avg_dqi, 2), "at_risk": at_risk
            })
            combined_result["total_sites"] += len(study_sites)
            combined_result["total_at_risk"] += at_risk
        combined_result["overall_average_dqi"] = round(np.mean(all_dqi_scores), 2) if all_dqi_scores else 0
        combined_result["worst_sites_across_studies"] = sorted(all_site_scores, key=lambda x: x['dqi_score'])[:10]
        return combined_result

    study = loader.get_study_data(study_name)

    if not study:
        return {"error": f"Study '{study_name}' not found"}

    site_scores = []

    for site_id, site_data in study['sites'].items():
        patients = site_data['patients']
        num_patients = len(patients)

        if num_patients == 0:
            continue

        # Calculate deficiency rates
        open_queries = site_data['open_queries']
        missing_pages = site_data['missing_pages']
        lab_issues = site_data['lab_issues']
        sae_issues = site_data.get('sae_issues', 0)

        # Normalize rates (per patient)
        query_rate = min(open_queries / num_patients, 1.0)
        missing_rate = min(missing_pages / (num_patients * 10), 1.0)  # Assume ~10 pages per patient
        lab_rate = min(lab_issues / num_patients, 1.0)
        sae_rate = min(sae_issues / max(num_patients * 0.1, 1), 1.0)  # ~10% SAE expected

        # Calculate DQI
        deductions = (
            DQI_WEIGHTS.open_queries * query_rate * 100 +
            DQI_WEIGHTS.crf_completion * missing_rate * 100 +
            DQI_WEIGHTS.lab_integrity * lab_rate * 100 +
            DQI_WEIGHTS.sae_reconciliation * sae_rate * 100
        )

        dqi = max(0, 100 - deductions)

        # Determine status
        if dqi >= 95:
            status = "EXCELLENT"
        elif dqi >= 85:
            status = "GOOD"
        elif dqi >= 70:
            status = "NEEDS_ATTENTION"
        else:
            status = "AT_RISK"

        site_scores.append({
            "site_id": site_id,
            "dqi_score": round(dqi, 2),
            "status": status,
            "patients": num_patients,
            "open_queries": open_queries,
            "missing_pages": missing_pages,
            "lab_issues": lab_issues,
            "region": site_data['region'],
            "country": site_data['country']
        })

    # Sort by DQI score (ascending to show worst first)
    site_scores.sort(key=lambda x: x['dqi_score'])

    # Get sites needing attention
    at_risk = [s for s in site_scores if s['status'] in ['AT_RISK', 'NEEDS_ATTENTION']]

    return {
        "study": study_name,
        "total_sites": len(site_scores),
        "average_dqi": round(np.mean([s['dqi_score'] for s in site_scores]), 2),
        "sites_at_risk": len(at_risk),
        "worst_5_sites": site_scores[:5],
        "best_5_sites": site_scores[-5:][::-1],
        "status_distribution": {
            "EXCELLENT": len([s for s in site_scores if s['status'] == 'EXCELLENT']),
            "GOOD": len([s for s in site_scores if s['status'] == 'GOOD']),
            "NEEDS_ATTENTION": len([s for s in site_scores if s['status'] == 'NEEDS_ATTENTION']),
            "AT_RISK": len([s for s in site_scores if s['status'] == 'AT_RISK'])
        }
    }


def get_high_risk_sites(study_name: str = None, dqi_threshold: float = 85.0) -> Dict:
    """
    Identify high-risk sites across studies per RBQM framework.

    Uses QTL threshold for site DQI to flag sites requiring intervention.
    """
    loader = get_data_loader()

    if study_name:
        studies = {study_name: loader.get_study_data(study_name)}
    else:
        studies = loader.get_all_studies()

    high_risk_sites = []

    for name, study in studies.items():
        if not study:
            continue

        for site_id, site_data in study['sites'].items():
            patients = site_data['patients']
            num_patients = len(patients)

            if num_patients == 0:
                continue

            # Quick DQI calculation
            issues = (
                site_data['open_queries'] +
                site_data['missing_pages'] +
                site_data['lab_issues']
            )
            issue_rate = issues / num_patients

            # Simplified risk score
            dqi = max(0, 100 - issue_rate * 20)

            if dqi < dqi_threshold:
                high_risk_sites.append({
                    "study": name,
                    "site_id": site_id,
                    "dqi_score": round(dqi, 2),
                    "patients": num_patients,
                    "open_queries": site_data['open_queries'],
                    "missing_pages": site_data['missing_pages'],
                    "lab_issues": site_data['lab_issues'],
                    "country": site_data['country'],
                    "risk_factors": []
                })

                # Add specific risk factors
                if site_data['open_queries'] > 5:
                    high_risk_sites[-1]['risk_factors'].append(f"{site_data['open_queries']} open queries")
                if site_data['missing_pages'] > 10:
                    high_risk_sites[-1]['risk_factors'].append(f"{site_data['missing_pages']} missing pages")
                if site_data['lab_issues'] > 5:
                    high_risk_sites[-1]['risk_factors'].append(f"{site_data['lab_issues']} lab issues")

    # Sort by DQI
    high_risk_sites.sort(key=lambda x: x['dqi_score'])

    return {
        "threshold_used": dqi_threshold,
        "total_high_risk_sites": len(high_risk_sites),
        "sites": high_risk_sites[:20],  # Top 20 worst
        "recommended_actions": [
            "Schedule immediate CRA visit for sites with DQI < 70",
            "Prioritize query resolution for sites with >10 open queries",
            "Conduct lab data review for sites with >5 lab issues"
        ]
    }


# Tool definitions for agent registration
DATA_TOOLS = [
    {
        "name": "get_study_overview",
        "description": "Get overview of one or all studies with key metrics including patient count, site count, open queries, and missing pages.",
        "parameters": {
            "type": "object",
            "properties": {
                "study_name": {
                    "type": "string",
                    "description": "Name of the study (e.g., 'Study 1'). If not provided, returns overview of all studies."
                }
            },
            "required": [],
            "additionalProperties": False
        },
        "function": get_study_overview
    },
    {
        "name": "get_query_aging_report",
        "description": "Get query aging report showing queries by age category. Identifies sites with aging queries exceeding RBQM thresholds.",
        "parameters": {
            "type": "object",
            "properties": {
                "study_name": {
                    "type": "string",
                    "description": "Name of the study to analyze"
                },
                "threshold_days": {
                    "type": "integer",
                    "description": "Day threshold for highlighting aging queries (default: 15)"
                }
            },
            "required": ["study_name"],
            "additionalProperties": False
        },
        "function": get_query_aging_report
    },
    {
        "name": "get_missing_pages_by_site",
        "description": "Get missing CRF pages report organized by site. Identifies sites with data completeness issues.",
        "parameters": {
            "type": "object",
            "properties": {
                "study_name": {
                    "type": "string",
                    "description": "Name of the study to analyze"
                }
            },
            "required": ["study_name"],
            "additionalProperties": False
        },
        "function": get_missing_pages_by_site
    },
    {
        "name": "get_site_dqi_scores",
        "description": "Calculate Data Quality Index (DQI) scores for all sites in a study. DQI uses patient-safety-weighted metrics per ICH E6(R3).",
        "parameters": {
            "type": "object",
            "properties": {
                "study_name": {
                    "type": "string",
                    "description": "Name of the study to analyze"
                }
            },
            "required": ["study_name"],
            "additionalProperties": False
        },
        "function": get_site_dqi_scores
    },
    {
        "name": "get_high_risk_sites",
        "description": "Identify high-risk sites across one or all studies based on DQI threshold. Per RBQM framework for centralized monitoring.",
        "parameters": {
            "type": "object",
            "properties": {
                "study_name": {
                    "type": "string",
                    "description": "Name of the study (optional - if not provided, analyzes all studies)"
                },
                "dqi_threshold": {
                    "type": "number",
                    "description": "DQI threshold below which sites are flagged as high-risk (default: 85.0)"
                }
            },
            "required": [],
            "additionalProperties": False
        },
        "function": get_high_risk_sites
    }
]
