# medical_report_analysis/main.py

import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.metrics.pairwise import cosine_similarity
import json
import re
from typing import List, Dict, Any, Tuple

class MedicalReportAnalyzer:
    """
    Analyseur de rapports médicaux utilisant BioBERT pour extraire des informations
    et générer des recommandations de traitement
    """
    
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT"):
        """
        Initialise l'analyseur avec le modèle spécifié
        
        Args:
            model_name: Nom du modèle à utiliser (par défaut: Bio_ClinicalBERT)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Utilisation de l'appareil: {self.device}")
    
    # Chargement du tokenizer et du modèle
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
    
    # Au lieu d'utiliser la pipeline NER, nous utiliserons des expressions régulières
    # pour l'extraction d'entités (plus simple mais moins précis)
        self.ner_pipeline = None
    
    # Chargement de la base de connaissances
        self.knowledge_base = self._load_knowledge_base()
    
    # Patterns pour l'extraction d'informations
        self.patterns = {
        'age': r'(\d+)[-\s]ans',
        'birads': r'BI-?RADS\s*(\d[A-C]?)',
        'size': r'(\d+(?:\.\d+)?)\s*(?:mm|cm)',
        'location': r'quadrant\s+(\w+[\s-]\w+|supéro-externe|inféro-externe|supéro-interne|inféro-interne|QSE|QSI|QIE|QII|UQO|UQI|LQO|LQI)'
        }
        
        # Chargement de la base de connaissances
        self.knowledge_base = self._load_knowledge_base()
        
        # Patterns pour l'extraction d'informations
        self.patterns = {
            'age': r'(\d+)[-\s]ans',
            'birads': r'BI-?RADS\s*(\d[A-C]?)',
            'size': r'(\d+(?:\.\d+)?)\s*(?:mm|cm)',
            'location': r'quadrant\s+(\w+[\s-]\w+|supéro-externe|inféro-externe|supéro-interne|inféro-interne|QSE|QSI|QIE|QII|UQO|UQI|LQO|LQI)'
        }
    
    def _load_knowledge_base(self) -> Dict:
        """
        Charge la base de connaissances médicales pour les recommandations
        
        Returns:
            Dictionnaire contenant les guidelines et recommandations
        """
        # Dans un système réel, cela pourrait charger depuis une base de données
        # Pour l'exemple, nous définissons une base de connaissances simplifiée
        return {
            "birads_guidelines": {
                "0": {
                    "description": "Évaluation incomplète",
                    "recommendations": ["Examens d'imagerie supplémentaires requis"],
                    "follow_up": "Immédiat",
                    "malignancy_risk": "N/A"
                },
                "1": {
                    "description": "Négatif",
                    "recommendations": ["Dépistage de routine"],
                    "follow_up": "1 an",
                    "malignancy_risk": "< 2%"
                },
                "2": {
                    "description": "Bénin",
                    "recommendations": ["Dépistage de routine"],
                    "follow_up": "1 an",
                    "malignancy_risk": "< 2%"
                },
                "3": {
                    "description": "Probablement bénin",
                    "recommendations": ["Suivi à court terme"],
                    "follow_up": "6 mois",
                    "malignancy_risk": "> 2% mais ≤ 10%"
                },
                "4A": {
                    "description": "Anomalie suspecte (faible suspicion)",
                    "recommendations": ["Biopsie sous guidage échographique", "Biopsie sous guidage stéréotaxique"],
                    "follow_up": "Biopsie dans les 2 semaines",
                    "malignancy_risk": "> 10% mais ≤ 25%"
                },
                "4B": {
                    "description": "Anomalie suspecte (suspicion modérée)",
                    "recommendations": ["Biopsie sous guidage échographique", "Biopsie sous guidage stéréotaxique"],
                    "follow_up": "Biopsie dans la semaine",
                    "malignancy_risk": "> 25% mais ≤ 50%"
                },
                "4C": {
                    "description": "Anomalie suspecte (suspicion élevée)",
                    "recommendations": ["Biopsie sous guidage échographique", "Biopsie sous guidage stéréotaxique", "IRM mammaire"],
                    "follow_up": "Biopsie immédiate",
                    "malignancy_risk": "> 50% mais < 95%"
                },
                "5": {
                    "description": "Haute suspicion de malignité",
                    "recommendations": ["Biopsie sous guidage échographique", "Biopsie sous guidage stéréotaxique", "IRM mammaire", "Consultation oncologique"],
                    "follow_up": "Biopsie immédiate et consultation en oncologie",
                    "malignancy_risk": "≥ 95%"
                },
                "6": {
                    "description": "Malignité prouvée par biopsie",
                    "recommendations": ["Consultation oncologique", "IRM mammaire pour bilan d'extension", "Évaluation des ganglions axillaires"],
                    "follow_up": "Traitement multidisciplinaire",
                    "malignancy_risk": "100%"
                }
            },
            "treatment_options": {
                "early_stage": {
                    "surgery": ["Tumorectomie", "Mastectomie"],
                    "radiotherapy": ["Radiothérapie adjuvante"],
                    "chemotherapy": ["Chimiothérapie adjuvante (selon facteurs de risque)"],
                    "hormone_therapy": ["Tamoxifène", "Inhibiteurs d'aromatase"],
                    "targeted_therapy": ["Trastuzumab (si HER2+)"]
                },
                "locally_advanced": {
                    "chemotherapy": ["Chimiothérapie néoadjuvante"],
                    "surgery": ["Mastectomie avec curage axillaire"],
                    "radiotherapy": ["Radiothérapie adjuvante"],
                    "hormone_therapy": ["Tamoxifène", "Inhibiteurs d'aromatase"],
                    "targeted_therapy": ["Trastuzumab (si HER2+)", "Pertuzumab (si HER2+)"]
                }
            }
        }
    
    def extract_sections(self, report_text: str) -> Dict[str, str]:
        """
        Extrait les différentes sections d'un rapport médical
        
        Args:
            report_text: Texte du rapport médical
            
        Returns:
            Dictionnaire des sections extraites
        """
        # Définir les sections à rechercher
        section_patterns = {
            "informations_generales": r"((?:Date|Patient|Type)[^\n]*(?:\n(?:Date|Patient|Type)[^\n]*)*)",
            "observations": r"(?:OBSERVATIONS|FINDINGS):\s*(.*?)(?=(?:IMPRESSION|CONCLUSION):)",
            "impression": r"(?:IMPRESSION):\s*(.*?)(?=(?:CONCLUSION):)",
            "conclusion": r"(?:CONCLUSION):\s*(.*?)(?=$|(?:\n\s*\n))"
        }
        
        sections = {}
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, report_text, re.DOTALL | re.IGNORECASE)
            if match:
                sections[section_name] = match.group(1).strip()
            else:
                sections[section_name] = ""
                
        return sections
    
    def extract_patient_info(self, general_info: str) -> Dict[str, Any]:
        """
        Extrait les informations patient du texte général
        
        Args:
            general_info: Texte contenant les informations générales
            
        Returns:
            Dictionnaire des informations patient
        """
        patient_info = {}
        
        # Extraction de l'âge
        age_match = re.search(self.patterns['age'], general_info, re.IGNORECASE)
        if age_match:
            patient_info['age'] = int(age_match.group(1))
        
        # Extraction du sexe
        if re.search(r'\b(?:femme|patiente)\b', general_info, re.IGNORECASE):
            patient_info['gender'] = 'F'
        elif re.search(r'\b(?:homme|patient)\b', general_info, re.IGNORECASE):
            patient_info['gender'] = 'M'
            
        # Extraction de la date d'examen
        date_match = re.search(r'Date[^:]*:\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', general_info)
        if date_match:
            patient_info['exam_date'] = date_match.group(1)
            
        # Type d'examen
        exam_match = re.search(r'Type[^:]*:\s*([^\n]+)', general_info)
        if exam_match:
            patient_info['exam_type'] = exam_match.group(1).strip()
            
        return patient_info
    
    def extract_medical_entities(self, sections: Dict[str, str]) -> Dict[str, Any]:
    
        entities = {
        "patient_info": {},
        "findings": [],
        "birads_classification": None,
        "explicit_recommendations": []
        }
    
    # Extraire les informations du patient
        if "informations_generales" in sections:
            entities["patient_info"] = self.extract_patient_info(sections["informations_generales"])
    
    # Extraire la classification BI-RADS
        all_text = " ".join(sections.values())
        birads_match = re.search(self.patterns['birads'], all_text, re.IGNORECASE)
        if birads_match:
            entities["birads_classification"] = birads_match.group(1)
    
    # Utiliser des regex pour identifier les anomalies
        if "observations" in sections and sections["observations"]:
        # Rechercher des patterns comme "masse/nodule/opacité... XX mm... localisation"
            mass_patterns = [
                r'(masse|nodule|opacité|lésion|calcification)s?\s+([^.;,]*?)\s+(\d+)\s*mm\s+([^.;,]*?)(quadrant|QS|QI)',
                r'(masse|nodule|opacité|lésion|calcification)s?\s+([^.;,]*?)(quadrant|QS|QI)([^.;,]*?)\s+(\d+)\s*mm',
                r'(QSE|QSI|QIE|QII|UQO|UQI|LQO|LQI)[^.;,]*?(masse|nodule|opacité|lésion|calcification)s?[^.;,]*?(\d+)\s*mm'
            ]
        
            for pattern in mass_patterns:
                for match in re.finditer(pattern, sections["observations"], re.IGNORECASE):
                    # Extraire les groupes selon le pattern spécifique
                    groups = match.groups()
                    if len(groups) >= 3:  # Au moins type, taille et quelque chose sur la localisation
                        finding = {}
                    
                        # Déterminer l'index où se trouve chaque information selon le pattern
                        if pattern == mass_patterns[0]:
                            finding["type"] = groups[0]
                            finding["size_mm"] = float(groups[2])
                            finding["location"] = groups[4] if len(groups) > 4 else "non précisée"
                        elif pattern == mass_patterns[1]:
                            finding["type"] = groups[0]
                            finding["location"] = groups[2]
                            finding["size_mm"] = float(groups[4]) if len(groups) > 4 else 0
                        else:  # pattern == mass_patterns[2]
                            finding["location"] = groups[0]
                            finding["type"] = groups[1]
                            finding["size_mm"] = float(groups[2]) if len(groups) > 2 else 0
                    
                        entities["findings"].append(finding)
        
         # Si aucune anomalie n'a été détectée avec les patterns complexes, utiliser un pattern plus simple
            if not entities["findings"]:
                # Pattern simple: recherche de taille en mm
                size_matches = re.finditer(r'(\d+)\s*mm', sections["observations"], re.IGNORECASE)
                for match in size_matches:
                    size_mm = float(match.group(1))
                    # Chercher le contexte autour de cette taille
                    context = sections["observations"][max(0, match.start() - 50):min(len(sections["observations"]), match.end() + 50)]
                
                    # Chercher le type d'anomalie dans le contexte
                    type_match = re.search(r'(masse|nodule|opacité|lésion|calcification)s?', context, re.IGNORECASE)
                    anomaly_type = type_match.group(1) if type_match else "anomalie"
                
                    # Chercher la localisation dans le contexte
                    loc_match = re.search(r'(QSE|QSI|QIE|QII|quadrant\s+\w+)', context, re.IGNORECASE)
                    location = loc_match.group(1) if loc_match else "non précisée"
                
                    entities["findings"].append({
                        "type": anomaly_type,
                        "size_mm": size_mm,
                        "location": location
                    })
    
        # Extraire les recommandations explicites
        if "conclusion" in sections:
            if re.search(r'biopsie', sections["conclusion"], re.IGNORECASE):
                entities["explicit_recommendations"].append("biopsie")
            if re.search(r'contrôle|suivi|surveillance', sections["conclusion"], re.IGNORECASE):
                entities["explicit_recommendations"].append("suivi")
            if re.search(r'échographie', sections["conclusion"], re.IGNORECASE):
                entities["explicit_recommendations"].append("échographie")
            if re.search(r'IRM', sections["conclusion"], re.IGNORECASE):
                entities["explicit_recommendations"].append("IRM")
    
        return entities
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Génère un embedding vectoriel pour un texte donné
        
        Args:
            text: Texte à encoder
            
        Returns:
            Embedding vectoriel
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Utiliser la moyenne des tokens comme représentation
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return embedding
    
    def generate_recommendations(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Génère des recommandations basées sur les entités extraites
        
        Args:
            entities: Dictionnaire des entités médicales
            
        Returns:
            Dictionnaire des recommandations
        """
        recommendations = {
            "examens_complementaires": [],
            "traitements_suggeres": [],
            "suivi_recommande": [],
            "justification": ""
        }
        
        # Si une classification BI-RADS est disponible, utiliser les guidelines correspondantes
        if entities["birads_classification"]:
            birads = entities["birads_classification"]
            if birads in self.knowledge_base["birads_guidelines"]:
                guideline = self.knowledge_base["birads_guidelines"][birads]
                
                # Recommandations d'examens
                recommendations["examens_complementaires"] = [
                    {"type": rec, "priorite": "haute" if birads in ["4C", "5"] else "moyenne"}
                    for rec in guideline["recommendations"] if "biopsie" in rec.lower() or "irm" in rec.lower() or "examen" in rec.lower()
                ]
                
                # Recommandations de suivi
                recommendations["suivi_recommande"].append({
                    "type": "Suivi selon classification BI-RADS",
                    "delai": guideline["follow_up"],
                    "details": guideline["description"]
                })
                
                # Justification
                recommendations["justification"] = (
                    f"Classification BI-RADS {birads} ({guideline['description']}) "
                    f"avec risque de malignité {guideline['malignancy_risk']}. "
                )
                
                # Si le BI-RADS est élevé (4B, 4C, 5, 6), ajouter des recommandations de traitement
                if birads in ["4B", "4C", "5", "6"]:
                    if entities["findings"] and any(finding.get("size_mm", 0) > 20 for finding in entities["findings"]):
                        stage = "locally_advanced"
                    else:
                        stage = "early_stage"
                    
                    # Ajouter une note sur les options de traitement possibles
                    if birads in ["5", "6"]:
                        treatment_options = self.knowledge_base["treatment_options"][stage]
                        for category, options in treatment_options.items():
                            if options:
                                recommendations["traitements_suggeres"].append({
                                    "categorie": category,
                                    "options": options,
                                    "note": "À discuter en réunion de concertation pluridisciplinaire"
                                })
                        
                        recommendations["justification"] += (
                            "Les options de traitement proposées sont indicatives et devront être "
                            "confirmées après évaluation histologique et discussion en RCP."
                        )
        else:
            # Si pas de BI-RADS, baser les recommandations sur les anomalies détectées
            if entities["findings"]:
                for finding in entities["findings"]:
                    # Si masse > 10mm ou caractéristiques suspectes, recommander une biopsie
                    if finding.get("size_mm", 0) > 10 or "suspect" in finding.get("type", "").lower():
                        recommendations["examens_complementaires"].append({
                            "type": "Biopsie sous guidage échographique",
                            "priorite": "moyenne",
                            "justification": f"Anomalie de {finding.get('size_mm', 'taille non précisée')} mm"
                        })
                        
                        recommendations["suivi_recommande"].append({
                            "type": "Contrôle après biopsie",
                            "delai": "1 mois après biopsie"
                        })
                        
                        recommendations["justification"] = (
                            f"Anomalie de type {finding.get('type', 'non précisé')} "
                            f"de {finding.get('size_mm', 'taille non précisée')} mm "
                            f"localisée au niveau {finding.get('location', 'localisation non précisée')}. "
                            f"Une évaluation histologique est recommandée."
                        )
                    else:
                        # Sinon, recommander un suivi
                        recommendations["suivi_recommande"].append({
                            "type": "Contrôle mammographique",
                            "delai": "6 mois"
                        })
                        
                        recommendations["justification"] = (
                            f"Anomalie de type {finding.get('type', 'non précisé')} "
                            f"de {finding.get('size_mm', 'taille non précisée')} mm "
                            f"localisée au niveau {finding.get('location', 'localisation non précisée')}. "
                            f"Les caractéristiques ne sont pas hautement suspectes, "
                            f"un suivi à court terme est recommandé."
                        )
            else:
                # Si pas d'anomalies détectées, recommander un suivi standard
                recommendations["suivi_recommande"].append({
                    "type": "Mammographie de dépistage",
                    "delai": "1 an"
                })
                
                recommendations["justification"] = (
                    "Aucune anomalie significative détectée. "
                    "Suivi standard recommandé."
                )
        
        return recommendations
    
    def process_report(self, report_text: str) -> Dict[str, Any]:
        """
        Traite un rapport médical complet et génère une analyse structurée
        
        Args:
            report_text: Texte du rapport médical
            
        Returns:
            Dictionnaire contenant l'analyse structurée et les recommandations
        """
        # Extraire les sections du rapport
        sections = self.extract_sections(report_text)
        
        # Extraire les entités médicales
        entities = self.extract_medical_entities(sections)
        
        # Générer des recommandations
        recommendations = self.generate_recommendations(entities)
        
        # Retourner l'analyse complète
        return {
            "sections": sections,
            "entities": entities,
            "recommendations": recommendations
        }
    
    def find_similar_cases(self, entities: Dict[str, Any], case_database: List[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Trouve des cas similaires dans une base de données de cas
        
        Args:
            entities: Entités extraites du cas actuel
            case_database: Liste de cas précédents
            top_n: Nombre de cas similaires à retourner
            
        Returns:
            Liste des cas les plus similaires
        """
        # Dans un système réel, vous utiliseriez une base de données de cas précédents
        # Pour cet exemple, nous supposons que case_database est une liste de dictionnaires
        
        if not case_database or not entities["findings"]:
            return []
        
        # Créer une description textuelle du cas actuel
        current_case_text = ""
        if entities["birads_classification"]:
            current_case_text += f"BI-RADS {entities['birads_classification']} "
        
        for finding in entities["findings"]:
            current_case_text += f"{finding.get('type', '')} {finding.get('size_mm', '')}mm {finding.get('location', '')} "
        
        # Générer l'embedding du cas actuel
        current_embedding = self.generate_embedding(current_case_text)
        
        # Calculer la similarité avec chaque cas de la base
        similarities = []
        for i, case in enumerate(case_database):
            case_text = f"BI-RADS {case.get('birads', '')} "
            for finding in case.get('findings', []):
                case_text += f"{finding.get('type', '')} {finding.get('size_mm', '')}mm {finding.get('location', '')} "
            
            case_embedding = self.generate_embedding(case_text)
            similarity = cosine_similarity(current_embedding, case_embedding)[0][0]
            similarities.append((i, similarity))
        
        # Trier par similarité décroissante et prendre les top_n
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in similarities[:top_n]]
        
        # Retourner les cas les plus similaires
        return [case_database[idx] for idx in top_indices]