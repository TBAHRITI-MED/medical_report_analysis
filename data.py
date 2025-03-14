# medical_report_analysis/data.py

"""
Module pour la gestion des données d'exemple et de test
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any

class MedicalReportDataset:
    """
    Classe pour gérer les données d'exemples et de test pour les rapports médicaux
    """
    
    def __init__(self, data_dir="./data"):
        """
        Initialise le gestionnaire de dataset
        
        Args:
            data_dir: Répertoire contenant les données
        """
        self.data_dir = data_dir
        self.ensure_data_dir_exists()
    
    def ensure_data_dir_exists(self):
        """Crée le répertoire de données s'il n'existe pas"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def save_example_reports(self):
        """
        Sauvegarde des exemples de rapports pour le test et la démonstration
        """
        example_reports = [
            {
                "id": "example001",
                "text": """
Date d'examen: 10/02/2024
Patiente: Femme, 54 ans
Type d'examen: Mammographie bilatérale

OBSERVATIONS:
Densité mammaire de type C (hétérogène).
Sein droit: Présence d'une opacité nodulaire dans le QSE, mesurant 12 mm, 
à contours mal définis. Pas de microcalcifications associées.
Sein gauche: Aucune anomalie décelée.
Absence d'adénopathie axillaire suspecte.

IMPRESSION:
Masse suspecte dans le sein droit, classification BI-RADS 4A.
Une biopsie est recommandée pour caractérisation histologique.

CONCLUSION:
Classification BI-RADS 4A (suspicion faible de malignité)
Une biopsie sous guidage échographique est conseillée.
Contrôle rapproché recommandé après biopsie.
""",
                "metadata": {
                    "birads": "4A",
                    "patient_age": 54,
                    "patient_gender": "F"
                }
            },
            {
                "id": "example002",
                "text": """
Date d'examen: 15/03/2024
Patiente: Femme, 48 ans
Type d'examen: Mammographie bilatérale

OBSERVATIONS:
Densité mammaire de type B (fibroglandulaire dispersé).
Sein droit: Présence d'une opacité ronde bien circonscrite dans le quadrant inféro-externe, 
mesurant 8 mm. Pas de microcalcifications associées.
Sein gauche: Aucune anomalie décelée.
Absence d'adénopathie axillaire suspecte.

IMPRESSION:
Aspect probablement bénin dans le sein droit, classification BI-RADS 3.
Une surveillance à court terme est recommandée.

CONCLUSION:
Classification BI-RADS 3 (probablement bénin)
Contrôle mammographique dans 6 mois conseillé.
""",
                "metadata": {
                    "birads": "3",
                    "patient_age": 48,
                    "patient_gender": "F"
                }
            },
            {
                "id": "example003",
                "text": """
Date d'examen: 22/01/2024
Patiente: Femme, 62 ans
Type d'examen: Mammographie bilatérale

OBSERVATIONS:
Densité mammaire de type C (hétérogène).
Sein gauche: Présence d'une masse spiculée dans le quadrant supéro-externe, 
mesurant 22 mm, avec des microcalcifications polymorphes associées.
Sein droit: Aucune anomalie décelée.
Présence d'une adénopathie axillaire gauche suspecte.

IMPRESSION:
Masse hautement suspecte dans le sein gauche, classification BI-RADS 5.
Une biopsie est nécessaire.

CONCLUSION:
Classification BI-RADS 5 (haute suspicion de malignité)
Une biopsie sous guidage échographique est nécessaire en urgence.
IRM mammaire recommandée pour bilan d'extension.
Évaluation des ganglions axillaires par échographie.
""",
                "metadata": {
                    "birads": "5",
                    "patient_age": 62,
                    "patient_gender": "F"
                }
            },
            {
                "id": "example004",
                "text": """
Date d'examen: 05/04/2024
Patiente: Femme, 35 ans
Type d'examen: Mammographie bilatérale

OBSERVATIONS:
Densité mammaire de type D (extrêmement dense).
Sein droit: Pas d'anomalie visible.
Sein gauche: Pas d'anomalie visible.
Absence d'adénopathie axillaire suspecte.

IMPRESSION:
Examen normal, classification BI-RADS 1.
Toutefois, sensibilité limitée en raison de la densité mammaire importante.

CONCLUSION:
Classification BI-RADS 1 (négatif)
Contrôle mammographique annuel recommandé.
Échographie complémentaire à considérer en raison de la densité mammaire.
""",
                "metadata": {
                    "birads": "1",
                    "patient_age": 35,
                    "patient_gender": "F"
                }
            },
            {
                "id": "example005",
                "text": """
Date d'examen: 18/02/2024
Patient: Homme, 65 ans
Type d'examen: Mammographie unilatérale (sein gauche)

OBSERVATIONS:
Sein gauche: Présence d'une masse subcentimétrique rétroaréolaire, mesurant 8 mm,
à contours irréguliers, avec épaississement cutané en regard.
Pas de microcalcifications.
Présence d'une adénopathie axillaire gauche suspecte.

IMPRESSION:
Masse suspecte dans le sein gauche, évocatrice d'un cancer du sein chez l'homme.
Classification BI-RADS 4C.

CONCLUSION:
Classification BI-RADS 4C (suspicion élevée de malignité)
Biopsie sous guidage échographique nécessaire en priorité.
Bilan d'extension à prévoir.
""",
                "metadata": {
                    "birads": "4C",
                    "patient_age": 65,
                    "patient_gender": "M"
                }
            }
        ]
        
        # Sauvegarder les exemples individuellement
        for report in example_reports:
            filename = os.path.join(self.data_dir, f"{report['id']}.json")
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Sauvegarder la collection complète
        with open(os.path.join(self.data_dir, "example_reports.json"), 'w', encoding='utf-8') as f:
            json.dump(example_reports, f, indent=2, ensure_ascii=False)
        
        print(f"Sauvegardé {len(example_reports)} rapports d'exemple dans {self.data_dir}")
    
    def load_example_reports(self) -> List[Dict[str, Any]]:
        """
        Charge les exemples de rapports
        
        Returns:
            Liste des rapports d'exemple
        """
        collection_path = os.path.join(self.data_dir, "example_reports.json")
        
        # Si le fichier de collection n'existe pas, créer les exemples
        if not os.path.exists(collection_path):
            self.save_example_reports()
        
        # Charger et retourner les exemples
        with open(collection_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_sample_case_database(self):
        """
        Crée et sauvegarde une base de données échantillon de cas médicaux
        pour les démonstrations de recherche de cas similaires
        """
        sample_cases = [
            {
                "id": "case001",
                "birads": "4A",
                "findings": [
                    {
                        "type": "opacité nodulaire",
                        "size_mm": 12,
                        "location": "QSE sein droit"
                    }
                ],
                "treatment": "Biopsie sous guidage échographique",
                "result": "Fibroadénome",
                "follow_up": "Surveillance à 6 mois"
            },
            {
                "id": "case002",
                "birads": "4A",
                "findings": [
                    {
                        "type": "masse",
                        "size_mm": 14,
                        "location": "QSE sein droit"
                    }
                ],
                "treatment": "Biopsie sous guidage échographique",
                "result": "Carcinome canalaire in situ",
                "follow_up": "Chirurgie conservatrice + radiothérapie"
            },
            {
                "id": "case003",
                "birads": "3",
                "findings": [
                    {
                        "type": "opacité ronde",
                        "size_mm": 8,
                        "location": "QIE sein droit"
                    }
                ],
                "treatment": "Surveillance",
                "result": "Stable",
                "follow_up": "Contrôle à 6 mois sans changement"
            },
            {
                "id": "case004",
                "birads": "5",
                "findings": [
                    {
                        "type": "masse spiculée",
                        "size_mm": 22,
                        "location": "QSE sein gauche"
                    },
                    {
                        "type": "adénopathie axillaire",
                        "size_mm": 15,
                        "location": "aisselle gauche"
                    }
                ],
                "treatment": "Biopsie + IRM",
                "result": "Carcinome canalaire infiltrant",
                "follow_up": "Chimiothérapie néoadjuvante puis chirurgie"
            },
            {
                "id": "case005",
                "birads": "4C",
                "findings": [
                    {
                        "type": "masse irrégulière",
                        "size_mm": 8,
                        "location": "rétroaréolaire sein gauche"
                    },
                    {
                        "type": "adénopathie",
                        "size_mm": 12,
                        "location": "aisselle gauche"
                    }
                ],
                "treatment": "Biopsie",
                "result": "Carcinome canalaire infiltrant",
                "follow_up": "Mastectomie + curage axillaire + chimiothérapie",
                "notes": "Patient masculin"
            }
        ]
        
        # Sauvegarder la base de cas
        with open(os.path.join(self.data_dir, "sample_case_database.json"), 'w', encoding='utf-8') as f:
            json.dump(sample_cases, f, indent=2, ensure_ascii=False)
        
        print(f"Sauvegardé {len(sample_cases)} cas dans la base de données échantillon")
    
    def load_sample_case_database(self) -> List[Dict[str, Any]]:
        """
        Charge la base de données échantillon de cas médicaux
        
        Returns:
            Liste des cas médicaux
        """
        db_path = os.path.join(self.data_dir, "sample_case_database.json")
        
        # Si le fichier n'existe pas, créer la base d'échantillons
        if not os.path.exists(db_path):
            self.save_sample_case_database()
        
        # Charger et retourner la base de cas
        with open(db_path, 'r', encoding='utf-8') as f:
            return json.load(f)


