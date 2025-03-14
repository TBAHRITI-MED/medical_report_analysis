# medical_report_analysis/evaluation.py

"""
Module pour l'évaluation des performances du système d'analyse et de recommandation
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import List, Dict, Any, Tuple

class MedicalReportEvaluator:
    """
    Classe pour évaluer les performances du système d'analyse et de recommandation
    """
    
    def __init__(self):
        """Initialise l'évaluateur"""
        pass
    
    def evaluate_entity_extraction(self, 
                                   predicted_entities: List[Dict[str, Any]], 
                                   ground_truth_entities: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Évalue la performance de l'extraction d'entités
        
        Args:
            predicted_entities: Entités prédites par le système
            ground_truth_entities: Entités de référence (vérité terrain)
            
        Returns:
            Dictionnaire des métriques d'évaluation
        """
        # Calculer les métriques pour différentes entités
        metrics = {}
        
        # Évaluer la classification BI-RADS
        birads_pred = [e.get("birads_classification") for e in predicted_entities]
        birads_gt = [e.get("birads_classification") for e in ground_truth_entities]
        
        valid_pairs = [(p, g) for p, g in zip(birads_pred, birads_gt) if p is not None and g is not None]
        if valid_pairs:
            pred_valid, gt_valid = zip(*valid_pairs)
            metrics["birads_accuracy"] = accuracy_score(gt_valid, pred_valid)
        else:
            metrics["birads_accuracy"] = np.nan
        
        # Évaluer la détection des anomalies
        # Cette partie est simplifiée et devrait être adaptée selon le format exact des données
        findings_count_pred = [len(e.get("findings", [])) for e in predicted_entities]
        findings_count_gt = [len(e.get("findings", [])) for e in ground_truth_entities]
        
        metrics["findings_count_diff"] = np.mean([abs(p - g) for p, g in zip(findings_count_pred, findings_count_gt)])
        
        # Évaluation plus détaillée possible sur les attributs des findings
        # (type, taille, localisation), mais cela nécessite un matching complexe des findings
        
        return metrics
    
    def evaluate_recommendations(self, 
                               predicted_recommendations: List[Dict[str, Any]], 
                               ground_truth_recommendations: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Évalue la pertinence des recommandations générées
        
        Args:
            predicted_recommendations: Recommandations générées par le système
            ground_truth_recommendations: Recommandations de référence (vérité terrain)
            
        Returns:
            Dictionnaire des métriques d'évaluation
        """
        metrics = {}
        
        # Pour simplifier, nous allons vérifier si les types d'examens recommandés correspondent
        # Dans un système réel, une évaluation plus nuancée serait nécessaire
        
        # Extraire les types d'examens recommandés
        exams_pred = []
        exams_gt = []
        
        for pred, gt in zip(predicted_recommendations, ground_truth_recommendations):
            # Examens complémentaires
            pred_exams = [e.get("type", "").lower() for e in pred.get("examens_complementaires", [])]
            gt_exams = [e.get("type", "").lower() for e in gt.get("examens_complementaires", [])]
            
            # Simplifier en vérifiant si "biopsie" est mentionné
            pred_has_biopsy = any("biopsie" in e for e in pred_exams)
            gt_has_biopsy = any("biopsie" in e for e in gt_exams)
            
            exams_pred.append(pred_has_biopsy)
            exams_gt.append(gt_has_biopsy)
        
        # Calculer les métriques de classification binaire pour la recommandation de biopsie
        metrics["biopsy_accuracy"] = accuracy_score(exams_gt, exams_pred)
        metrics["biopsy_precision"] = precision_score(exams_gt, exams_pred, zero_division=0)
        metrics["biopsy_recall"] = recall_score(exams_gt, exams_pred, zero_division=0)
        metrics["biopsy_f1"] = f1_score(exams_gt, exams_pred, zero_division=0)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(exams_gt, exams_pred).ravel()
        metrics["true_negative"] = tn
        metrics["false_positive"] = fp
        metrics["false_negative"] = fn
        metrics["true_positive"] = tp
        
        return metrics
    
    def evaluate_system(self, analyzer, test_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Évalue les performances globales du système sur un ensemble de rapports de test
        
        Args:
            analyzer: Instance de MedicalReportAnalyzer
            test_reports: Liste de rapports de test avec vérité terrain
            
        Returns:
            Dictionnaire des métriques d'évaluation
        """
        predictions = []
        
        # Générer les prédictions
        for report in test_reports:
            result = analyzer.process_report(report["text"])
            predictions.append(result)
        
        # Évaluer l'extraction d'entités
        entity_metrics = self.evaluate_entity_extraction(
            [p["entities"] for p in predictions],
            [r.get("ground_truth", {}).get("entities", {}) for r in test_reports]
        )
        
        # Évaluer les recommandations
        recommendation_metrics = self.evaluate_recommendations(
            [p["recommendations"] for p in predictions],
            [r.get("ground_truth", {}).get("recommendations", {}) for r in test_reports]
        )
        
        # Combiner les métriques
        return {
            "entity_extraction": entity_metrics,
            "recommendations": recommendation_metrics,
            "num_reports": len(test_reports)
        }
