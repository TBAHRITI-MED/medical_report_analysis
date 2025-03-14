# medical_report_analysis/app.py

import streamlit as st
import pandas as pd
import json
from main import MedicalReportAnalyzer

def main():
    st.title("Analyse de rapports médicaux et recommandations")
    
    # Initialiser l'analyseur
    analyzer = MedicalReportAnalyzer()
    
    # Interface pour l'entrée du rapport
    st.header("Rapport médical")
    report_text = st.text_area(
        "Entrez le texte du rapport médical ici:",
        height=300,
        value="""
Date d'examen: 10/02/2024
Patiente: Femme, 54 ans
Type d'examen: Mammographie bilatérale

OBSERVATIONS:
Densité mammaire de type C (hétérogène).
Sein droit: Présence d'une opacité nodulaire dans le QSE, mesurant 12 mm, 
à contours mal définis. Pas de microcalcifications associées.
Sein gauche: Aucune anomalie décelée.
Absence d'adénopathie axillaire suspecte.DXWL

IMPRESSION:
Masse suspecte dans le sein droit, classification BI-RADS 4A.
Une biopsie est recommandée pour caractérisation histologique.

CONCLUSION:
Classification BI-RADS 4A (suspicion faible de malignité)
Une biopsie sous guidage échographique est conseillée.
Contrôle rapproché recommandé après biopsie.
"""
    )
    
    if st.button("Analyser le rapport"):
        with st.spinner("Analyse en cours..."):
            # Traiter le rapport
            result = analyzer.process_report(report_text)
            
            # Afficher les sections extraites
            st.header("Sections extraites du rapport")
            for section_name, section_text in result["sections"].items():
                with st.expander(f"{section_name.replace('_', ' ').title()}"):
                    st.text(section_text)
            
            # Afficher les entités médicales
            st.header("Entités médicales identifiées")
            
            # Informations du patient
            patient_info = result["entities"]["patient_info"]
            if patient_info:
                st.subheader("Informations du patient")
                patient_df = pd.DataFrame([patient_info])
                st.dataframe(patient_df)
            
            # Classification BI-RADS
            if result["entities"]["birads_classification"]:
                st.subheader("Classification BI-RADS")
                st.info(f"BI-RADS {result['entities']['birads_classification']}")
            
            # Anomalies détectées
            if result["entities"]["findings"]:
                st.subheader("Anomalies détectées")
                findings_df = pd.DataFrame(result["entities"]["findings"])
                st.dataframe(findings_df)
            else:
                st.info("Aucune anomalie significative détectée")
            
            # Recommandations explicites dans le rapport
            if result["entities"]["explicit_recommendations"]:
                st.subheader("Recommandations mentionnées dans le rapport")
                for rec in result["entities"]["explicit_recommendations"]:
                    st.text(f"- {rec}")
            
            # Afficher les recommandations générées
            st.header("Recommandations générées")
            
            # Examens complémentaires
            if result["recommendations"]["examens_complementaires"]:
                st.subheader("Examens complémentaires recommandés")
                for exam in result["recommendations"]["examens_complementaires"]:
                    st.markdown(f"- **{exam['type']}** (Priorité: {exam.get('priorite', 'non spécifiée')})")
            else:
                st.info("Aucun examen complémentaire recommandé")
            
            # Traitements suggérés
            if result["recommendations"]["traitements_suggeres"]:
                st.subheader("Options de traitement à considérer")
                for treatment in result["recommendations"]["traitements_suggeres"]:
                    st.markdown(f"**{treatment['categorie'].title()}**:")
                    for option in treatment['options']:
                        st.markdown(f"- {option}")
                    if 'note' in treatment:
                        st.info(treatment['note'])
            
            # Suivi recommandé
            if result["recommendations"]["suivi_recommande"]:
                st.subheader("Suivi recommandé")
                for follow_up in result["recommendations"]["suivi_recommande"]:
                    st.markdown(f"- **{follow_up['type']}** ({follow_up['delai']})")
                    if 'details' in follow_up:
                        st.info(follow_up['details'])
            
            # Justification
            if result["recommendations"]["justification"]:
                st.subheader("Justification")
                st.write(result["recommendations"]["justification"])
    
    # Fonctionnalités supplémentaires
    st.sidebar.header("Fonctionnalités avancées")
    
    # Option pour charger un exemple
    example_option = st.sidebar.selectbox(
        "Charger un exemple",
        ["Exemple actuel", "BI-RADS 3 - Probablement bénin", "BI-RADS 5 - Hautement suspect"]
    )
    
    if example_option != "Exemple actuel" and st.sidebar.button("Charger cet exemple"):
        if example_option == "BI-RADS 3 - Probablement bénin":
            report_text = """
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
"""
        elif example_option == "BI-RADS 5 - Hautement suspect":
            report_text = """
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
"""
        st.rerun()
    
    # Option pour exporter l'analyse au format JSON
    if st.sidebar.button("Exporter l'analyse (JSON)"):
        if 'result' in locals():
            result_json = json.dumps(result, indent=2, ensure_ascii=False)
            st.sidebar.download_button(
                label="Télécharger JSON",
                data=result_json,
                file_name="analyse_rapport_medical.json",
                mime="application/json",
            )

if __name__ == "__main__":
    main()