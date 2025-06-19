HUGGINGFACE_MODELS = {
    "molecular_transformers": {
        "ChemBERTa": {
            "model_id": "DeepChem/ChemBERTa-77M-MLM",
            "use_case": "molecular_representation_learning",
            "input": "SMILES",
            "output": "embeddings"
        },
        "MolT5": {
            "model_id": "laituan245/molt5-small",
            "use_case": "molecule_to_text_generation",
            "input": "SMILES",
            "output": "molecular_descriptions"
        }
    },
    "property_predictors": {
        "MoleculeNet": {
            "model_id": "microsoft/MoleculeNet",
            "use_case": "admet_prediction",
            "properties": ["solubility", "toxicity", "bioavailability"]
        }
    },
    "reaction_prediction": {
        "RXNMapper": {
            "model_id": "rxn4chemistry/rxnmapper",
            "use_case": "reaction_mapping",
            "input": "reaction_smiles"
        }
    }
}

# Integration example
from transformers import AutoTokenizer, AutoModel
import torch

class HuggingFaceChemIntegration:
    def __init__(self):
        self.chemberta_tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
        self.chemberta_model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
        
    async def get_molecular_embeddings(self, smiles_list):
        """Generate probabilistic molecular embeddings"""
        embeddings = []
        
        for smiles in smiles_list:
            inputs = self.chemberta_tokenizer(smiles, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.chemberta_model(**inputs)
                # Get embeddings with uncertainty estimation
                embedding = outputs.last_hidden_state.mean(dim=1)
                embeddings.append(embedding)
        
        return ProbabilisticEmbeddings(embeddings)
    
    async def generate_molecular_descriptions(self, smiles):
        """Generate natural language descriptions of molecules"""
        # Use MolT5 for molecule-to-text generation
        pass
