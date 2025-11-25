from dataclasses import dataclass
from typing import List, Dict, Any
import json

@dataclass
class ExperimentResult:
    epoch: int
    iters: int
    test_loss: float
    test_acc: float

@dataclass
class Experiment:
    commit: str
    timestamp: str
    model: str
    dataset: str
    optimizer: str
    epochs: int
    results: List[ExperimentResult]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Experiment':
        """Create an Experiment from a dictionary, handling both old and new formats."""
        if 'commit' in data:
            # New format
            results = [ExperimentResult(**result) for result in data['results']]
            return cls(
                commit=data['commit'],
                timestamp=data['timestamp'],
                model=data['model'],
                dataset=data['dataset'],
                optimizer=data['optimizer'],
                epochs=data['epochs'],
                results=results
            )
        else:
            # Old format - data is a list of results
            results = [ExperimentResult(**result) for result in data]
            return cls(
                commit="unknown",
                timestamp="unknown", 
                model="unknown",
                dataset="unknown",
                optimizer="unknown",
                epochs=0,
                results=results
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Experiment to dictionary for JSON serialization."""
        return {
            'commit': self.commit,
            'timestamp': self.timestamp,
            'model': self.model,
            'dataset': self.dataset,
            'optimizer': self.optimizer,
            'epochs': self.epochs,
            'results': [
                {
                    'epoch': result.epoch,
                    'iters': result.iters,
                    'test_loss': result.test_loss,
                    'test_acc': result.test_acc
                }
                for result in self.results
            ]
        }
    
    def save(self, filename: str) -> None:
        """Save experiment to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f)
    
    @classmethod
    def load(cls, filename: str) -> 'Experiment':
        """Load experiment from JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)