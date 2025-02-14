import numpy as np
import json
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F

class SensoryProcessor(nn.Module):
    def __init__(self, input_dims):
        super(SensoryProcessor, self).__init__()
        
        # Sensory processing layers
        self.visual = nn.Sequential(
            nn.Linear(input_dims['visual'], 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        
        self.auditory = nn.Sequential(
            nn.Linear(input_dims['auditory'], 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        
        self.tactile = nn.Sequential(
            nn.Linear(input_dims['tactile'], 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        self.olfactory = nn.Sequential(
            nn.Linear(input_dims['olfactory'], 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        self.gustatory = nn.Sequential(
            nn.Linear(input_dims['gustatory'], 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Integration layer
        self.integration = nn.Linear(512, 256)
        
    def forward(self, sensory_inputs):
        # Process each sense
        vis = self.visual(sensory_inputs['visual'])
        aud = self.auditory(sensory_inputs['auditory'])
        tac = self.tactile(sensory_inputs['tactile'])
        olf = self.olfactory(sensory_inputs['olfactory'])
        gus = self.gustatory(sensory_inputs['gustatory'])
        
        # Concatenate all processed inputs
        combined = torch.cat([vis, aud, tac, olf, gus], dim=1)
        
        # Integrate
        return F.relu(self.integration(combined))

class EmotionalCore(nn.Module):
    def __init__(self):
        super(EmotionalCore, self).__init__()
        
        self.curiosity_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.courage_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.kindness_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.interaction_layer = nn.Linear(3, 3)
        
    def forward(self, x):
        curiosity = self.curiosity_net(x)
        courage = self.courage_net(x)
        kindness = self.kindness_net(x)
        
        # Combine emotional states
        emotions = torch.cat([curiosity, courage, kindness], dim=1)
        
        # Apply interactions between emotions
        final_emotions = F.sigmoid(self.interaction_layer(emotions))
        
        return {
            'curiosity': final_emotions[:, 0],
            'courage': final_emotions[:, 1],
            'kindness': final_emotions[:, 2]
        }

class ExperienceMemory:
    def __init__(self, filepath="mind_experiences.json"):
        self.filepath = filepath
        self.memories = []
        self.load_memories()
        
    def load_memories(self):
        try:
            with open(self.filepath, 'r') as f:
                self.memories = json.load(f)
        except FileNotFoundError:
            self.save_memories()
    
    def record_experience(self, sensory_input, emotional_state, response):
        memory = {
            'timestamp': str(datetime.now()),
            'sensory_input': sensory_input,
            'emotional_state': emotional_state,
            'response': response,
            'connections': []
        }
        self.memories.append(memory)
        self.save_memories()
        return len(self.memories) - 1
    
    def save_memories(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.memories, f, indent=2)

class IntegratedMind:
    def __init__(self, sensory_dims):
        self.sensory = SensoryProcessor(sensory_dims)
        self.emotional = EmotionalCore()
        self.memory = ExperienceMemory()
        self.development_stage = 'newborn'
        self.experience_count = 0
        
        # Development thresholds
        self.stage_thresholds = {
            'newborn': 100,
            'infant': 500,
            'toddler': 1000,
            'child': 5000
        }
        
    def process_sensory_input(self, raw_input):
        """Convert raw input into tensor format for each sense"""
        processed = {}
        for sense, data in raw_input.items():
            processed[sense] = torch.tensor(data, dtype=torch.float32)
        return processed
    
    def calculate_sensory_equation(self, processed_input):
        """Implementation of S = Σ(Di × Wi × Pi) / (L × N)"""
        weights = {
            'visual': 0.3,
            'auditory': 0.25,
            'tactile': 0.2,
            'olfactory': 0.15,
            'gustatory': 0.1
        }
        
        latency = 1.0  # Baseline latency
        noise = 1.0    # Baseline noise
        
        weighted_sum = sum(
            torch.mean(processed_input[sense]) * weights[sense]
            for sense in processed_input
        )
        
        return weighted_sum / (latency * noise)
    
    def forward(self, raw_input):
        # Process sensory input
        processed_input = self.process_sensory_input(raw_input)
        sensory_value = self.calculate_sensory_equation(processed_input)
        
        # Integrate sensory information
        integrated = self.sensory(processed_input)
        
        # Process emotional response
        emotional_state = self.emotional(integrated)
        
        # Record experience
        self.experience_count += 1
        memory_index = self.memory.record_experience(
            raw_input,
            emotional_state,
            sensory_value.item()
        )
        
        # Check development progression
        self._check_development_stage()
        
        return {
            'sensory_value': sensory_value.item(),
            'emotional_state': emotional_state,
            'development_stage': self.development_stage,
            'memory_index': memory_index
        }
    
    def _check_development_stage(self):
        """Progress through development stages based on experience"""
        for stage, threshold in self.stage_thresholds.items():
            if self.experience_count <= threshold:
                self.development_stage = stage
                break
    
    def get_status(self):
        return {
            'development_stage': self.development_stage,
            'experience_count': self.experience_count,
            'memory_count': len(self.memory.memories)
        }

# Example usage
if __name__ == "__main__":
    # Define input dimensions for each sense
    sensory_dims = {
        'visual': 1024,    # e.g., 32x32 pixel image
        'auditory': 512,   # e.g., audio frequency spectrum
        'tactile': 256,    # e.g., pressure sensor array
        'olfactory': 128,  # e.g., chemical sensor array
        'gustatory': 128   # e.g., taste sensor array
    }
    
    # Create the integrated mind
    mind = IntegratedMind(sensory_dims)
    
    # Example sensory input
    sample_input = {
        'visual': np.random.randn(1024),
        'auditory': np.random.randn(512),
        'tactile': np.random.randn(256),
        'olfactory': np.random.randn(128),
        'gustatory': np.random.randn(128)
    }
    
    # Process input
    result = mind.forward(sample_input)
    
    # Check status
    status = mind.get_status()
    print(f"Development Stage: {status['development_stage']}")
    print(f"Total Experiences: {status['experience_count']}")
    print(f"Memories Stored: {status['memory_count']}")