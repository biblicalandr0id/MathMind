import numpy as np
from datetime import datetime
import json

class ExperienceMemory:
    def __init__(self, filepath="mind_experiences.json"):
        self.filepath = filepath
        self.memories = []
        try:
            with open(filepath, 'r') as f:
                self.memories = json.load(f)
        except FileNotFoundError:
            self.save_memories()
    
    def record_experience(self, stimulus, response, emotional_state):
        memory = {
            'timestamp': str(datetime.now()),
            'stimulus': stimulus,
            'response': response,
            'emotional_state': emotional_state,
            'connections': []
        }
        self.memories.append(memory)
        self.save_memories()
        return len(self.memories) - 1  # Return memory index
    
    def form_connection(self, memory_index, connected_memory_index, strength):
        if memory_index < len(self.memories):
            self.memories[memory_index]['connections'].append({
                'connected_to': connected_memory_index,
                'strength': strength
            })
            self.save_memories()
    
    def save_memories(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.memories, f, indent=2)

class EmotionalNeuralNetwork:
    def __init__(self, learning_rate=0.01):
        # Previous weights initialization remains the same
        self.curiosity_weights = {
            'novelty': np.random.randn(),
            'uncertainty': np.random.randn(),
            'knowledge': np.random.randn(),
            'fear': np.random.randn()
        }
        
        self.courage_weights = {
            'willingness': np.random.randn(),
            'values': np.random.randn(),
            'hope': np.random.randn(),
            'risk': np.random.randn(),
            'fear': np.random.randn()
        }
        
        self.kindness_weights = {
            'empathy': np.random.randn(),
            'action': np.random.randn(),
            'patience': np.random.randn(),
            'self_focus': np.random.randn(),
            'depletion': np.random.randn()
        }
        
        self.interaction_weights = np.random.randn(3, 3)
        self.learning_rate = learning_rate
        self.experience_memory = ExperienceMemory()
        
        # New: Development stages
        self.development_stage = 'newborn'
        self.experience_count = 0
        
    def process_input(self, input_text):
        """Process raw input like a developing mind"""
        # Convert input to emotional signals based on development stage
        if self.development_stage == 'newborn':
            # Newborn stage: Focus on basic patterns and emotional tones
            signals = {
                'curiosity_inputs': {
                    'novelty': len(set(input_text)) / len(input_text),  # Unique characters ratio
                    'uncertainty': 0.8,  # High uncertainty as everything is new
                    'knowledge': 0.1,    # Limited knowledge
                    'fear': 0.5 if any(c.isupper() for c in input_text) else 0.2  # React to intensity
                },
                'courage_inputs': {
                    'willingness': 0.7,
                    'values': 0.3,
                    'hope': 0.8,
                    'risk': 0.4,
                    'fear': 0.3
                },
                'kindness_inputs': {
                    'empathy': 0.6,
                    'action': 0.4,
                    'patience': 0.9,
                    'self_focus': 0.7,
                    'depletion': 0.2
                }
            }
        else:
            # More advanced stages would have different processing here
            signals = {}  # Placeholder for more advanced processing
            
        return signals
    
    def respond(self, input_text):
        """Generate a response based on current development stage"""
        signals = self.process_input(input_text)
        emotional_state = self.forward(**signals)
        
        # Record experience
        memory_index = self.experience_memory.record_experience(
            input_text,
            emotional_state,
            self.development_stage
        )
        
        # Generate response based on development stage
        if self.development_stage == 'newborn':
            # Respond with simple emotional patterns
            response = self._generate_newborn_response(emotional_state)
        else:
            response = "Complex response"  # Placeholder for more advanced stages
            
        self.experience_count += 1
        self._check_development_progression()
        
        return response, emotional_state
    
    def _generate_newborn_response(self, emotional_state):
        """Generate response patterns similar to newborn reactions"""
        curiosity = emotional_state['curiosity']
        courage = emotional_state['courage']
        kindness = emotional_state['kindness']
        
        # Simple pattern responses based on emotional states
        response_patterns = []
        
        if curiosity > 0.7:
            response_patterns.append("!")  # Excitement/interest
        if courage > 0.6:
            response_patterns.append("o")  # Open/engaged
        if kindness > 0.5:
            response_patterns.append("~")  # Gentle/warm
            
        # Combine patterns based on strength of emotions
        intensity = int((curiosity + courage + kindness) * 5)
        return "".join(response_patterns) * intensity
    
    def _check_development_progression(self):
        """Check if it's time to progress to next development stage"""
        if self.experience_count > 100 and self.development_stage == 'newborn':
            self.development_stage = 'infant'
            # This would trigger more complex processing patterns
            
    # Previous methods (sigmoid, calculate_curiosity, etc.) remain the same

    def get_development_status(self):
        return {
            'stage': self.development_stage,
            'experiences': self.experience_count,
            'memory_count': len(self.experience_memory.memories)
        }