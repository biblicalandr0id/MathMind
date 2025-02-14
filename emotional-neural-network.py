import numpy as np

class EmotionalNeuralNetwork:
    def __init__(self, learning_rate=0.01):
        # Initialize network layers
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
        
        self.interaction_weights = np.random.randn(3, 3)  # How traits influence each other
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def calculate_curiosity(self, inputs):
        """
        inputs: dict with keys 'novelty', 'uncertainty', 'knowledge', 'fear'
        """
        numerator = (inputs['novelty'] * self.curiosity_weights['novelty'] * 
                    inputs['uncertainty'] * self.curiosity_weights['uncertainty'] * 
                    inputs['knowledge'] * self.curiosity_weights['knowledge'])
        denominator = max(0.1, inputs['fear'] * self.curiosity_weights['fear'])
        return self.sigmoid(numerator / denominator)
    
    def calculate_courage(self, inputs):
        """
        inputs: dict with keys 'willingness', 'values', 'hope', 'risk', 'fear'
        """
        numerator = (inputs['willingness'] * self.courage_weights['willingness'] * 
                    inputs['values'] * self.courage_weights['values'] * 
                    inputs['hope'] * self.courage_weights['hope'])
        denominator = max(0.1, inputs['risk'] * self.courage_weights['risk'] * 
                         inputs['fear'] * self.courage_weights['fear'])
        return self.sigmoid(numerator / denominator)
    
    def calculate_kindness(self, inputs):
        """
        inputs: dict with keys 'empathy', 'action', 'patience', 'self_focus', 'depletion'
        """
        numerator = (inputs['empathy'] * self.kindness_weights['empathy'] * 
                    inputs['action'] * self.kindness_weights['action'] * 
                    inputs['patience'] * self.kindness_weights['patience'])
        denominator = max(0.1, inputs['self_focus'] * self.kindness_weights['self_focus'] + 
                         inputs['depletion'] * self.kindness_weights['depletion'])
        return self.sigmoid(numerator / denominator)
    
    def forward(self, curiosity_inputs, courage_inputs, kindness_inputs):
        # Calculate base values
        curiosity = self.calculate_curiosity(curiosity_inputs)
        courage = self.calculate_courage(courage_inputs)
        kindness = self.calculate_kindness(kindness_inputs)
        
        # Create initial state vector
        state = np.array([curiosity, courage, kindness])
        
        # Apply interactions between traits
        final_state = self.sigmoid(np.dot(self.interaction_weights, state))
        
        return {
            'curiosity': final_state[0],
            'courage': final_state[1],
            'kindness': final_state[2]
        }
    
    def train(self, inputs, target_outputs, epochs=100):
        """
        Basic training loop using gradient descent
        """
        for _ in range(epochs):
            # Forward pass
            outputs = self.forward(**inputs)
            
            # Calculate error
            errors = {k: target_outputs[k] - v for k, v in outputs.items()}
            
            # Update weights (simplified gradient descent)
            for weight_set in [self.curiosity_weights, self.courage_weights, self.kindness_weights]:
                for key in weight_set:
                    weight_set[key] += self.learning_rate * np.mean(list(errors.values()))
            
            self.interaction_weights += self.learning_rate * np.mean(list(errors.values()))

    def get_trait_relationships(self):
        """
        Analyze how traits influence each other based on interaction weights
        """
        traits = ['Curiosity', 'Courage', 'Kindness']
        relationships = []
        
        for i, source in enumerate(traits):
            for j, target in enumerate(traits):
                if i != j:
                    strength = self.interaction_weights[i][j]
                    relationships.append(f"{source} influences {target}: {strength:.2f}")
        
        return relationships