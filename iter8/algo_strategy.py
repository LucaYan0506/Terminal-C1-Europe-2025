import gamelib
import random
import math
import warnings
from sys import maxsize
import json
import numpy as np


"""
Most of the algo code you write will be in this file unless you create new
modules yourself. Start by modifying the 'on_turn' function.

Advanced strategy tips: 

  - You can analyze action frames by modifying on_action_frame function

  - The GameState.map object can be manually manipulated to create hypothetical 
  board states. Though, we recommended making a copy of the map to preserve 
  the actual current map state.
"""

class AttackPredictor:
    def __init__(self):
        # Simple neural network with 1 hidden layer
        # Input features: 
        # - Enemy MP
        # - Number of enemy units spawned on left in last 3 turns
        # - Number of enemy units spawned on right in last 3 turns
        # - Number of our wall/turrets on left side
        # - Number of our wall/turrets on right side
        # - Turn number (normalized)
        # - Previous attack direction (-1=left, 1=right, 0=none or both)
        # - Enemy wall removals on left/right (recent history)
        
        # Network architecture (8 inputs, 12 hidden, 2 outputs)
        self.input_size = 9
        self.hidden_size = 12
        self.output_size = 2  # [probability_left, probability_right]
        
        # Define feature names for debugging
        self.feature_names = [
            "EnemyMP", "LeftSpawns", "RightSpawns", 
            "LeftDefense", "RightDefense", "TurnNumber",
            "PrevAttackDir", "LeftRemovals", "RightRemovals"
        ]
        
        # Initialize with small random weights
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.1
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.1
        self.b2 = np.zeros((1, self.output_size))
        
        # Tracking data
        self.left_spawn_history = [0, 0, 0]  # Last 3 turns
        self.right_spawn_history = [0, 0, 0]  # Last 3 turns
        self.left_removal_history = [0, 0]    # Last 2 turns
        self.right_removal_history = [0, 0]   # Last 2 turns
        self.previous_prediction = [0.5, 0.5]  # Initial even prediction
        self.previous_attack_direction = 0  # 0=none/both, -1=left, 1=right
        self.learning_rate = 0.2  # Increased learning rate for faster adaptation
        
        # Track validation performance
        self.correct_predictions = 0
        self.total_predictions = 0
        
        # Feature importance tracking
        self.feature_importance = np.zeros(self.input_size)

        # Prediction history (for visualization)
        self.prediction_history = {
            "left": [],
            "right": [],
            "actual_left": [],
            "actual_right": []
        }
        
        # Run pretraining to initialize the network with common patterns
        self.pretrain()
        
    def pretrain(self):
        """Pretrain the network with common attack patterns"""
        # Common patterns:
        # 1. If enemy removes walls on left, they'll likely attack left
        # 2. If enemy has low defenses on one side, they might attack that side
        # 3. If enemy has attacked from one side repeatedly, they might switch
        
        # Scenario 1: Left wall removals → Left attack
        for _ in range(10):
            # Features: enemy_mp, left_spawns, right_spawns, left_defense, right_defense, 
            #           turn_norm, prev_dir, left_removals, right_removals
            features = np.array([[
                0.5,     # Enemy MP
                0.2,     # Left spawns history
                0.0,     # Right spawns history
                0.8,     # Left defense
                0.8,     # Right defense
                0.2,     # Turn number normalized
                0.0,     # Previous direction
                0.8,     # Left removals
                0.0      # Right removals
            ]])
            
            # Feed forward
            self.last_input = features
            self.z1 = np.dot(features, self.W1) + self.b1
            self.a1 = self.sigmoid(self.z1)
            self.z2 = np.dot(self.a1, self.W2) + self.b2
            self.a2 = self.softmax(self.z2)
            
            # Left attack
            target = np.array([[0.9, 0.1]])
            
            # Backprop
            dz2 = self.a2 - target
            dW2 = np.dot(self.a1.T, dz2)
            db2 = np.sum(dz2, axis=0, keepdims=True)
            
            da1 = np.dot(dz2, self.W2.T)
            dz1 = da1 * self.a1 * (1 - self.a1)
            dW1 = np.dot(features.T, dz1)
            db1 = np.sum(dz1, axis=0, keepdims=True)
            
            # Update weights
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2
        
        # Scenario 2: Right wall removals → Right attack
        for _ in range(10):
            features = np.array([[
                0.5,     # Enemy MP
                0.0,     # Left spawns history
                0.2,     # Right spawns history
                0.8,     # Left defense
                0.8,     # Right defense
                0.2,     # Turn number normalized
                0.0,     # Previous direction
                0.0,     # Left removals
                0.8      # Right removals
            ]])
            
            # Feed forward
            self.last_input = features
            self.z1 = np.dot(features, self.W1) + self.b1
            self.a1 = self.sigmoid(self.z1)
            self.z2 = np.dot(self.a1, self.W2) + self.b2
            self.a2 = self.softmax(self.z2)
            
            # Right attack
            target = np.array([[0.1, 0.9]])
            
            # Backprop
            dz2 = self.a2 - target
            dW2 = np.dot(self.a1.T, dz2)
            db2 = np.sum(dz2, axis=0, keepdims=True)
            
            da1 = np.dot(dz2, self.W2.T)
            dz1 = da1 * self.a1 * (1 - self.a1)
            dW1 = np.dot(features.T, dz1)
            db1 = np.sum(dz1, axis=0, keepdims=True)
            
            # Update weights
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2
            
        # Scenario 3: Strong left defense and previous right attacks → Left attack
        for _ in range(10):
            features = np.array([[
                0.7,     # Enemy MP
                0.1,     # Left spawns history
                0.7,     # Right spawns history
                0.3,     # Left defense (weak)
                0.9,     # Right defense (strong)
                0.4,     # Turn number normalized
                1.0,     # Previous direction (right)
                0.1,     # Left removals
                0.1      # Right removals
            ]])
            
            # Feed forward
            self.last_input = features
            self.z1 = np.dot(features, self.W1) + self.b1
            self.a1 = self.sigmoid(self.z1)
            self.z2 = np.dot(self.a1, self.W2) + self.b2
            self.a2 = self.softmax(self.z2)
            
            # Left attack (enemy switching sides)
            target = np.array([[0.8, 0.2]])
            
            # Backprop
            dz2 = self.a2 - target
            dW2 = np.dot(self.a1.T, dz2)
            db2 = np.sum(dz2, axis=0, keepdims=True)
            
            da1 = np.dot(dz2, self.W2.T)
            dz1 = da1 * self.a1 * (1 - self.a1)
            dW1 = np.dot(features.T, dz1)
            db1 = np.sum(dz1, axis=0, keepdims=True)
            
            # Update weights
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2
        
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -15, 15)))  # Clip to avoid overflow
    
    def softmax(self, x):
        """Softmax function for output layer"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
    def forward(self, X):
        """Forward pass through the network"""
        # Store input for backpropagation
        self.last_input = X
        
        # First layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        
        return self.a2
    
    def predict_attack(self, game_state):
        """Predict whether enemy will attack from left or right"""
        # Extract features
        features = self.extract_features(game_state)
        features_array = np.array([features])
        
        # Make prediction
        prediction = self.forward(features_array)[0]
        self.previous_prediction = prediction
        
        # Store prediction in history
        self.prediction_history["left"].append(prediction[0])
        self.prediction_history["right"].append(prediction[1])
        # Actual attack will be updated in learn_from_actual
        
        # Calculate and update feature importance
        self.update_feature_importance(features)
        
        # Return the prediction and the most likely attack direction
        left_prob, right_prob = prediction
        
        attack_dirs = []
        # Use thresholds to determine if an attack is likely
        if left_prob > 0.55:
            attack_dirs.append("LEFT")
        if right_prob > 0.55:
            attack_dirs.append("RIGHT")
            
        return attack_dirs, prediction
    
    def extract_features(self, game_state):
        """Extract relevant features for the prediction"""
        # Enemy MP (normalized)
        enemy_mp = game_state.get_resource(1, 1) / 30.0  # Normalize by max expected MP
        
        # Count our defenses on left and right side
        left_defense = 0
        right_defense = 0
        
        # Define left and right regions
        left_region = [(x,y) for x in range(0,10) for y in range(8,14)]
        right_region = [(x,y) for x in range(18,28) for y in range(8,14)]
        
        for x,y in left_region:
            if game_state.contains_stationary_unit([x,y]):
                unit = game_state.game_map[x,y][0]
                if unit.player_index == 0 and unit.unit_type in [0, 2]:  # Wall or Turret
                    left_defense += 1
                    
        for x,y in right_region:
            if game_state.contains_stationary_unit([x,y]):
                unit = game_state.game_map[x,y][0]
                if unit.player_index == 0 and unit.unit_type in [0, 2]:  # Wall or Turret
                    right_defense += 1
        
        # Normalize defenses
        left_defense = min(left_defense / 20.0, 1.0)
        right_defense = min(right_defense / 20.0, 1.0)
        
        # Turn number (normalized)
        turn_normalized = min(game_state.turn_number / 30.0, 1.0)
        
        # Combine features
        left_spawns = sum(self.left_spawn_history) / 10.0  # Normalize
        right_spawns = sum(self.right_spawn_history) / 10.0  # Normalize
        left_removals = sum(self.left_removal_history) / 3.0  # Normalize
        right_removals = sum(self.right_removal_history) / 3.0  # Normalize
        
        features = [
            enemy_mp,
            left_spawns, 
            right_spawns,
            left_defense,
            right_defense,
            turn_normalized,
            self.previous_attack_direction,
            left_removals,
            right_removals
        ]
        
        return features
    
    def update_feature_importance(self, features):
        """Calculate feature importance based on weights and current feature values"""
        # Only update if we have valid weights
        if hasattr(self, 'W1') and hasattr(self, 'W2'):
            # Calculate gradient of output with respect to input
            # This is a simplified calculation for feature importance
            for i in range(len(features)):
                # Calculate importance as absolute sum of weights from this feature to each hidden neuron
                importance = np.sum(np.abs(self.W1[i, :]))
                # Multiply by the feature value to get current impact
                importance *= abs(features[i])
                # Update running average
                self.feature_importance[i] = 0.9 * self.feature_importance[i] + 0.1 * importance
    
    def print_feature_importance(self):
        """Print the most important features based on network weights"""
        if len(self.feature_importance) > 0:
            # Sort features by importance
            sorted_indices = np.argsort(-self.feature_importance)
            
            result = "Feature importance ranking:\n"
            for i in sorted_indices:
                if i < len(self.feature_names):
                    result += f"{self.feature_names[i]}: {self.feature_importance[i]:.4f}\n"
            
            return result
        return "Feature importance not yet calculated"
    
    def print_network_weights(self):
        """Print a summary of the neural network weights"""
        result = "Neural Network Weights Summary:\n"
        
        # Summary of first layer weights
        w1_avg = np.mean(self.W1)
        w1_std = np.std(self.W1) 
        w1_min = np.min(self.W1)
        w1_max = np.max(self.W1)
        
        result += f"Layer 1 weights: avg={w1_avg:.4f}, std={w1_std:.4f}, min={w1_min:.4f}, max={w1_max:.4f}\n"
        
        # Summary of second layer weights
        w2_avg = np.mean(self.W2)
        w2_std = np.std(self.W2)
        w2_min = np.min(self.W2)
        w2_max = np.max(self.W2)
        
        result += f"Layer 2 weights: avg={w2_avg:.4f}, std={w2_std:.4f}, min={w2_min:.4f}, max={w2_max:.4f}\n"
        
        # Summary of feature importance
        result += self.print_feature_importance()
        
        return result
    
    def update_spawn_history(self, left_spawns, right_spawns):
        """Update the spawn history with latest information"""
        self.left_spawn_history.pop(0)
        self.left_spawn_history.append(left_spawns)
        
        self.right_spawn_history.pop(0)
        self.right_spawn_history.append(right_spawns)
    
    def update_removal_history(self, left_removals, right_removals):
        """Update the wall removal history"""
        self.left_removal_history.pop(0)
        self.left_removal_history.append(left_removals)
        
        self.right_removal_history.pop(0)
        self.right_removal_history.append(right_removals)
    
    def learn_from_actual(self, actual_attack_dirs):
        """Update weights based on actual attack direction"""
        # Store actual attack in history
        self.prediction_history["actual_left"].append("LEFT" in actual_attack_dirs)
        self.prediction_history["actual_right"].append("RIGHT" in actual_attack_dirs)
        
        # Keep history size manageable
        max_history = 50
        if len(self.prediction_history["left"]) > max_history:
            for key in self.prediction_history:
                self.prediction_history[key] = self.prediction_history[key][-max_history:]
        
        # Convert attack direction to target output
        target = np.zeros((1, self.output_size))
        
        if "LEFT" in actual_attack_dirs and "RIGHT" in actual_attack_dirs:
            target[0] = [0.5, 0.5]  # Both sides
            self.previous_attack_direction = 0
        elif "LEFT" in actual_attack_dirs:
            target[0] = [0.9, 0.1]  # Left side
            self.previous_attack_direction = -1
        elif "RIGHT" in actual_attack_dirs:
            target[0] = [0.1, 0.9]  # Right side
            self.previous_attack_direction = 1
        else:
            target[0] = [0.5, 0.5]  # No attack
            self.previous_attack_direction = 0
            
        # Track prediction accuracy for debugging
        if hasattr(self, 'a2') and len(actual_attack_dirs) > 0:
            prediction = self.a2[0]
            predicted_dir = "LEFT" if prediction[0] > prediction[1] else "RIGHT"
            actual_dir = actual_attack_dirs[0] if len(actual_attack_dirs) == 1 else "BOTH"
            
            if (predicted_dir == actual_dir) or (predicted_dir in actual_attack_dirs and len(actual_attack_dirs) > 1):
                self.correct_predictions += 1
            
            self.total_predictions += 1
            
            if self.total_predictions % 5 == 0:
                accuracy = self.correct_predictions / self.total_predictions
                # You can use gamelib.debug_write here if you want to log the accuracy
            
        # Only learn if we had a prediction (forward pass was called)
        if hasattr(self, 'a2'):
            # Calculate gradients (simplified backpropagation)
            dz2 = self.a2 - target
            dW2 = np.dot(self.a1.T, dz2)
            db2 = np.sum(dz2, axis=0, keepdims=True)
            
            da1 = np.dot(dz2, self.W2.T)
            dz1 = da1 * self.a1 * (1 - self.a1)  # Derivative of sigmoid
            dW1 = np.dot(self.last_input.T, dz1) if hasattr(self, 'last_input') else 0
            db1 = np.sum(dz1, axis=0, keepdims=True)
            
            # Update weights with momentum (to smooth learning)
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2
            
            if hasattr(self, 'last_input'):
                self.W1 -= self.learning_rate * dW1
                self.b1 -= self.learning_rate * db1

    def visualize_predictions(self):
        """Create an ASCII visualization of recent predictions vs actual attacks"""
        if len(self.prediction_history["left"]) < 5:
            return "Not enough prediction history for visualization"
        
        result = "Recent Attack Predictions (Last 10 turns):\n"
        result += "L=Left, R=Right, l/r=weak prediction, *=actual attack\n"
        
        # Get last 10 predictions (or fewer if not available)
        history_len = min(10, len(self.prediction_history["left"]))
        
        # Draw the prediction chart
        for i in range(history_len):
            idx = len(self.prediction_history["left"]) - history_len + i
            
            left_prob = self.prediction_history["left"][idx]
            right_prob = self.prediction_history["right"][idx]
            
            # Get actual attack info if available
            actual_left = len(self.prediction_history["actual_left"]) > idx and self.prediction_history["actual_left"][idx]
            actual_right = len(self.prediction_history["actual_right"]) > idx and self.prediction_history["actual_right"][idx]
            
            # Format: Turn | LEFT prediction | RIGHT prediction
            turn_num = idx + 1  # 1-indexed turn number
            result += f"Turn {turn_num:2d} | "
            
            # Left prediction
            if left_prob > 0.7:
                result += "L"
            elif left_prob > 0.55:
                result += "l"
            else:
                result += " "
                
            if actual_left:
                result += "*"
            else:
                result += " "
                
            result += " | "
            
            # Right prediction
            if right_prob > 0.7:
                result += "R"
            elif right_prob > 0.55:
                result += "r"
            else:
                result += " "
                
            if actual_right:
                result += "*"
            else:
                result += " "
                
            result += " |\n"
            
        return result

class AlgoStrategy(gamelib.AlgoCore):
    def __init__(self):
        super().__init__()
        seed = random.randrange(maxsize)
        random.seed(seed)
        # gamelib.debug_write('Random seed: {}'.format(seed))

        self.firstLineWall = {
            'LEFT':[[0,13],[3,13],[1,12],[3,10]],
            'MID':[],
            'RIGHT':[[23,12],[22,11],[21,10],[20,9]]
        }

        for x in range(24,28):
            self.firstLineWall['RIGHT'].append([x,13])

        for x in range(8,20):
            self.firstLineWall['MID'].append([x,8])

        self.moreWalls = [[8,9],[7,10],[6,11],[5,12]]

        self.turret_locations = {
            'LEFT' : [[5,11],[6,10],[7,9], [2,11], [4,9]],
            'MID' : [],
            'RIGHT' : []
        } 

        self.support_locations = [[10,7], [17,7]]

        # add more moreSupportLocations and buiild them when SP > 10
        self.moreSupportLocations = []
        for x in range(8,20):
            self.moreSupportLocations.append([x, 7])
        
        self.enemy_append_removal_unit = set()

        self.right_kamikaze_attack_locations = [[24, 14], [25, 14], [26, 14], [27, 14], [24, 15], [25, 15], [26, 15], [24,16], [25,16]]
        self.left_kamikaze_attack_locations = [[0, 14], [1, 14], [2, 14], [3, 14], [1, 15], [2, 15], [3, 15], [2,16], [3,16]]

        self.will_kamikaze_attack = False
        self.kamikaze_attack_location = 0
        
        # Initialize the attack predictor
        self.attack_predictor = AttackPredictor()
        
        # Track enemy spawn locations
        self.enemy_spawn_locations = {
            "LEFT": [],
            "RIGHT": []
        }
        
    def on_game_start(self, config):
        """ 
        Read in config and perform any initial setup here 
        """
        # gamelib.debug_write('Configuring your custom algo strategy...')
        self.config = config
        global WALL, SUPPORT, TURRET, SCOUT, DEMOLISHER, INTERCEPTOR, MP, SP, LEFT_KAMIKAZE, RIGHT_KAMIKAZE
        WALL = config["unitInformation"][0]["shorthand"]
        SUPPORT = config["unitInformation"][1]["shorthand"]
        TURRET = config["unitInformation"][2]["shorthand"]
        SCOUT = config["unitInformation"][3]["shorthand"]
        DEMOLISHER = config["unitInformation"][4]["shorthand"]
        INTERCEPTOR = config["unitInformation"][5]["shorthand"]
        MP = 1
        SP = 0
        # This is a good place to do initial setup
        LEFT_KAMIKAZE = 1
        RIGHT_KAMIKAZE = 2
        self.scored_on_locations = []
        
        # Run a test of the attack predictor
        self.test_attack_predictor()
        
    def test_attack_predictor(self):
        """Test the attack predictor with some simulated scenarios"""
        gamelib.debug_write("Testing Attack Predictor Neural Network...")
        
        # Create dummy features for testing
        test_scenarios = [
            # enemy_mp, left_spawns, right_spawns, left_defense, right_defense, turn_norm, prev_dir, left_removals, right_removals
            [0.7, 0.0, 0.0, 0.8, 0.8, 0.1, 0.0, 0.9, 0.0],  # Left wall removals → Should predict LEFT
            [0.7, 0.0, 0.0, 0.8, 0.8, 0.1, 0.0, 0.0, 0.9],  # Right wall removals → Should predict RIGHT
            [0.9, 0.8, 0.0, 0.8, 0.8, 0.3, -1.0, 0.0, 0.0], # Previous left attacks → Should predict LEFT
            [0.9, 0.0, 0.8, 0.8, 0.8, 0.3, 1.0, 0.0, 0.0],  # Previous right attacks → Should predict RIGHT
            [0.9, 0.0, 0.0, 0.3, 0.9, 0.5, 0.0, 0.0, 0.0],  # Weak left defense → Should predict LEFT
            [0.9, 0.0, 0.0, 0.9, 0.3, 0.5, 0.0, 0.0, 0.0]   # Weak right defense → Should predict RIGHT
        ]
        
        expected_sides = ["LEFT", "RIGHT", "LEFT", "RIGHT", "LEFT", "RIGHT"]
        
        # Run tests
        for i, features in enumerate(test_scenarios):
            # Create a feature array
            features_array = np.array([features])
            
            # Get prediction
            prediction = self.attack_predictor.forward(features_array)[0]
            
            # Determine predicted direction
            left_prob, right_prob = prediction
            predicted_side = "LEFT" if left_prob > right_prob else "RIGHT"
            
            # Check if prediction matches expectation
            result = "✓" if predicted_side == expected_sides[i] else "✗"
            
            gamelib.debug_write(f"Test {i+1}: Expected {expected_sides[i]}, Predicted {predicted_side} ({left_prob:.2f}, {right_prob:.2f}) {result}")
        
        gamelib.debug_write("Attack predictor test complete.")

    def on_turn(self, turn_state):
        """
        This function is called every turn with the game state wrapper as
        an argument. The wrapper stores the state of the arena and has methods
        for querying its state, allocating your current resources as planned
        unit deployments, and transmitting your intended deployments to the
        game engine.
        """
        game_state = gamelib.GameState(self.config, turn_state)
        # gamelib.debug_write('Performing turn {} of your custom algo strategy'.format(game_state.turn_number))
        game_state.suppress_warnings(True)  #Comment or remove this line to enable warnings.

        # Predict enemy attack pattern using neural network
        predicted_attack_directions, probabilities = self.predict_enemy_attack(game_state)
        gamelib.debug_write(f"Turn {game_state.turn_number}: Predicted attack from {predicted_attack_directions}")
        gamelib.debug_write(f"Probabilities: LEFT={probabilities[0]:.2f}, RIGHT={probabilities[1]:.2f}")
        
        # Display visualization every 5 turns
        if game_state.turn_number >= 5 and game_state.turn_number % 5 == 0:
            vis = self.attack_predictor.visualize_predictions()
            gamelib.debug_write(vis)

        if game_state.turn_number < 1:
            self.early_game(game_state)
        else:
            self.mid_game(game_state, predicted_attack_directions)

        game_state.submit_turn()


    """
    NOTE: All the methods after this point are part of the sample starter-algo
    strategy and can safely be replaced for your custom algo.
    """
    def early_game(self,game_state):
        self.build_wall(game_state)
        self.build_tower(game_state)
        self.build_support(game_state)
        game_state.attempt_spawn(SCOUT, [14, 0], 5)

    def mid_game(self, game_state, predicted_attack_directions=[]):
        """
        For defense we will use a spread out layout and some interceptors early on.
        We will place turrets near locations the opponent managed to score on.
        For offense we will use long range demolishers if they place stationary units near the enemy's front.
        If there are no stationary units to attack in the front, we will send Scouts to try and score quickly.
        """
        if not self.firstLineWallIsFull(game_state):
            self.will_kamikaze_attack = False
            self.kamikaze_attack_location = 0

        if self.will_kamikaze_attack:
            if self.kamikaze_attack_location == LEFT_KAMIKAZE:
                game_state.attempt_spawn(WALL, [2,13])
            if self.kamikaze_attack_location == RIGHT_KAMIKAZE:
                game_state.attempt_spawn(WALL, [25,13])
            
            game_state.attempt_spawn(WALL, [4,12])
            self.kamikaze_attack(game_state)
            self.will_kamikaze_attack = False
            self.kamikaze_attack_location = 0
        else:
            if game_state.game_map[4,12] and game_state.game_map[4,12][0].unit_type == WALL:
                game_state.attempt_remove([4,12])

            game_state.attempt_spawn(TURRET, [[26,12]])

            # If neural network didn't provide predictions, use the old method
            if not predicted_attack_directions:
                predicted_attack_directions = self.check_potential_enemy_attack(game_state)
                
            self.build_tower(game_state, predicted_attack_directions)
            self.build_wall(game_state, predicted_attack_directions)
            self.build_support(game_state)
            game_state.attempt_spawn(WALL, self.moreWalls)
            game_state.attempt_upgrade(self.moreWalls)


            self.update_tower(game_state, predicted_attack_directions)
            self.update_wall(game_state, predicted_attack_directions)
            self.update_support(game_state)

            self.spawnInterceptor(game_state, predicted_attack_directions)
        
        if game_state.get_resource(MP) > 10 and self.firstLineWallIsFull(game_state):
            self.will_kamikaze_attack = True
            self.kamikaze_attack_location = self.plan_kamikaze_attack(game_state)
        else:
            # replace wall only when we are not plannign to attack
            self.replaceWall(game_state)

        for x, y in self.moreSupportLocations:
            game_state.attempt_spawn(SUPPORT, [x,y])
            game_state.attempt_upgrade([x,y])

    def predict_enemy_attack(self, game_state):
        """
        Use the neural network to predict enemy attack direction based on game state
        """
        # Get prediction from neural network
        predicted_directions, probabilities = self.attack_predictor.predict_attack(game_state)
        
        # Periodically log neural network information (every 15 turns)
        if game_state.turn_number > 0 and game_state.turn_number % 15 == 0:
            network_summary = self.attack_predictor.print_network_weights()
            gamelib.debug_write(f"Turn {game_state.turn_number} Neural Network Status:\n{network_summary}")
        
        # If it's early in the game and we have no strong predictions, use traditional method
        if game_state.turn_number < 5 and not predicted_directions:
            traditional_prediction = self.check_potential_enemy_attack(game_state)
            if traditional_prediction:
                return traditional_prediction, [0.8 if "LEFT" in traditional_prediction else 0.2, 
                                               0.8 if "RIGHT" in traditional_prediction else 0.2]
        
        return predicted_directions, probabilities

    def build_tower(self, game_state, predicted_attack_directions = []):
        # build turret where we predict an attack to happen first
        for dir in predicted_attack_directions:
            game_state.attempt_spawn(TURRET, self.turret_locations[dir])

        # build the rest
        for dir in self.turret_locations.keys():
            if dir not in predicted_attack_directions:
                game_state.attempt_spawn(TURRET, self.turret_locations[dir])

    def firstLineWallIsFull(self, game_state):
        walls = [[x,8] for x in range(8,20)] + [[20,9],[21,10],[22,11],[23,12],[24,13]]
        for [x,y] in walls:
            if len(game_state.game_map[x,y]) == 0:
                return False
        return True

    def build_support(self, game_state):
        # Build supports in positions that shield our mobile units
        game_state.attempt_spawn(SUPPORT, self.support_locations)

    def build_wall(self, game_state, predicted_attack_directions = []):
        # if top right wall are low hp, start build tower at edge (potentially, enemy will attack to this side)
        if game_state.turn_number != 0 and self.top_right_wall_weak(game_state):
            game_state.attempt_spawn(TURRET, [[22,11],[21,10],[20,9],[25,11],[23,9]])
            self.turret_locations['RIGHT'] += [[22,11],[21,10],[20,9],[25,11],[23,9]]
            
            self.removeWall(game_state,[[20,9],[21,10]])

            newWall = [[24,10],[26,12]]
            for wall in newWall:
                if wall not in self.firstLineWall['RIGHT']:
                    self.firstLineWall['RIGHT'].append(wall)
            
            newWall = [[22,12],[23,12]]
            for wall in newWall:
                if wall not in self.moreWalls:
                    self.moreWalls.append(wall)


        # build turret where we predict an attack to happen first
        for dir in predicted_attack_directions:
            game_state.attempt_spawn(WALL, self.firstLineWall[dir])

        # build the rest
        for dir in self.turret_locations.keys():
            if dir not in predicted_attack_directions:
                game_state.attempt_spawn(WALL, self.firstLineWall[dir])

    def removeWall(self,game_state, walls):
            for wall in walls:
                if len(game_state.game_map[wall[0],wall[1]]) > 0 and game_state.game_map[wall[0],wall[1]][0].unit_type == WALL:
                    game_state.attempt_remove(wall)
                for dir in self.firstLineWall.keys():
                    if wall in self.firstLineWall[dir]:
                        self.firstLineWall[dir].remove(wall)

    def top_right_wall_weak(self, game_state):
        top_right_wall = [[24,13],[23,12],[22,11]]
        for [x,y] in top_right_wall:
            if len(game_state.game_map[x, y]) == 0:
                return True
            unit = game_state.game_map[x, y][0]
            currHP = unit.health
            originHP = unit.max_health
            if currHP / originHP < 0.6:
                return True
            
        return False

    def update_wall(self, game_state, predicted_attack_directions = []):
        # upgrade wall where we predict an attack to happen first
        for dir in predicted_attack_directions:
            game_state.attempt_upgrade(self.firstLineWall[dir])

        # upgrade the rest
        for dir in self.firstLineWall.keys():
            if dir not in predicted_attack_directions:
                game_state.attempt_upgrade(self.firstLineWall[dir])

    def update_support(self, game_state):
        game_state.attempt_upgrade(self.support_locations)
    
    def update_tower(self, game_state, predicted_attack_directions = []):
        # upgrade turret where we predict an attack to happen first
        for dir in predicted_attack_directions:
            game_state.attempt_upgrade(self.turret_locations[dir])

        # upgrade the rest
        for dir in self.turret_locations.keys():
            if dir not in predicted_attack_directions:
                game_state.attempt_upgrade(self.turret_locations[dir])

    def spawnInterceptor(self, game_state, predicted_attack_directions):
        enemyMP = game_state.get_resource(MP,1)
        # gamelib.debug_write('enemyMP {}'.format(enemyMP))
        base = 7.0

        enemy_support_counter = min(self.count_enemy_support(game_state),6)
        
        gamelib.debug_write("Turn: {}".format(game_state.turn_number))
        gamelib.debug_write("support: {}".format(self.count_enemy_support(game_state)))
        interceptor_num = 0
        if enemyMP >= base + 5.0:
            interceptor_num = 5 + int(enemy_support_counter / 3)
        elif enemyMP >= base + 3.0:
            interceptor_num = 4 + int(enemy_support_counter / 3)
        elif enemyMP >= base:
            interceptor_num = 2 + int(enemy_support_counter / 3)
        elif enemy_support_counter > 0 and enemyMP >= 5:
            interceptor_num = 1 + int(enemy_support_counter / 3)

        # if nothing is predicted, spawn both side
        if len(predicted_attack_directions) == 0:
            predicted_attack_directions = ["LEFT","RIGHT"]

        interceptor_num = interceptor_num // len(predicted_attack_directions)
        for dir in predicted_attack_directions:
            if dir == "LEFT":
                game_state.attempt_spawn(INTERCEPTOR, [5, 8], interceptor_num)
            elif dir == "RIGHT":
                game_state.attempt_spawn(INTERCEPTOR, [22, 8], interceptor_num)

        if self.top_right_wall_weak(game_state):
                game_state.attempt_spawn(INTERCEPTOR, [22, 8], int(enemy_support_counter / 3))

    def replaceWall(self, game_state):    
        allWalls = self.moreWalls 
        for dir in self.firstLineWall.keys():
            allWalls += self.firstLineWall[dir]

        for [x,y] in allWalls:
            tile = game_state.game_map[x, y]
            if len(tile) > 0:
                tile = tile[0]
                if tile.unit_type != WALL:
                    continue
            else:
                continue
            if tile.health < 40:
                game_state.attempt_remove([x,y])
    
    def ideal_exit(self, game_state, final_position):
        """Check if the left-side diagonal path [0,14] to [13,27] is clear (no wall or turret)."""
        
        for i in range(6):
            x = i
            y = 14 + i
            if [x,y] == final_position:
                return True
        return False

    def is_path_clear_of_enemy_walls(self, game_state, path):
        for loc in path:
            # gamelib.debug_write("path: {}".format(loc))
            
            units = game_state.game_map[loc[0], loc[1]]
            for unit in units:
                if unit.player_index == 1 and unit.unit_type in [WALL, TURRET]:
                    return False
        return True

    def count_enemy_support(self, game_state):
        counter = 0
        for i in range(14):
            y = 14 + i
            for x in range(i,28 - i):
                if len(game_state.game_map[x,y]) > 0 and game_state.game_map[x,y][0].unit_type == SUPPORT:
                    counter += 1

        return counter

    def check_potential_enemy_attack(self, game_state):
        predicted_attack_directions = []
        if (1,14) in self.enemy_append_removal_unit or (2,14) in self.enemy_append_removal_unit:
            game_state.attempt_remove([[1,13],[2,13]])
            self.removeWall(game_state, [[1,13],[2,13]])
            predicted_attack_directions.append("LEFT")

        if (25,14) in self.enemy_append_removal_unit or (26,14) in self.enemy_append_removal_unit:
            game_state.attempt_remove([[25,13],[26,13]])
            self.removeWall(game_state, [[25,13],[26,13]])
            predicted_attack_directions.append("RIGHT")

        return predicted_attack_directions
    
    def detect_enemy_trapped(self, game_state):
        """
        Detect if the enemy has no position to send units from and no pending wall removals
        Returns True if enemy is detected to be trapped, False otherwise
        """
        # Check if the enemy has any border positions without units (to spawn from)
        has_valid_path = False
        all_edges = game_state.game_map.get_edges()
        left_edge = game_state.game_map.get_edge_locations(game_state.game_map.TOP_LEFT)
        right_edge = game_state.game_map.get_edge_locations(game_state.game_map.TOP_RIGHT)
        enemy_points = left_edge + right_edge

        for enemy_point in enemy_points:
            path = game_state.find_path_to_edge(enemy_point)
            if path and len(path) > 1:
                final_position = path[-1]
                for edge_list in all_edges:
                    if final_position in edge_list:
                        has_valid_path = True
                        break

       
        # Check if enemy has any walls set for removal
        has_pending_removal = False
        
        # Check both sides of the map for pending removals
        if len(self.enemy_append_removal_unit) > 0:
            has_pending_removal = True
        
        # Also check locations right at the halfway point for our side
        # for x in range(game_state.game_map.ARENA_SIZE):
        #     if game_state.contains_stationary_unit([x, 13]):
        #         unit = game_state.game_map[x, 13][0]
        #         if unit.player_index == 0 and unit.pending_removal:
        #             has_pending_removal = True
        #             break
        
        # Enemy is trapped if they have no spawn positions and no pending removals
        is_trapped = not has_valid_path and not has_pending_removal
        if is_trapped:
            gamelib.debug_write("Enemy detected as trapped! No open spawn positions and no pending wall removals.")
        return is_trapped
    
    def execute_finishing_attack(self, game_state):
        """
        If enemy is trapped with no way to spawn or remove units, execute a heavy attack
        to finish them off
        """
        # Build up a heavy attack force with demolishers for structures
        mp = game_state.get_resource(MP)
        
        # First ensure our own defenses are solid
        self.build_wall(game_state)
        self.build_tower(game_state)
        
        # Launch a coordinated attack with a mix of units
        demolisher_count = min(5, mp // game_state.type_cost(DEMOLISHER)[MP])
        mp -= demolisher_count * game_state.type_cost(DEMOLISHER)[MP]
        
        # Send demolishers from both sides to destroy their structures
        if demolisher_count >= 2:
            left_demo = demolisher_count // 2
            right_demo = demolisher_count - left_demo
            
            # Left side demolishers
            if left_demo > 0:
                game_state.attempt_spawn(DEMOLISHER, [3, 10], left_demo)
                
            # Right side demolishers
            if right_demo > 0:
                game_state.attempt_spawn(DEMOLISHER, [24, 10], right_demo)
        else:
            # If only 1 demolisher, send it down the middle
            game_state.attempt_spawn(DEMOLISHER, [14, 0], demolisher_count)
        
        # Use remaining MP for scouts
        scout_count = mp // game_state.type_cost(SCOUT)[MP]
        if scout_count > 0:
            game_state.attempt_spawn(SCOUT, [14, 0], scout_count)


    def on_action_frame(self, turn_string):
        frame_data = json.loads(turn_string)
        
        # Track enemy unit spawns for the predictor
        left_spawns = 0
        right_spawns = 0
        
        # Track wall removals for each side
        left_removals = 0
        right_removals = 0
        
        # Define the left and right regions of enemy territory
        left_region_x = range(0, 14)
        right_region_x = range(14, 28)
        
        # Process enemy mobile units
        mobile_units = [SCOUT, DEMOLISHER, INTERCEPTOR]
        for unit_type in range(len(mobile_units)):
            # The unit types in the frame data are indexed as strings
            # SCOUT is "3", DEMOLISHER is "4", INTERCEPTOR is "5"
            unit_type_index = str(unit_type + 3)  # Convert to string index
            
            if unit_type_index in frame_data["p2Units"]:
                for unit in frame_data["p2Units"][unit_type_index]:
                    x, y = unit[0], unit[1]
                    
                    # Only count units just spawned (near the edge)
                    if y >= 14:
                        if x in left_region_x:
                            left_spawns += 1
                        else:
                            right_spawns += 1
        
        # Track wall removals for prediction
        if "6" in frame_data["p2Units"]:  # Unit type 6 is for removals
            for unit in frame_data["p2Units"]["6"]:
                x = unit[0]
                y = unit[1]
                self.enemy_append_removal_unit.add((x,y))
                
                # Count removals by side
                if x < 14:
                    left_removals += 1
                else:
                    right_removals += 1
        
        # Update the attack predictor with the spawns and removals we saw
        self.attack_predictor.update_spawn_history(left_spawns, right_spawns)
        self.attack_predictor.update_removal_history(left_removals, right_removals)
        
        # Use the current frame's actual attack directions to train the model
        actual_attack_dirs = []
        if left_spawns > 0:
            actual_attack_dirs.append("LEFT")
        if right_spawns > 0:
            actual_attack_dirs.append("RIGHT")
            
        self.attack_predictor.learn_from_actual(actual_attack_dirs)
        
        # Log prediction accuracy periodically
        if hasattr(self.attack_predictor, 'total_predictions') and self.attack_predictor.total_predictions > 0 and self.attack_predictor.total_predictions % 10 == 0:
            accuracy = self.attack_predictor.correct_predictions / self.attack_predictor.total_predictions
            gamelib.debug_write(f"Neural network prediction accuracy: {accuracy:.2f} ({self.attack_predictor.correct_predictions}/{self.attack_predictor.total_predictions})")

    def plan_kamikaze_attack(self, game_state) -> int:
        # check if enemy has a turret on the top right corner
        attackRight = True
        # determine if there are weak points on the right side
        rightScore = 0
        leftScore = 0
        for [x,y] in self.right_kamikaze_attack_locations:
            if game_state.contains_stationary_unit([x,y]):
                rightScore += 1
        for [x,y] in self.left_kamikaze_attack_locations:
            if game_state.contains_stationary_unit([x,y]):
                leftScore += 1

        if rightScore < leftScore:
            attackRight = True
        else:
            attackRight = False

        if attackRight:
            game_state.attempt_remove([[26,12],[26,13]])
            # self.portToOpen.append([26,13])
            # self.portOpened = True
            return RIGHT_KAMIKAZE
        else:
            game_state.attempt_remove([[1,12],[1,13],[3,10]])

            return LEFT_KAMIKAZE

    def kamikaze_attack(self, game_state):
        if self.kamikaze_attack_location == RIGHT_KAMIKAZE:
            game_state.attempt_spawn(INTERCEPTOR, [24,10], 4)
            game_state.attempt_spawn(SCOUT, [13,0], game_state.number_affordable(SCOUT))
        else:
            game_state.attempt_spawn(INTERCEPTOR, [3,10], 4)
            game_state.attempt_spawn(SCOUT, [14,0], game_state.number_affordable(SCOUT))

        if game_state.game_map[4,12] and game_state.game_map[4,12][0].unit_type == WALL:
                game_state.attempt_remove([4,12])
        
   

if __name__ == "__main__":
    algo = AlgoStrategy()
    algo.start()