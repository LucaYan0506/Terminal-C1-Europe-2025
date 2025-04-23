import gamelib
import random
import math
import warnings
from sys import maxsize
import json
import copy


"""
Most of the algo code you write will be in this file unless you create new
modules yourself. Start by modifying the 'on_turn' function.

Advanced strategy tips: 

  - You can analyze action frames by modifying on_action_frame function

  - The GameState.map object can be manually manipulated to create hypothetical 
  board states. Though, we recommended making a copy of the map to preserve 
  the actual current map state.
"""

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
        
        # Tracking for opening prediction accuracy
        self.opening_predictions = []  # Will store tuples of (turn, prediction, actual_attack_dir)
        self.last_prediction = 0
        self.last_actual_attack = 0
        self.prediction_correct_count = 0
        self.prediction_total_count = 0

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

        # Use predict_opening to analyze enemy layout
        

        if game_state.turn_number < 1:
            self.early_game(game_state)
        else:
            self.mid_game(game_state)

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

    def mid_game(self, game_state):
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

            # Use both traditional attack detection and opening prediction
            predicted_attack_directions = self.check_potential_enemy_attack(game_state)


            opening_prediction = self.predict_opening(game_state)
            self.last_prediction = opening_prediction
            
            if opening_prediction != 0:
                direction = "LEFT" if opening_prediction == -1 else "RIGHT"
                predicted_attack_directions.append(direction)
                
                gamelib.debug_write(f"Turn {game_state.turn_number}: Predicted enemy attack from {direction} side based on opening analysis")
            
            # Log prediction accuracy if we have enough data
            if self.prediction_total_count >= 5:
                accuracy = self.prediction_correct_count / self.prediction_total_count
                gamelib.debug_write(f"Current opening prediction accuracy: {accuracy:.2f}")
            
            # Use opening prediction if traditional method didn't find anything
            # if not predicted_attack_directions and game_state.turn_number >= 3:
            #     opening_prediction = self.predict_opening(game_state)
            #     if opening_prediction == -1:
            #         predicted_attack_directions.append("LEFT")
            #         gamelib.debug_write("Reinforcing LEFT side based on opening prediction")
            #     elif opening_prediction == 1:
            #         predicted_attack_directions.append("RIGHT")
            #         gamelib.debug_write("Reinforcing RIGHT side based on opening prediction")
            
            # Handle right side attack detection
            if "RIGHT" in predicted_attack_directions:
                game_state.attempt_spawn(TURRET, [[22,11],[21,10],[20,9],[23,9]])
                self.turret_locations['RIGHT'] += [[22,11],[21,10],[20,9],[23,9]]
                
                self.removeWall(game_state,[[20,9],[21,10]])
                self.removeWall(game_state, [[25,13],[26,13]])

                newWall = [[24,10],[26,12]]
                for wall in newWall:
                    if wall not in self.firstLineWall['RIGHT']:
                        self.firstLineWall['RIGHT'].append(wall)
                
                newWall = [[22,12],[23,12]]
                for wall in newWall:
                    if wall not in self.moreWalls:
                        self.moreWalls.append(wall)

                self.removeWall(game_state, [[24,13]])
            
            # Handle left side attack detection
            # if "LEFT" in predicted_attack_directions:
            #     game_state.attempt_spawn(TURRET, [[6,13],[6,12],[2,13]])
            #     # Add to turret locations if not already there
            #     for loc in [[6,13],[6,12],[2,13]]:
            #         if loc not in self.turret_locations['LEFT']:
            #             self.turret_locations['LEFT'].append(loc)
                
            #     # Strengthen left wall
            #     newWall = [[1,12],[1,13]]
            #     for wall in newWall:
            #         if wall not in self.firstLineWall['LEFT']:
            #             self.firstLineWall['LEFT'].append(wall)

            # Build and update defenses
            self.build_tower(game_state, predicted_attack_directions)
            self.build_wall(game_state, predicted_attack_directions)
            self.build_support(game_state)
            game_state.attempt_spawn(WALL, self.moreWalls)
            game_state.attempt_upgrade(self.moreWalls)

            # Dynamically adjust defenses based on changing attack patterns
            # self.adjust_defenses_dynamically(game_state)

            self.update_tower(game_state, predicted_attack_directions)
            self.update_wall(game_state, predicted_attack_directions)
            self.update_support(game_state)

            self.spawnInterceptor(game_state, predicted_attack_directions)
        
        if game_state.get_resource(MP) > 10 and self.firstLineWallIsFull(game_state):
            self.will_kamikaze_attack = True
            self.kamikaze_attack_location = self.plan_kamikaze_attack(game_state)
        else:
            # replace wall only when we are not planning to attack
            self.replaceWall(game_state)

        for x, y in self.moreSupportLocations:
            game_state.attempt_spawn(SUPPORT, [x,y])
            game_state.attempt_upgrade([x,y])

    def build_tower(self, game_state, predicted_attack_directions = []):
        # If no attack directions predicted and not in early game, use opening prediction
        if not predicted_attack_directions and game_state.turn_number >= 3:
            opening_prediction = self.predict_opening(game_state)
            if opening_prediction == -1:
                predicted_attack_directions = ["LEFT"]
                gamelib.debug_write("Building towers on LEFT based on opening prediction")
            elif opening_prediction == 1:
                predicted_attack_directions = ["RIGHT"]
                gamelib.debug_write("Building towers on RIGHT based on opening prediction")

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
            self.removeWall(game_state, [[25,13],[26,13]])

            newWall = [[24,10],[26,12]]
            for wall in newWall:
                if wall not in self.firstLineWall['RIGHT']:
                    self.firstLineWall['RIGHT'].append(wall)
            
            newWall = [[22,12],[23,12]]
            for wall in newWall:
                if wall not in self.moreWalls:
                    self.moreWalls.append(wall)

        # If no attack directions predicted and not in early game, use opening prediction
        if not predicted_attack_directions and game_state.turn_number >= 3:
            opening_prediction = self.predict_opening(game_state)
            if opening_prediction == -1:
                predicted_attack_directions = ["LEFT"]
            elif opening_prediction == 1:
                predicted_attack_directions = ["RIGHT"]

        # build turret where we predict an attack to happen first
        for dir in predicted_attack_directions:
            game_state.attempt_spawn(WALL, self.firstLineWall[dir])

        # build the rest
        for dir in self.firstLineWall.keys():
            if dir not in predicted_attack_directions:
                game_state.attempt_spawn(WALL, self.firstLineWall[dir])

    def removeWall(self,game_state, walls):
            for wall in walls:
                if len(game_state.game_map[wall[0],wall[1]]) > 0 and game_state.game_map[wall[0],wall[1]][0].unit_type == WALL:
                    game_state.attempt_remove(wall)
                for dir in self.firstLineWall.keys():
                    if wall in self.firstLineWall[dir]:
                        self.firstLineWall[dir].remove(wall)
                if wall in self.moreWalls:
                        self.moreWalls.remove(wall)

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
        if enemyMP >= base + 8.0:
            interceptor_num = 5 + int(enemy_support_counter / 3)
            interceptor_num += int((enemyMP - base - 5.0) // 4)
        elif enemyMP >= base + 5.0:
            interceptor_num = 5 + int(enemy_support_counter / 3)
        elif enemyMP >= base + 3.0:
            interceptor_num = 4 + int(enemy_support_counter / 3)
        elif enemyMP >= base:
            interceptor_num = 2 + int(enemy_support_counter / 3)
        elif enemy_support_counter > 0 and enemyMP >= 5:
            interceptor_num = 1 + int(enemy_support_counter / 3)

        # if nothing is predicted by traditional methods, check opening prediction
        if len(predicted_attack_directions) == 0 and game_state.turn_number >= 3:
            opening_prediction = self.predict_opening(game_state)
            if opening_prediction == -1:
                predicted_attack_directions = ["LEFT"]
                gamelib.debug_write("Spawning interceptors on LEFT based on opening analysis")
            elif opening_prediction == 1:
                predicted_attack_directions = ["RIGHT"]
                gamelib.debug_write("Spawning interceptors on RIGHT based on opening analysis")
            else:
                # If still no prediction, spawn on both sides
                predicted_attack_directions = ["LEFT", "RIGHT"]
        elif len(predicted_attack_directions) == 0:
            # Default to both sides if we don't have a prediction and early in game
            predicted_attack_directions = ["LEFT", "RIGHT"]
        
        # Calculate interceptors per side
        interceptor_num = max(interceptor_num // len(predicted_attack_directions), 1)
        
        # Deploy interceptors based on predictions
        for dir in predicted_attack_directions:
            if dir == "LEFT":
                game_state.attempt_spawn(INTERCEPTOR, [5, 8], interceptor_num//2)
                game_state.attempt_spawn(INTERCEPTOR, [7, 6], interceptor_num//2)
            elif dir == "RIGHT":
                game_state.attempt_spawn(INTERCEPTOR, [22, 8], interceptor_num//2)
                game_state.attempt_spawn(INTERCEPTOR, [20, 6], interceptor_num//2)

        # If right wall is weak, add extra interceptors on that side regardless of prediction
        if self.top_right_wall_weak(game_state):
            extra_interceptors = int(enemy_support_counter / 3) + 1
            game_state.attempt_spawn(INTERCEPTOR, [22, 8], extra_interceptors)
            gamelib.debug_write(f"Added {extra_interceptors} extra interceptors on RIGHT due to weak wall")

    def replaceWall(self, game_state):    
        allWalls = copy.deepcopy(self.moreWalls) 
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
        
        # Check based on wall removals (traditional method)
        if (1,14) in self.enemy_append_removal_unit or (2,14) in self.enemy_append_removal_unit:
            game_state.attempt_remove([[1,13],[2,13]])
            self.removeWall(game_state, [[1,13],[2,13]])
            predicted_attack_directions.append("LEFT")

        if (25,14) in self.enemy_append_removal_unit or (26,14) in self.enemy_append_removal_unit:
            game_state.attempt_remove([[25,13],[26,13]])
            self.removeWall(game_state, [[25,13],[26,13]])
            predicted_attack_directions.append("RIGHT")
        
        # If we didn't detect removals but have observed the game for a few turns,
        # use opening prediction as a backup
        if not predicted_attack_directions and game_state.turn_number >= 3:
            opening_prediction = self.predict_opening(game_state)
            if opening_prediction != 0:  # If there's a clear prediction
                if opening_prediction == -1:
                    predicted_attack_directions.append("LEFT")
                else:
                    predicted_attack_directions.append("RIGHT")
                gamelib.debug_write(f"Using opening prediction: {opening_prediction}")

        return predicted_attack_directions


    def on_action_frame(self, turn_string):
        frame_data = json.loads(turn_string)
        p2_units = frame_data["p2Units"]
        
        # Track removals
        for unit in p2_units[6]:
            x = unit[0]
            y = unit[1]
            self.enemy_append_removal_unit.add((x,y))
        
        # Track enemy attack directions for prediction accuracy
        left_attack = False
        right_attack = False
        
        # Look for enemy mobile units (SCOUT=3, DEMOLISHER=4, INTERCEPTOR=5)
        for unit_type in ["3", "4", "5"]:
            if unit_type in p2_units:
                for unit in p2_units[unit_type]:
                    x, y = unit[0], unit[1]
                    if y >= 14:  # Near our side
                        if x < 14:
                            left_attack = True
                        else:
                            right_attack = True
        
        # Record the actual attack direction
        if left_attack and right_attack:
            actual_attack = 0  # both sides
        elif left_attack:
            actual_attack = -1  # left side
        elif right_attack:
            actual_attack = 1  # right side
        else:
            actual_attack = 0  # no attack
        
        # Update tracking of prediction accuracy
        if self.last_prediction != 0 and actual_attack != 0:
            self.last_actual_attack = actual_attack
            self.prediction_total_count += 1
            
            # Check if prediction was correct
            if (self.last_prediction < 0 and actual_attack < 0) or (self.last_prediction > 0 and actual_attack > 0):
                self.prediction_correct_count += 1
                if self.prediction_total_count % 5 == 0:
                    accuracy = self.prediction_correct_count / self.prediction_total_count
                    gamelib.debug_write(f"Opening prediction accuracy: {accuracy:.2f} ({self.prediction_correct_count}/{self.prediction_total_count})")

    def plan_kamikaze_attack(self, game_state) -> int:
        # Use opening prediction to help determine attack side
        opening_prediction = self.predict_opening(game_state)
        
        # By default, check weak points
        attackRight = True
        rightScore = 0
        leftScore = 0
        
        # Check number of enemy structures on each side
        for [x,y] in self.right_kamikaze_attack_locations:
            if game_state.contains_stationary_unit([x,y]):
                rightScore += 1
        for [x,y] in self.left_kamikaze_attack_locations:
            if game_state.contains_stationary_unit([x,y]):
                leftScore += 1
        
        # If opening prediction suggests a side attack, give it more weight
        if opening_prediction != 0:
            if opening_prediction == -1:  # Enemy likely to attack left
                # Attack where enemy is weaker (their right, our left)
                leftScore -= 3  # Lower score means more favorable to attack
                gamelib.debug_write("Opening prediction favors attacking LEFT")
            else:  # Enemy likely to attack right
                # Attack where enemy is weaker (their left, our right)
                rightScore -= 3
                gamelib.debug_write("Opening prediction favors attacking RIGHT")
        
        # Compare scores and decide
        if rightScore < leftScore:
            attackRight = True
            gamelib.debug_write(f"Attacking RIGHT side (scores: Right={rightScore}, Left={leftScore})")
        else:
            attackRight = False
            gamelib.debug_write(f"Attacking LEFT side (scores: Right={rightScore}, Left={leftScore})")

        if attackRight:
            game_state.attempt_remove([[26,12],[26,13], [24,10]])
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
        
    
    def predict_opening(self, game_state):
        """
        Looks at opponent layout to determine which side attacks will most likely come from.
        Returns -1 for left side, 1 for right side, and 0 if unknown/equally likely
        """
        left_edges = game_state.game_map.get_edge_locations(game_state.game_map.TOP_LEFT)
        right_edges = game_state.game_map.get_edge_locations(game_state.game_map.TOP_RIGHT)

        left_exit_side = 0
        right_exit_side = 0

        # run through all the spawn edges starting with the center most and see where the path leads
        for left, right in zip(left_edges, right_edges):
            if left_exit_side == 0 and not game_state.contains_stationary_unit(left):
                path = game_state.find_path_to_edge(left)
                if path[-1][1] >= game_state.game_map.HALF_ARENA:
                    for x, y in path[::-1]:
                        if y == game_state.game_map.HALF_ARENA:
                            left_exit_side = -1 if x < game_state.HALF_ARENA else 1
                            break

            if right_exit_side == 0 and not game_state.contains_stationary_unit(right):
                path = game_state.find_path_to_edge(right)
                if path[-1][1] >= game_state.game_map.HALF_ARENA:
                    for x, y in path[::-1]:
                        if y == game_state.game_map.HALF_ARENA:
                            right_exit_side = -1 if x < game_state.HALF_ARENA else 1
                            break

        # if both left + right edges exit to the same side
        # or if exactly one edge does not exit we are done
        if left_exit_side == right_exit_side and left_exit_side + right_exit_side != 0:
            return left_exit_side if left_exit_side != 0 else right_exit_side

        # otherwise: sum structure costs of both sides and compare; if there is
        # a large difference the more expensive side is more likely to attack
        left_coords = [(x, y) for x in range(14) for y in range(14, 14 + x)]
        right_coords = [(x, y) for x in range(14, 28) for y in range(14, 42 - x)]

        left_price = 0
        right_price = 0
        for (lx, ly), (rx, ry) in zip(left_coords, right_coords):
            units = game_state.game_map[lx, ly]
            for unit in units:
                left_price += unit.cost[game_state.SP]
            units = game_state.game_map[rx, ry]
            for unit in units:
                right_price += unit.cost[game_state.SP]

        if left_price >= right_price + 10:
            return -1
        elif right_price >= left_price + 10:
            return 1

        # last resort: follow the structures along the corners and check symmetry
        left_corner = [(x, 14) for x in range(14)]
        right_corner = [(x, 14) for x in range(27, 13, -1)]

        for (lx, ly), (rx, ry) in zip(left_corner, right_corner):
            left_struct = game_state.game_map[lx, ly][0] if game_state.game_map[lx, ly] else None
            right_struct = game_state.game_map[rx, ry][0] if game_state.game_map[rx, ry] else None
            if left_struct is None and right_struct is None:
                return 0
            if left_struct is None and right_struct is not None:
                return -1
            if left_struct is not None and right_struct is None:
                return 1
            if not left_struct.upgraded and right_struct.upgraded:
                return -1
            if left_struct.upgraded and not right_struct.upgraded:
                return 1
            if left_struct in self.enemy_append_removal_unit and not right_struct in self.enemy_append_removal_unit:
                return -1
            if not left_struct in self.enemy_append_removal_unit and right_struct in self.enemy_append_removal_unit:
                return 1

        # theoretically possible to reach this code (if top edge is perfectly symmetrical with no holes)
        # but in practice will probably never happen
        return 0
        
    def adjust_defenses_dynamically(self, game_state):
        """
        Dynamically adjusts defense strategy based on the current state of the game
        and predictions about enemy attack patterns
        """
        # Only run this adjustment after a few turns to gather enough data
        if game_state.turn_number < 5:
            return
        
        # Use opening prediction to detect likely attack direction
        opening_prediction = self.predict_opening(game_state)
        
        # Count existing defenses on each side
        left_defense_count = 0
        right_defense_count = 0
        
        # Define defense regions
        left_region = [(x,y) for x in range(0,10) for y in range(8,14)]
        right_region = [(x,y) for x in range(18,28) for y in range(8,14)]
        
        # Count turrets and walls on each side
        for x,y in left_region:
            if game_state.contains_stationary_unit([x,y]):
                unit = game_state.game_map[x,y][0]
                if unit.player_index == 0 and unit.unit_type in [WALL, TURRET]:
                    left_defense_count += 1
                
        for x,y in right_region:
            if game_state.contains_stationary_unit([x,y]):
                unit = game_state.game_map[x,y][0]
                if unit.player_index == 0 and unit.unit_type in [WALL, TURRET]:
                    right_defense_count += 1
        
        # Determine imbalance in defenses
        imbalance = abs(left_defense_count - right_defense_count)
        
        # If significant imbalance exists (more than 5 units difference)
        if imbalance > 5:
            # If we have more defenses on left but prediction suggests right attack
            if left_defense_count > right_defense_count and opening_prediction == 1:
                gamelib.debug_write(f"Defense imbalance detected: LEFT={left_defense_count}, RIGHT={right_defense_count}")
                gamelib.debug_write("Reinforcing RIGHT side based on opening prediction")
                
                # Add turrets to right side
                additional_turrets = [[22,12], [21,12]]
                for turret_loc in additional_turrets:
                    game_state.attempt_spawn(TURRET, [turret_loc])
                    if turret_loc not in self.turret_locations['RIGHT']:
                        self.turret_locations['RIGHT'].append(turret_loc)
            
            # If we have more defenses on right but prediction suggests left attack
            elif right_defense_count > left_defense_count and opening_prediction == -1:
                gamelib.debug_write(f"Defense imbalance detected: LEFT={left_defense_count}, RIGHT={right_defense_count}")
                gamelib.debug_write("Reinforcing LEFT side based on opening prediction")
                
                # Add turrets to left side
                additional_turrets = [[3,11], [4,10], [5,9]]
                for turret_loc in additional_turrets:
                    game_state.attempt_spawn(TURRET, [turret_loc])
                    if turret_loc not in self.turret_locations['LEFT']:
                        self.turret_locations['LEFT'].append(turret_loc)
        
        # Check if enemy's attack pattern has changed from previous observations
        # by looking at recent wall removals
        if game_state.turn_number > 10:
            recent_left_removals = sum(1 for x,y in self.enemy_append_removal_unit if x < 14 and y >= 13)
            recent_right_removals = sum(1 for x,y in self.enemy_append_removal_unit if x >= 14 and y >= 13)
            
            # If removals indicate a shift in attack pattern
            if recent_left_removals > recent_right_removals + 2:
                gamelib.debug_write(f"Attack pattern shift detected: More recent removals on LEFT ({recent_left_removals} vs {recent_right_removals})")
                # Strengthen left defenses
                game_state.attempt_spawn(TURRET, [[3,12], [2,13]])
            elif recent_right_removals > recent_left_removals + 2:
                gamelib.debug_write(f"Attack pattern shift detected: More recent removals on RIGHT ({recent_right_removals} vs {recent_left_removals})")
                # Strengthen right defenses
                game_state.attempt_spawn(TURRET, [[24,12], [25,11]])

if __name__ == "__main__":
    algo = AlgoStrategy()
    algo.start()