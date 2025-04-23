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

            predicted_attack_directions = self.predict_attack_side(game_state)
            if "RIGHT" in predicted_attack_directions:
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
            if self.detect_enemy_trapped(game_state):
                predicted_attack_directions = ['TRAPPED']

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

    def build_tower(self, game_state, predicted_attack_directions = []):
        # build turret where we predict an attack to happen first
        for dir in predicted_attack_directions:
            if dir in self.turret_locations.keys():
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


        # build turret where we predict an attack to happen first
        for dir in predicted_attack_directions:
            if dir in self.firstLineWall.keys():
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
            if dir in self.firstLineWall.keys():
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
            if dir in self.turret_locations.keys():
                game_state.attempt_upgrade(self.turret_locations[dir])

        # upgrade the rest
        for dir in self.turret_locations.keys():
            if dir not in predicted_attack_directions:
                game_state.attempt_upgrade(self.turret_locations[dir])

    def spawnInterceptor(self, game_state, predicted_attack_directions):
        if "TRAPPED" in predicted_attack_directions:
            return

        enemyMP = game_state.get_resource(MP,1)
        # gamelib.debug_write('enemyMP {}'.format(enemyMP))
        base = 7.0

        enemy_support_counter = min(self.count_enemy_support(game_state),6)
        
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
        if (1,14) in self.enemy_append_removal_unit or (2,14) in self.enemy_append_removal_unit:
            game_state.attempt_remove([[1,13],[2,13]])
            self.removeWall(game_state, [[1,13],[2,13]])
            predicted_attack_directions.append("LEFT")

        if (25,14) in self.enemy_append_removal_unit or (26,14) in self.enemy_append_removal_unit:
            game_state.attempt_remove([[25,13],[26,13]])
            self.removeWall(game_state, [[25,13],[26,13]])
            predicted_attack_directions.append("RIGHT")

        return predicted_attack_directions
    
    def predict_attack_side(self, game_state):
        """
        Looks at opponent layout to determine which side attacks will most likely come from.
        Returns -1 for left side, 1 for right side, and 0 if unknown/equally likely
        """
        newAns = {
            -1: ['LEFT'],
            0: ['LEFT','RIGHT'],
            1: ['RIGHT'],
        }

        left_edges = [(13 - i, 27 - i) for i in range(14)]
        right_edges = [(i + 14, 27 - i) for i in range(14)]

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
            return newAns[left_exit_side] if left_exit_side != 0 else newAns[right_exit_side]

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
            return newAns[1]
        elif right_price >= left_price + 10:
            return newAns[-1]

        # last resort: follow the structures along the corners and check symmetry
        left_corner = [(x, 14) for x in range(14)]
        right_corner = [(x, 14) for x in range(27, 13, -1)]

        for (lx, ly), (rx, ry) in zip(left_corner, right_corner):
            left_struct = game_state.game_map[lx, ly][0] if game_state.game_map[lx, ly] else None
            right_struct = game_state.game_map[rx, ry][0] if game_state.game_map[rx, ry] else None
            if left_struct is None and right_struct is None:
                return newAns[0]
            if left_struct is None and right_struct is not None:
                return newAns[-1]
            if left_struct is not None and right_struct is None:
                return newAns[1]
            if not left_struct.upgraded and right_struct.upgraded:
                return newAns[-1]
            if left_struct.upgraded and not right_struct.upgraded:
                return newAns[1]
            if left_struct.pending_removal and not right_struct.pending_removal:
                return newAns[-1]
            if not left_struct.pending_removal and right_struct.pending_removal:
                return newAns[1]

        # theoretically possible to reach this code (if top edge is perfectly symmetrical with no holes)
        # but in practice will probably never happen
        return newAns[0]
    
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
        # if is_trapped:
            # gamelib.debug_write("Enemy detected as trapped! No open spawn positions and no pending wall removals.")
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
        p2_units = frame_data["p2Units"]
        for unit in p2_units[6]:
            x = unit[0]
            y = unit[1]
            self.enemy_append_removal_unit.add((x,y))
                

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
            game_state.attempt_remove([[26,12],[26,13], [24,10]])
            # check if health of wall at [24,13] is less than 50
            if game_state.game_map[24,13] and game_state.game_map[24,13][0].unit_type == WALL:
                if game_state.game_map[24,13][0].health < 50:
                    game_state.attempt_remove([24,13])

            return RIGHT_KAMIKAZE
        else:
            game_state.attempt_remove([[1,12],[1,13],[3,10]])
            # check if health of wall at [3,13] is less than 50
            if game_state.game_map[3,13] and game_state.game_map[3,13][0].unit_type == WALL:
                if game_state.game_map[3,13][0].health < 50:
                    game_state.attempt_remove([3,13])

            return LEFT_KAMIKAZE

    def kamikaze_attack(self, game_state):
        if self.kamikaze_attack_location == RIGHT_KAMIKAZE:
            game_state.attempt_spawn(WALL, [5,9]) # block the left side
            game_state.attempt_remove([5,9])

            game_state.attempt_spawn(WALL, [24,13]) # reinforce
            game_state.attempt_spawn(WALL, [22,13])
            game_state.attempt_spawn(INTERCEPTOR, [24,10], 4)
            game_state.attempt_spawn(SCOUT, [13,0], game_state.number_affordable(SCOUT))
        else:
            game_state.attempt_spawn(WALL, [22,9]) # block the left side
            game_state.attempt_remove([22,9])

            game_state.attempt_spawn(WALL, [3,13])
            game_state.attempt_spawn(WALL, [4,13])
            game_state.attempt_spawn(INTERCEPTOR, [3,10], 4)
            game_state.attempt_spawn(SCOUT, [14,0], game_state.number_affordable(SCOUT))

        if game_state.game_map[4,12] and game_state.game_map[4,12][0].unit_type == WALL:
                game_state.attempt_remove([4,12])
        
   

if __name__ == "__main__":
    algo = AlgoStrategy()
    algo.start()